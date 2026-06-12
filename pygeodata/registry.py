"""Read-side owners for the .source/ tree.

Two registries, both keyed by resolved root path:

    SourceRegistry  — scans code/*/source.json, indexes by class name
    TreeRegistry    — scans snapshots/ directory names, reads tree.json lazily

Both use a per-path instance cache so repeated calls with the same root
return the same object without rescanning.  Call :meth:`reload` on an
instance to re-scan in place; held references remain valid.

Typical usage::

    src = SourceRegistry.instance()  # uses get_config().path_registry
    src = SourceRegistry.instance(path)  # explicit root
    src.reload()

    tree = TreeRegistry.instance()
    tree = TreeRegistry.instance(path)
    tree.reload()
"""

from __future__ import annotations

from pathlib import Path

from pygeodata.config import get_config
from pygeodata.paths import CodeRegistryResolver, TreeRegistryResolver
from pygeodata.registry_types import CodeState, TreeSnapshot


class SourceRegistry:
    """Per-root index of code states, scanned from code/*/source.json.

    Instances are cached in :attr:`_instances` by resolved root path.
    :meth:`reload` re-scans and atomically replaces the state dict on the
    existing instance, so held references stay valid.
    """

    _instances: dict[Path, SourceRegistry] | None = None

    def __init__(self, registry_root: Path) -> None:
        self._root = registry_root
        self._states: dict[str, list[CodeState]] = {}
        self._hash_index: dict[str, CodeState] = {}
        self._load(self._scan())

    @classmethod
    def instance(cls, path: Path | str | None = None) -> SourceRegistry:
        """Return the cached instance for *path*, creating it if necessary."""
        if cls._instances is None:
            cls._instances = {}
        p = Path(path).resolve() if path else Path(get_config().path_registry).resolve()
        if p not in cls._instances:
            cls._instances[p] = cls(p)
        return cls._instances[p]

    def reload(self) -> None:
        """Rescan the root and atomically replace the cached snapshot dict."""
        self._load(self._scan())

    def _load(self, states: dict[str, list[CodeState]]) -> None:
        self._states = states
        self._hash_index = {s.source_hash: s for sl in states.values() for s in sl}

    def _scan(self) -> dict[str, list[CodeState]]:
        code_root = self._root / 'code'
        raw: dict[str, list[CodeState]] = {}
        if not code_root.exists():
            return raw
        for meta_path in code_root.glob('*/source.json'):
            try:
                state = CodeState.from_file(meta_path)
            except (KeyError, ValueError):
                continue
            if not state.class_name:
                continue
            raw.setdefault(state.class_name, []).append(state)
        return raw

    @property
    def class_names(self) -> list[str]:
        """All class names that have at least one snapshot."""
        return list(self._states.keys())

    def get_states(self, class_name: str) -> list[CodeState]:
        """All snapshots for class_name, or empty list if unknown."""
        return self._states.get(class_name, [])

    def get_state_by_hash(self, source_hash: str) -> CodeState | None:
        """Return the CodeState for source_hash, or None if not found."""
        return self._hash_index.get(source_hash)

    def get_source(self, source_hash: str) -> str | None:
        """Return source.py text for source_hash, or None if absent."""
        path = CodeRegistryResolver.from_source_hash(source_hash).source_path
        return path.read_text(encoding='utf-8') if path.exists() else None

    def latest_for_class(self, class_name: str) -> CodeState | None:
        """Most-recent snapshot for class_name by registered_at, or None."""
        states = self._states.get(class_name)
        if not states:
            return None
        return max(states, key=lambda s: s.registered_at)

    def is_version_change(self, state: CodeState) -> bool:
        """True when snapshot is not the oldest registration for its class."""
        siblings = self._states.get(state.class_name, [])
        if len(siblings) <= 1:
            return False
        oldest = min(s.registered_at for s in siblings)
        return state.registered_at != oldest

    def hash_to_mtime(self, source_hash: str) -> str | None:
        """Return registered_at for source_hash, or None if not found."""
        s = self._hash_index.get(source_hash)
        return s.registered_at if s is not None else None

    def code_groups_dict(self) -> dict[str, list[dict]]:
        """Return code states grouped by class_name as plain dicts for API consumers."""
        return {
            class_name: [
                {
                    'source_hash': s.source_hash,
                    'mtime': s.registered_at,
                    'object_type': s.object_type,
                    'is_version_change': self.is_version_change(s),
                }
                for s in states
            ]
            for class_name, states in self._states.items()
        }


class TreeRegistry:
    """Per-root index of dependency tree snapshots, scanned from snapshots/.

    Dep hashes are collected at construction; tree.json content is read lazily
    on demand.  Instances are cached in :attr:`_instances` by resolved root
    path.  Call :meth:`reload` to re-scan after the filesystem changes.
    """

    _instances: dict[Path, TreeRegistry] | None = None

    def __init__(self, registry_root: Path) -> None:
        self._root = registry_root
        self._dep_hashes: set[str] = self._scan()

    @classmethod
    def instance(cls, path: Path | str | None = None) -> TreeRegistry:
        """Return the cached instance for *path*, creating it if necessary."""
        if cls._instances is None:
            cls._instances = {}
        p = Path(path).resolve() if path else Path(get_config().path_registry).resolve()
        if p not in cls._instances:
            cls._instances[p] = cls(p)
        return cls._instances[p]

    def reload(self) -> None:
        """Rescan the snapshots/ directory and atomically replace the index."""
        self._dep_hashes = self._scan()

    def _scan(self) -> set[str]:
        snapshots_root = self._root / 'snapshots'
        if not snapshots_root.exists():
            return set()
        return {p.name for p in snapshots_root.iterdir() if p.is_dir()}

    @property
    def dep_hashes(self) -> list[str]:
        """All known dep hashes."""
        return list(self._dep_hashes)

    def get_snapshot(self, dep_hash: str) -> TreeSnapshot | None:
        """Return a TreeSnapshot for dep_hash, or None if absent/invalid."""
        if dep_hash not in self._dep_hashes:
            return None
        path = TreeRegistryResolver.from_dep_tree_hash(dep_hash).tree_path
        if not path.exists():
            return None
        try:
            return TreeSnapshot.from_file(dep_hash, path)
        except (KeyError, ValueError, OSError):
            return None

    def get_nodes(self, dep_hash: str) -> dict[str, dict] | None:
        """Return the nodes dict for dep_hash, or None if tree is absent."""
        tree = self.get_snapshot(dep_hash)
        return tree.nodes if tree is not None else None

    def get_tree_path(self, dep_hash: str) -> Path:
        """Return the path to tree.json for dep_hash (may not exist)."""
        return TreeRegistryResolver.from_dep_tree_hash(dep_hash).tree_path

    def get_call_deps(self, dep_hash: str) -> list[str]:
        """Sorted direct call-dependency names for dep_hash, or empty list."""
        tree = self.get_snapshot(dep_hash)
        return tree.get_call_deps() if tree is not None else []

    def get_inheritance_deps(self, dep_hash: str) -> list[str]:
        """Sorted direct inheritance-dependency names for dep_hash, or empty list."""
        tree = self.get_snapshot(dep_hash)
        return tree.get_inheritance_deps() if tree is not None else []

    def find_by_class(self, class_name: str) -> str | None:
        """Return the dep_hash of the first tree where class_name is the root."""
        for dep_hash in self._dep_hashes:
            tree = self.get_snapshot(dep_hash)
            if tree is not None and class_name in tree.tree:
                return dep_hash
        return None
