"""Read-side owners for the .source/ tree and the entry cache.

Three registries, all keyed by resolved root path:

    SourceRegistry  — scans code/*/source.json, indexes by class name
    TreeRegistry    — scans snapshots/ directory names, reads tree.json lazily
    EntryRegistry   — scans cache roots for *.params.json, owns EntryInfo/GroupInfo

All use a per-path instance cache so repeated calls with the same root
return the same object without rescanning.  Call :meth:`reload` on an
instance to re-scan in place; held references remain valid.

Typical usage::

    src = SourceRegistry.instance()  # uses get_config().path_registry
    src = SourceRegistry.instance(path)  # explicit root
    src.reload()

    tree = TreeRegistry.instance()
    tree = TreeRegistry.instance(path)
    tree.reload()

    reg = EntryRegistry.instance()
    reg.reload()
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pygeodata.config import get_config
from pygeodata.paths import CodeRegistryResolver, TreeRegistryResolver
from pygeodata.registry_types import CodeState, EntryRecord, GroupRecord, TreeSnapshot
from pygeodata.tracked_object import TrackedObject


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


class EntryRegistry:
    """Per-root registry of cache entries, keyed by state_hash.

    The single place that scans for *.hash.json files.  Reads each via
    :meth:`EntryRecord.from_hash_path`, assembles typed :class:`EntryRecord`
    and :class:`GroupRecord` dicts, and owns the lightweight disk cache
    ('.entry_registry_cache.json').

    state_hash is the unique key.  On collision:
    - If all identity fields match → silently deduplicate (same entry in two locations).
    - If any field differs → raise ValueError (hash collision with divergent data is a
      serious integrity problem that must surface immediately).

    Browser display fields (rows, spec strings, linked entries, primary file) are
    NOT populated here — that enrichment is done by
    :func:`pygeodata.registry_browser.entry_catalog.discover_entries`.

    Instances are cached in :attr:`_instances` by resolved registry root.
    Call :meth:`reload` to re-scan; held references stay valid.
    """

    _instances: dict[Path, EntryRegistry] | None = None
    _CACHE_FILE = '.entry_registry_cache.json'

    def __init__(self, registry_root: Path) -> None:
        self._root = registry_root
        self.records: dict[str, EntryRecord] = {}
        self.groups: dict[str, GroupRecord] = {}
        self.diagnostics: dict = {}
        self._reload()

    @classmethod
    def instance(cls, path: Path | str | None = None) -> EntryRegistry:
        """Return the cached instance for *path*, creating it if necessary."""
        if cls._instances is None:
            cls._instances = {}
        p = Path(path).resolve() if path else Path(get_config().path_registry).resolve()
        if p not in cls._instances:
            cls._instances[p] = cls(p)
        return cls._instances[p]

    def reload(self) -> None:
        """Rescan all cache roots and replace records/groups/diagnostics in place."""
        self._reload()

    def _cache_path(self) -> Path:
        return self._root / self._CACHE_FILE

    def _load_disk_cache(self) -> tuple[dict[str, dict], dict[str, float]]:
        path = self._cache_path()
        if not path.exists():
            return {}, {}
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            return data.get('records', {}), data.get('mtimes', {})
        except (OSError, json.JSONDecodeError):
            return {}, {}

    def _save_disk_cache(self, records: dict[str, dict], mtimes: dict[str, float]) -> None:
        with contextlib.suppress(OSError):
            self._cache_path().write_text(
                json.dumps({'records': records, 'mtimes': mtimes}, separators=(',', ':')),
                encoding='utf-8',
            )

    @staticmethod
    def _find_hash_paths() -> list[Path]:
        cfg = get_config()
        cache_roots = [cfg.path_cache, cfg.path_figures]
        return sorted(
            {p for root in cache_roots if root.exists() for p in root.rglob('*.hash.json')},
        )

    @staticmethod
    def _mtime(hash_path: Path) -> float:
        try:
            return hash_path.stat().st_mtime
        except OSError:
            return 0.0

    def _reload(self) -> None:
        hash_paths = self._find_hash_paths()
        cached_records, cached_mtimes = self._load_disk_cache()

        partial: list[EntryRecord | None] = [None] * len(hash_paths)
        new_cache_records: dict[str, dict] = {}
        new_cache_mtimes: dict[str, float] = {}

        def _process(i: int, p: Path) -> tuple[int, EntryRecord | None]:
            key = str(p.resolve())
            mtime = self._mtime(p)
            if key in cached_records and cached_mtimes.get(key) == mtime:
                try:
                    return i, EntryRecord(**cached_records[key])
                except (KeyError, TypeError):
                    pass
            record = EntryRecord.from_hash_path(p)
            if record is not None:
                new_cache_records[key] = dataclasses.asdict(record)
                new_cache_mtimes[key] = mtime
            return i, record

        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(_process, i, p): i for i, p in enumerate(hash_paths)}
            for future in as_completed(futures):
                idx, record = future.result()
                partial[idx] = record
                key = str(hash_paths[idx].resolve())
                if key in cached_records and key not in new_cache_records:
                    new_cache_records[key] = cached_records[key]
                    new_cache_mtimes[key] = cached_mtimes[key]

        self._save_disk_cache(new_cache_records, new_cache_mtimes)

        # Assemble: keyed by state_hash, collision detection, dep_hash_stale, groups
        records: dict[str, EntryRecord] = {}
        groups: dict[str, GroupRecord] = {}
        diagnostics: dict = {
            'missing_state_hash': [],
            'scanned_hash_paths': len(hash_paths),
            'created_records': 0,
        }

        _identity_fields = (
            'class_name',
            'instance_hash',
            'dep_hash',
            'co_output_hashes',
            'object_type',
            'format_version',
        )

        for i, rec in enumerate(partial):
            if rec is None:
                diagnostics['missing_state_hash'].append(str(hash_paths[i]))
                continue

            key = rec.state_hash
            if key is None:
                diagnostics['missing_state_hash'].append(rec.hash_path or str(hash_paths[i]))
                continue

            if key in records:
                existing = records[key]
                if all(getattr(rec, f) == getattr(existing, f) for f in _identity_fields):
                    continue  # exact duplicate — silently skip
                raise ValueError(
                    f'state_hash collision with divergent data for hash {rec.state_hash!r}:\n'
                    f'  differing fields: {[f for f in _identity_fields if getattr(rec, f) != getattr(existing, f)]}',
                )

            if rec.dep_hash:
                obj_cls = TrackedObject.find_object_class(rec.class_name)
                if obj_cls is not None:
                    rec.dep_hash_stale = obj_cls.get_dependency_tree_hash() != rec.dep_hash

            records[key] = rec
            groups.setdefault(
                rec.class_name,
                GroupRecord(class_name=rec.class_name, object_type=rec.object_type),
            ).state_hashes.append(key)

        diagnostics['created_records'] = len(records)
        self.records = records
        self.groups = groups
        self.diagnostics = diagnostics
