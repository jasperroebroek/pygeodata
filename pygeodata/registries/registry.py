"""Read-side of the registry tree and the entry cache.

    SourceRegistry  — scans .source/code/*/source.json, indexes by class name
    TreeRegistry    — scans .source/snapshots/*/tree.json, indexes by dependency tree hash
    EntryRegistry   — index over the catalog of cache entries

Each is built once and never changes; build a new one to see newer data.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import msgspec

from pygeodata.catalog import store
from pygeodata.paths import (
    CodeRegistryPathConstructor,
    RegistryResolver,
    TreeRegistryPathConstructor,
)
from pygeodata.registries.registry_types import CodeState, TreeNode, TreeSnapshot

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from pygeodata.catalog.types import CatalogData, CatalogEntry, CatalogHeader
    from pygeodata.records import SpecRecord


def _resolve_prefix(keys: Iterable[str], prefix: str) -> str | None:
    matches = [k for k in keys if k.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


class SourceRegistry:
    """Index of code states, scanned from .source/code/*/source.json.

    States are indexed by class name and kept sorted oldest-first by
    registered_at.

    ``registry_root`` overrides the config registry path.  When omitted,
    defaults to ``get_config().path_registry``.
    """

    def __init__(self, registry_root: Path | None = None) -> None:
        self._registry_root = registry_root
        self._class_index: dict[str, list[CodeState]] = {}
        for meta_path in RegistryResolver(self._registry_root).glob_source_paths():
            try:
                state = CodeState.from_file(meta_path)
            except (OSError, msgspec.DecodeError):
                continue
            if not state.class_name:
                continue
            self._class_index.setdefault(state.class_name, []).append(state)

        for states in self._class_index.values():
            states.sort(key=lambda s: s.registered_at)

        self._hash_index = {s.source_hash: s for sl in self._class_index.values() for s in sl}

    @property
    def class_names(self) -> list[str]:
        """All class names that have at least one snapshot."""
        return list(self._class_index.keys())

    def get_states(self, class_name: str) -> list[CodeState]:
        """All snapshots for class_name, or empty list if unknown."""
        return self._class_index.get(class_name, [])

    def get_state_by_hash(self, source_hash: str) -> CodeState | None:
        """Return the CodeState for source_hash, or None if not found."""
        return self._hash_index.get(source_hash)

    def get_source(self, source_hash: str) -> str | None:
        """Return source.py text for source_hash, or None if absent."""
        path = CodeRegistryPathConstructor.from_source_hash(source_hash, self._registry_root).source_path
        return path.read_text(encoding='utf-8') if path.exists() else None

    def get_latest_state_for_class(self, class_name: str) -> CodeState | None:
        """Most-recent snapshot for class_name by registered_at, or None."""
        states = self._class_index.get(class_name)
        if not states:
            return None
        return max(states, key=lambda s: s.registered_at)

    def is_version_change(self, state: CodeState) -> bool:
        """True when snapshot is not the oldest registration for its class."""
        siblings = self._class_index.get(state.class_name, [])
        if len(siblings) <= 1:
            return False
        oldest = min(s.registered_at for s in siblings)
        return state.registered_at != oldest

    def get_mtime_from_hash(self, source_hash: str) -> str | None:
        """Return registered_at for source_hash, or None if not found."""
        s = self._hash_index.get(source_hash)
        return s.registered_at if s is not None else None

    def resolve_hash_prefix(self, prefix: str) -> str | None:
        """Return the full source hash matching prefix, or None if zero or multiple match."""
        return _resolve_prefix(self._hash_index, prefix)

    def get_class_name_from_hash(self, source_hash: str) -> str:
        """Return the class name for source_hash, or empty string if not found."""
        s = self._hash_index.get(source_hash)
        return s.class_name if s is not None else ''

    def get_previous_state(self, source_hash: str) -> CodeState | None:
        """Return the chronologically prior CodeState for source_hash, or None if oldest or not found."""
        state = self._hash_index.get(source_hash)
        if state is None:
            return None
        states = self._class_index.get(state.class_name, [])
        idx = next((i for i, s in enumerate(states) if s.source_hash == source_hash), None)
        return states[idx - 1] if idx is not None and idx > 0 else None


class TreeRegistry:
    """Index of dependency tree snapshots, scanned from .source/snapshots/*/tree.json.

    Snapshots are indexed by dependency_hash.

    ``registry_root`` overrides the config registry path.  When omitted,
    defaults to ``get_config().path_registry``.
    """

    def __init__(self, registry_root: Path | None = None) -> None:
        self._registry_root = registry_root
        self._class_index: dict[str, list[TreeSnapshot]] = {}
        self._hash_index: dict[str, TreeSnapshot] = {}

        for path in RegistryResolver(self._registry_root).glob_tree_paths():
            try:
                snapshot = TreeSnapshot.from_file(path)
            except (OSError, msgspec.DecodeError):
                continue
            self._hash_index[snapshot.dependency_tree_hash or path.parent.name] = snapshot
            if snapshot.root_class:
                self._class_index.setdefault(snapshot.root_class, []).append(snapshot)

    @property
    def class_names(self) -> list[str]:
        """All class names that own at least one snapshot."""
        return list(self._class_index.keys())

    @property
    def dependency_hashes(self) -> list[str]:
        """All known dep hashes."""
        return list(self._hash_index.keys())

    def get_snapshots(self, class_name: str) -> list[TreeSnapshot]:
        """Return all snapshots rooted at class_name, or empty list."""
        return self._class_index.get(class_name, [])

    def get_snapshot_for_source_hash(self, class_name: str, source_hash: str) -> TreeSnapshot | None:
        """Return the snapshot rooted at class_name whose root node matches source_hash, or None."""
        return next(
            (s for s in self._class_index.get(class_name, []) if s.get_source_hash(class_name) == source_hash),
            None,
        )

    def get_snapshot_from_hash(self, dependency_hash: str) -> TreeSnapshot | None:
        """Return the TreeSnapshot for dep_hash, or None if absent."""
        return self._hash_index.get(dependency_hash)

    def get_class_source_hash(self, dependency_hash: str, class_name: str) -> str | None:
        """Return the source_hash for class_name within the snapshot for dep_hash, or None."""
        snapshot = self._hash_index.get(dependency_hash)
        return snapshot.get_source_hash(class_name) if snapshot is not None else None

    def get_nodes(self, dependency_hash: str) -> dict[str, TreeNode] | None:
        """Return the nodes dict for dep_hash, or None if tree is absent."""
        tree = self._hash_index.get(dependency_hash)
        return tree.nodes if tree is not None else None

    def get_tree_path(self, dependency_hash: str) -> Path:
        """Return the path to tree.json for dep_hash (may not exist)."""
        return TreeRegistryPathConstructor.from_dep_tree_hash(dependency_hash, self._registry_root).tree_path

    def get_call_dependencies(self, dependency_hash: str) -> list[str]:
        """Sorted direct call-dependency names for dep_hash, or empty list."""
        tree = self._hash_index.get(dependency_hash)
        return tree.get_call_dependencies() if tree is not None else []

    def get_inheritance_dependencies(self, dependency_hash: str) -> list[str]:
        """Sorted direct inheritance-dependency names for dep_hash, or empty list."""
        tree = self._hash_index.get(dependency_hash)
        return tree.get_inheritance_dependencies() if tree is not None else []

    def resolve_hash_prefix(self, prefix: str) -> str | None:
        """Return the full dep hash matching prefix, or None if zero or multiple match."""
        return _resolve_prefix(self._hash_index, prefix)


class EntryRegistry:
    """Read-only index over the catalog of cache entries, keyed by state hash.

    Built from ``data`` when given, otherwise from :func:`~pygeodata.catalog.store.open_catalog`,
    which brings a writable catalog up to date with the entry dirs first (``read_only`` as there).
    Display assembly lives in :mod:`pygeodata.views`, staleness in :mod:`pygeodata.staleness`.
    """

    def __init__(self, data: CatalogData | None = None, read_only: bool | None = None) -> None:
        if data is None:
            data, self.status = store.open_catalog(read_only)
        else:
            self.status = None
        self._data = data
        self._class_index: dict[str, list[str]] = {}
        for state_hash, entry in data.entries.items():
            self._class_index.setdefault(entry.class_name, []).append(state_hash)

    @property
    def header(self) -> CatalogHeader:
        return self._data.header

    @property
    def entries(self) -> dict[str, CatalogEntry]:
        return self._data.entries

    @property
    def specs(self) -> dict[str, SpecRecord]:
        return self._data.specs

    @property
    def class_names(self) -> list[str]:
        return list(self._class_index)

    def get_entry(self, state_hash: str) -> CatalogEntry | None:
        return self._data.entries.get(state_hash)

    def get_spec(self, state_hash: str) -> SpecRecord | None:
        entry = self._data.entries.get(state_hash)
        return self._data.specs.get(entry.record.spec_hash) if entry is not None else None

    def entry_dir(self, state_hash: str) -> Path | None:
        entry = self._data.entries.get(state_hash)
        return store.entry_dir(entry) if entry is not None else None

    def get_state_hashes(self, class_name: str) -> list[str]:
        return self._class_index.get(class_name, [])

    def get_object_type(self, class_name: str) -> str | None:
        hashes = self._class_index.get(class_name)
        return self._data.entries[hashes[0]].object_type if hashes else None

    def match_prefix(self, prefix: str) -> list[str]:
        return [h for h in self._data.entries if h.startswith(prefix)]

    def resolve_hash_prefix(self, prefix: str) -> str | None:
        """Return the full state hash matching prefix, or None if zero or multiple match."""
        return _resolve_prefix(self._data.entries, prefix)

    def diagnostics(self) -> dict:
        diff = self.status.diff if self.status is not None else None
        return {
            'entries': len(self._data.entries),
            'catalog_id': self._data.header.catalog_id,
            'read_only': self.status.read_only if self.status is not None else None,
            'unindexed_dirs': len(diff.unindexed) if diff else 0,
            'unindexed_reasons': asdict(diff.unindexed_reasons) if diff and diff.unindexed_reasons else None,
            'missing_dirs': len(diff.missing) if diff else 0,
            'unavailable_roots': [str(r) for r in diff.unavailable_roots] if diff else [],
        }
