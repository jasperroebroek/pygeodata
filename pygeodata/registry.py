"""Read-side of the registry tree and the entry cache.

Three registries, all keyed by resolved root path:

    SourceRegistry  — scans .source/code/*/source.json, indexes by class name
    TreeRegistry    — scans .source/snapshots/*/tree.json, indexes directory names, reads tree.json lazily
    EntryRegistry   — scans cache roots for meta.json, owns EntryInfo/GroupInfo
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pygeodata.config import get_config
from pygeodata.paths import (
    CachePathResolver,
    CodeRegistryPathConstructor,
    RegistryResolver,
    TreeRegistryPathConstructor,
)
from pygeodata.registry_types import CodeState, EntryRecord, TreeSnapshot


class SourceRegistry:
    """Index of code states, scanned from .source/code/*/source.json.

    States are indexed by class name and kept sorted oldest-first by
    registered_at.  Call :meth:`reload` to re-scan in place; held references
    remain valid.

    ``registry_root`` overrides the config registry path.  When omitted,
    defaults to ``get_config().path_registry``.
    """

    def __init__(self, registry_root: Path | None = None) -> None:
        self._registry_root = registry_root
        self._class_index: dict[str, list[CodeState]]
        self._hash_index: dict[str, CodeState]
        self.reload()

    def reload(self) -> None:
        self._class_index: dict[str, list[CodeState]] = {}
        for meta_path in RegistryResolver(self._registry_root).glob_source_paths():
            try:
                state = CodeState.from_file(meta_path)
            except (KeyError, ValueError):
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
        matches = [h for h in self._hash_index if h.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

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

    Snapshots are indexed by dependency_hash. Call :meth:`reload` to
    re-scan in place; held references remain valid.

    ``registry_root`` overrides the config registry path.  When omitted,
    defaults to ``get_config().path_registry``.
    """

    def __init__(self, registry_root: Path | None = None) -> None:
        self._class_index: dict[str, list[TreeSnapshot]]
        self._hash_index: dict[str, TreeSnapshot]
        self._registry_root = registry_root
        self.reload()

    def reload(self) -> None:
        """Rescan snapshots/ and eagerly load all tree.json files."""
        self._class_index = {}
        self._hash_index = {}

        for path in RegistryResolver(self._registry_root).glob_tree_paths():
            try:
                snapshot = TreeSnapshot.from_file(path)
            except OSError:
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

    def get_nodes(self, dependency_hash: str) -> dict[str, dict] | None:
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
        matches = [h for h in self._hash_index if h.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None


class EntryRegistry:
    """Per-root registry of cache entries, keyed by state_hash.

    The single place that scans for meta.json files.  Reads each via
    :meth:`EntryRecord.from_file`, builds a hash index and class index,
    and owns the lightweight disk cache ('.entry_registry_cache.json').

    state_hash is the unique key.  On collision:
    - If all identity fields match → silently deduplicate (same entry in two locations).
    - If any field differs → raise ValueError (hash collision with divergent data is a
      serious integrity problem that must surface immediately).

    Browser display fields (rows, spec strings, linked entries, primary file) are
    NOT populated here — that enrichment is done by
    :func:`pygeodata.registry_browser.entry_catalog.discover_entries`.

    ``paths`` sets the cache roots to scan.  When omitted, defaults to
    ``[config.path_cache, config.path_figures]``.
    """

    def __init__(self, paths: list[Path] | None = None) -> None:
        self._hash_index: dict[str, EntryRecord]
        self._class_index: dict[str, list[str]]
        self._scanned: int
        self._missing: int
        self._paths = paths
        self.reload()

    def _cache_path(self) -> Path:
        return get_config().path_registry / '.entry_registry_cache.json'

    def _cache_resolver(self) -> CachePathResolver:
        if self._paths is not None:
            return CachePathResolver(tuple(self._paths))
        cfg = get_config()
        return CachePathResolver((cfg.path_cache, cfg.path_figures))

    def diagnostics(self) -> dict:
        return {
            'scanned_hash_paths': self._scanned,
            'missing_state_hash': self._missing,
            'created_records': len(self._hash_index),
        }

    @property
    def records(self) -> dict[str, EntryRecord]:
        return self._hash_index

    @property
    def class_names(self) -> list[str]:
        return list(self._class_index.keys())

    def get_record(self, state_hash: str) -> EntryRecord | None:
        return self._hash_index.get(state_hash)

    def get_state_hashes(self, class_name: str) -> list[str]:
        return self._class_index.get(class_name, [])

    def get_object_type(self, class_name: str) -> str | None:
        hashes = self._class_index.get(class_name)
        if not hashes:
            return None
        return self._hash_index[hashes[0]].object_type

    def resolve_hash_prefix(self, prefix: str) -> str | None:
        """Return the full state hash matching prefix, or None if zero or multiple match."""
        matches = [h for h in self._hash_index if h.startswith(prefix)]
        return matches[0] if len(matches) == 1 else None

    def reload(self) -> None:
        cache_path = self._cache_path()
        try:
            cached = json.loads(cache_path.read_text(encoding='utf-8'))
            cached_records, cached_mtimes = cached.get('records', {}), cached.get('mtimes', {})
        except (OSError, json.JSONDecodeError):
            cached_records, cached_mtimes = {}, {}

        new_cache_records: dict[str, dict] = {}
        new_cache_mtimes: dict[str, float] = {}

        def _process(p: Path) -> EntryRecord | None:
            key = str(p.resolve())
            mtime = p.stat().st_mtime
            if key in cached_records and cached_mtimes.get(key) == mtime:
                try:
                    return EntryRecord(**cached_records[key])
                except (KeyError, TypeError):
                    pass
            try:
                record = EntryRecord.from_file(p)
            except (OSError, json.JSONDecodeError, KeyError):
                return None
            new_cache_records[key] = dataclasses.asdict(record)
            new_cache_mtimes[key] = mtime
            return record

        scanned = 0
        records: list[EntryRecord | None] = []
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(_process, p): p for p in self._cache_resolver().glob_meta_paths()}
            for future in as_completed(futures):
                scanned += 1
                rec = future.result()
                records.append(rec)
                if rec is not None:
                    key = str(futures[future].resolve())
                    if key in cached_records and key not in new_cache_records:
                        new_cache_records[key] = cached_records[key]
                        new_cache_mtimes[key] = cached_mtimes[key]

        with contextlib.suppress(OSError):
            cache_path.write_text(
                json.dumps({'records': new_cache_records, 'mtimes': new_cache_mtimes}, separators=(',', ':')),
                encoding='utf-8',
            )

        self._hash_index: dict[str, EntryRecord] = {}
        self._class_index: dict[str, list[str]] = {}
        self._scanned = scanned
        self._missing = sum(1 for r in records if r is None or r.state_hash is None)
        for rec in records:
            if rec is None or rec.state_hash is None:
                continue
            if rec.state_hash in self._hash_index:
                if rec != self._hash_index[rec.state_hash]:
                    raise ValueError(f'state_hash collision with divergent data for hash {rec.state_hash!r}')
                continue
            self._hash_index[rec.state_hash] = rec
            self._class_index.setdefault(rec.class_name, []).append(rec.state_hash)
