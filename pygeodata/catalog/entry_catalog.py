import contextlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pygeodata.catalog.types import (
    EntryInfo,
    FileRef,
    LinkedEntry,
    SpecInfo,
)
from pygeodata.config import FORMAT_VERSION, get_config
from pygeodata.hash import calculate_dict_hash
from pygeodata.paths import CACHE_DIR_SUFFIXES, CACHE_META_FILES, CachePathConstructor, classify_file
from pygeodata.registries.registry import EntryRegistry
from pygeodata.catalog.params_index import flatten_params
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.spec import SpatialSpec
from pygeodata.tracked_object import TrackedObject
from pygeodata.registries.versioning import VersionRegistry


def _cache_file() -> Path:
    return get_config().path_registry / '.dashboard_cache.json'


def _cache_mtime_key(params_path: Path) -> float:
    """Combined mtime of params + spec files — any change invalidates the display cache."""
    resolver = CachePathConstructor.from_path(params_path)
    total = 0.0
    for p in (params_path, resolver.spec_path):
        with contextlib.suppress(OSError):
            total += p.stat().st_mtime
    return total


def _load_disk_cache() -> tuple[dict[str, dict], dict[str, float]]:
    """Load display cache from disk. Returns (results, mtimes) or empty dicts."""
    path = _cache_file()
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data.get('results', {}), data.get('mtimes', {})
    except (OSError, json.JSONDecodeError):
        return {}, {}


def _save_disk_cache(results: dict[str, dict], mtimes: dict[str, float]) -> None:
    try:
        path = _cache_file()
        path.write_text(
            json.dumps({'results': results, 'mtimes': mtimes}, separators=(',', ':')),
            encoding='utf-8',
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Per-entry display enrichment
# ---------------------------------------------------------------------------


def _is_output_file(path: Path) -> bool:
    if path.name in CACHE_META_FILES:
        return False
    if path.is_dir():
        return path.suffix.lower() in CACHE_DIR_SUFFIXES
    return path.is_file()


def _find_primary_file(resolver: CachePathConstructor) -> FileRef | None:
    for path in resolver.directory.iterdir():
        if not _is_output_file(path):
            continue
        return FileRef(label=path.name, path=str(path.resolve()), kind=classify_file(path))
    return None


def _enrich_params_path(params_path: Path) -> EntryInfo:
    """Read the display-layer files for one entry. Never raises.

    Returns a partial EntryInfo with all display fields populated but
    assembly-stage fields (record_id, dep_hash_stale, co_outputs) left
    at their defaults — those are filled by discover_entries.
    """
    params_path_str = str(params_path.resolve())
    resolver = CachePathConstructor.from_path(params_path)

    params = json.loads(params_path.read_text(encoding='utf-8')) if params_path.exists() else {}
    spec = json.loads(resolver.spec_path.read_text(encoding='utf-8')) if resolver.spec_path.exists() else {}

    primary_file = _find_primary_file(resolver)

    linked_entries: list[LinkedEntry] = []
    rows = flatten_params(params, linked_entries=linked_entries)

    record = EntryRecord.from_file(resolver.state_hash_path) if resolver.state_hash_path.exists() else None
    class_name = (record.class_name if record else None) or resolver.directory.name
    state_hash = record.state_hash if record else None
    instance_hash = record.instance_hash if record else None
    params_hash = record.params_hash if record else None
    spec_hash = record.spec_hash if record else None
    stored_dep_hash = record.dependency_tree_hash if record else None

    # Backfill for meta.json written before params_hash was persisted. Delete once no
    # cache entries predate that change.
    if params_hash is None:
        params_hash = calculate_dict_hash(params)
    co_output_hashes = record.co_output_hashes if record else []
    format_version = record.format_version if record else FORMAT_VERSION

    object_type = record.object_type if record else None

    warnings: list[str] = []
    if not state_hash:
        warnings.append('Missing state hash in meta.json')

    spatial_spec = SpatialSpec.from_dict(spec) if spec else None

    # Backfill for meta.json written before spec_hash was persisted. Delete once no
    # cache entries predate that change.
    if spec_hash is None and spatial_spec is not None:
        spec_hash = spatial_spec.get_hash()

    return EntryInfo(
        class_name=class_name,
        object_type=object_type,
        params_path=params_path_str,
        spec_path=str(resolver.spec_path) if resolver.spec_path.exists() else None,
        state_hash_path=str(resolver.state_hash_path) if resolver.state_hash_path.exists() else None,
        execution_graph_path=str(resolver.execution_graph_path) if resolver.execution_graph_path.exists() else None,
        state_hash=state_hash,
        instance_hash=instance_hash,
        params_hash=params_hash,
        spec_hash=spec_hash,
        params=params,
        spec=SpecInfo.from_spec(spatial_spec) if spatial_spec else SpecInfo(),
        rows=rows,
        linked_entries=linked_entries,
        co_output_hashes=co_output_hashes,
        primary_file=primary_file,
        warnings=warnings,
        format_version=format_version,
        dep_hash=stored_dep_hash,
    )


def _enrich_with_cache(
    params_path: Path,
    cached_results: dict[str, dict],
    cached_mtimes: dict[str, float],
) -> tuple[EntryInfo, bool]:
    """Return (entry, from_cache). Uses display cache if mtimes match."""
    key = str(params_path.resolve())
    current_mtime = _cache_mtime_key(params_path)
    if key in cached_results and cached_mtimes.get(key) == current_mtime:
        try:
            entry = EntryInfo.from_dict(dict(cached_results[key]))
            if entry.object_type is not None:
                return entry, True
            # object_type was missing when this entry was cached (e.g. meta.json
            # predates that field). Re-read it from disk rather than serving a
            # stale None forever — it's not part of the cache key.
        except (KeyError, TypeError):
            pass
    entry = _enrich_params_path(params_path)
    return entry, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def discover_entries(
    progress: dict | None = None,
    version_registry: VersionRegistry | None = None,
) -> tuple[dict[str, EntryInfo], EntryRegistry, dict]:
    """Enrich EntryRegistry records with display-layer data.

    Uses EntryRegistry as the single source of params paths (no second rglob).
    The display cache (.dashboard_cache.json) is keyed by params_path and
    covers only the browser-specific fields (spec, rows, linked_entries, etc.).
    """
    if version_registry is None:
        version_registry = VersionRegistry()

    entry_registry = EntryRegistry()
    # params_path → EntryRecord, for co_output_hashes
    record_by_params = {
        str(rec.params_path.resolve()): rec for rec in entry_registry.records.values() if rec.params_path is not None
    }
    params_paths = sorted(Path(p) for p in record_by_params)

    if progress is not None:
        progress['done'] = 0
        progress['total'] = len(params_paths)

    cached_results, cached_mtimes = _load_disk_cache()

    partial_entries: list[EntryInfo] = [None] * len(params_paths)  # type: ignore[list-item]
    new_results: dict[str, dict] = {}
    new_mtimes: dict[str, float] = {}

    def _process(i: int, p: Path) -> tuple[int, EntryInfo]:
        entry, from_cache = _enrich_with_cache(p, cached_results, cached_mtimes)
        key = str(p.resolve())
        if not from_cache and entry.error is None:
            new_results[key] = entry.to_dict()
            new_mtimes[key] = _cache_mtime_key(p)
        elif from_cache:
            new_results[key] = cached_results[key]
            new_mtimes[key] = cached_mtimes[key]
        return i, entry

    with ThreadPoolExecutor() as pool:
        future_to_idx = {pool.submit(_process, i, p): i for i, p in enumerate(params_paths)}
        for future in as_completed(future_to_idx):
            idx, entry = future.result()
            partial_entries[idx] = entry
            if progress is not None:
                progress['done'] += 1

    _save_disk_cache(new_results, new_mtimes)

    entries: dict[str, EntryInfo] = {}
    diagnostics = dict(entry_registry.diagnostics())
    diagnostics['created_entries'] = 0

    for entry in partial_entries:
        if entry.error is not None:
            continue

        rec = record_by_params.get(str(Path(entry.params_path).resolve()) if entry.params_path else '')
        if rec is None:
            continue

        # record_id == state_hash (the key in EntryRegistry.records)
        record_id = rec.state_hash or entry.params_path
        entry.record_id = record_id
        if entry.dep_hash:
            obj_cls = TrackedObject.find_object_class(entry.class_name)
            if obj_cls is not None:
                entry.dep_hash_stale = obj_cls.get_dependency_tree_hash() != entry.dep_hash
            elif version_registry is not None:
                entry.dep_hash_stale = version_registry.is_dependency_hash_stale(entry.dep_hash)

        entries[record_id] = entry

    # Second pass — resolve co_output_hashes to EntryInfo references
    for entry in entries.values():
        entry.co_outputs = [entries[h] for h in entry.co_output_hashes if h in entries]

    diagnostics['created_entries'] = len(entries)
    return entries, entry_registry, diagnostics
