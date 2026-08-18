import contextlib
import json
from pathlib import Path

from pygeodata.catalog.params_index import flatten_params
from pygeodata.catalog.types import (
    EntryInfo,
    FileRef,
    LinkedEntry,
    SpecInfo,
)
from pygeodata.config import FORMAT_VERSION, dashboard_cache_path
from pygeodata.figure import Figure
from pygeodata.paths import (
    CACHE_DIR_SUFFIXES,
    CACHE_META_FILES,
    CachePathConstructor,
    classify_file,
    write_json_atomic,
)
from pygeodata.registries.registry import EntryRegistry
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.spec import SpatialSpec
from pygeodata.table import Table
from pygeodata.tracked_object import TrackedObject


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
    path = dashboard_cache_path()
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data.get('results', {}), data.get('mtimes', {})
    except (OSError, json.JSONDecodeError):
        return {}, {}


def _save_disk_cache(results: dict[str, dict], mtimes: dict[str, float], dirty: bool = True) -> None:
    if not dirty:
        return
    write_json_atomic(dashboard_cache_path(), {'results': results, 'mtimes': mtimes})


# ---------------------------------------------------------------------------
# Per-entry display enrichment
# ---------------------------------------------------------------------------


#: Fixed extension per artifact family, for entries whose own class is not imported. `Data` is
#: absent deliberately: it resolves its extension from a driver, so there is no default.
_FAMILY_EXT: dict[str, str] = {'Figure': Figure.ext, 'Table': Table.ext}


def _is_output_file(path: Path) -> bool:
    if path.name in CACHE_META_FILES:
        return False
    if path.is_dir():
        return path.suffix.lower() in CACHE_DIR_SUFFIXES
    return path.is_file()


def _preferred_ext(class_name: str | None, object_type: str | None) -> str | None:
    """The extension this entry's output is expected to carry, or None if unknown.

    The class's own `ext` is the authority, but it is reachable only once the project's classes
    are imported, and `Data` leaves it None to be resolved from a driver. The `object_type` in
    meta.json is always recorded, so the family default stands in — only the concrete subclasses
    register themselves, so the families are named here rather than looked up.
    """
    obj_cls = TrackedObject.find_object_class(class_name) if class_name else None
    ext = getattr(obj_cls, 'ext', None)

    if ext is None:
        ext = _FAMILY_EXT.get(object_type)

    return f'.{ext}'.lower() if isinstance(ext, str) else None


def _find_primary_and_sidecar_files(
    resolver: CachePathConstructor,
    class_name: str | None = None,
    object_type: str | None = None,
) -> tuple[FileRef | None, list[FileRef]]:
    """The entry's main output, preferring the file whose suffix the class declares, plus
    every other output file in the directory as sidecars.

    A `_process` may write sidecars beside its declared output — a CSV next to a rendered HTML
    table, say. Taking whatever `iterdir` yields first picks between them by filesystem order,
    so one entry is classified 'table' and the next 'file' for the same class. Sorting breaks
    the tie deterministically; the declared extension breaks it correctly. The remaining
    candidates are not primary-file material but are still real outputs, so they are returned
    as sidecars rather than silently dropped.
    """
    candidates = sorted(
        (p for p in resolver.directory.iterdir() if _is_output_file(p)),
        key=lambda p: p.name,
    )
    if not candidates:
        return None, []

    ext = _preferred_ext(class_name, object_type)
    path = next((p for p in candidates if p.suffix.lower() == ext), candidates[0])
    primary = FileRef(label=path.name, path=str(path.resolve()), kind=classify_file(path))
    sidecars = [
        FileRef(label=p.name, path=str(p.resolve()), kind=classify_file(p))
        for p in candidates
        if p != path
    ]
    return primary, sidecars


def _enrich_params_path(params_path: Path, record: EntryRecord | None = None) -> EntryInfo:
    """Read the display-layer files for one entry. Never raises.

    Returns a partial EntryInfo with all display fields populated but
    assembly-stage fields (record_id, dep_hash_stale, co_outputs) left
    at their defaults — those are filled by discover_entries.

    ``record`` is the EntryRegistry record for this entry, which discover_entries
    already holds in memory; passing it avoids re-reading meta.json per entry.
    When omitted the record is read from disk, so standalone callers still work.
    """
    params_path_str = str(params_path.resolve())
    resolver = CachePathConstructor.from_path(params_path)

    params = json.loads(params_path.read_text(encoding='utf-8')) if params_path.exists() else {}
    spec = json.loads(resolver.spec_path.read_text(encoding='utf-8')) if resolver.spec_path.exists() else {}

    linked_entries: list[LinkedEntry] = []
    rows = flatten_params(params, linked_entries=linked_entries)

    if record is None and resolver.state_hash_path.exists():
        record = EntryRecord.from_file(resolver.state_hash_path)
    class_name = (record.class_name if record else None) or resolver.directory.name
    state_hash = record.state_hash if record else None
    instance_hash = record.instance_hash if record else None
    params_hash = record.params_hash if record else None
    spec_hash = record.spec_hash if record else None
    stored_dep_hash = record.dependency_tree_hash if record else None

    co_output_hashes = record.co_output_hashes if record else []
    format_version = record.format_version if record else FORMAT_VERSION

    object_type = record.object_type if record else None

    primary_file, sidecar_files = _find_primary_and_sidecar_files(resolver, class_name, object_type)

    warnings: list[str] = []
    if not state_hash:
        warnings.append('Missing state hash in meta.json')

    spatial_spec = SpatialSpec.from_dict(spec) if spec else None

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
        sidecar_files=sidecar_files,
        warnings=warnings,
        format_version=format_version,
        dep_hash=stored_dep_hash,
    )


def _enrich_with_cache(
    params_path: Path,
    cached_results: dict[str, dict],
    cached_mtimes: dict[str, float],
    record: EntryRecord | None = None,
) -> tuple[EntryInfo, bool]:
    """Return (entry, from_cache). Uses display cache if mtimes match."""
    key = str(params_path.resolve())
    current_mtime = _cache_mtime_key(params_path)
    if key in cached_results and cached_mtimes.get(key) == current_mtime:
        try:
            return EntryInfo.from_dict(dict(cached_results[key])), True
        except (KeyError, TypeError):
            pass
    entry = _enrich_params_path(params_path, record)
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

    partial_entries: list[EntryInfo] = []
    new_results: dict[str, dict] = {}
    new_mtimes: dict[str, float] = {}

    any_miss = False
    for p in params_paths:
        key = str(p.resolve())
        entry, from_cache = _enrich_with_cache(p, cached_results, cached_mtimes, record_by_params.get(key))
        if from_cache:
            new_results[key] = cached_results[key]
            new_mtimes[key] = cached_mtimes[key]
        else:
            any_miss = True
            if entry.error is None:
                new_results[key] = entry.to_dict()
                new_mtimes[key] = _cache_mtime_key(p)
        partial_entries.append(entry)
        if progress is not None:
            progress['done'] += 1

    dirty = any_miss or new_results.keys() != cached_results.keys() or not dashboard_cache_path().exists()
    _save_disk_cache(new_results, new_mtimes, dirty=dirty)

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
