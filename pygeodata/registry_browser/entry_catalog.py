import contextlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config
from pygeodata.paths import CACHE_DIR_SUFFIXES, CACHE_META_SUFFIXES, CachePathResolver, classify_file
from pygeodata.registry import EntryRegistry
from pygeodata.registry_types import GroupRecord
from pygeodata.registry_browser.class_catalog import source_info_from_disk
from pygeodata.registry_browser.io_utils import existing_path_str, read_json_dict
from pygeodata.registry_browser.models import (
    EntryInfo,
    FileRef,
    LinkedEntry,
    ParamRow,
    SpecInfo,
)
from pygeodata.registry_browser.params_index import flatten_params
from pygeodata.spec import SpatialSpec, SpecKeys
from pygeodata.tracked_object import TrackedObject
from pygeodata.versioning import VersionRegistry

_log = logging.getLogger(__name__)


def _cache_file() -> Path:
    return get_config().path_registry / '.dashboard_cache.json'


def _cache_mtime_key(params_path: Path) -> float:
    """Combined mtime of params + spec files — any change invalidates the display cache."""
    resolver = CachePathResolver.from_path(params_path)
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
    name = path.name
    if name.startswith('.'):
        return False
    for suffix in CACHE_META_SUFFIXES:
        if name.endswith(suffix):
            return False
    if path.is_dir():
        return path.suffix.lower() in CACHE_DIR_SUFFIXES
    return path.is_file()


def _find_primary_file(resolver: CachePathResolver) -> FileRef | None:
    for path in resolver.directory.iterdir():
        if not _is_output_file(path):
            continue
        if path.name.split('.')[0] == resolver.stem:
            return FileRef(label=path.name, path=str(path.resolve()), kind=classify_file(path))
    return None


def _enrich_params_path(params_path: Path) -> EntryInfo:
    """Read the display-layer files for one entry. Never raises.

    Returns a partial EntryInfo with all display fields populated but
    assembly-stage fields (record_id, dep_hash_stale, co_outputs) left
    at their defaults — those are filled by discover_entries.
    """
    params_path_str = str(params_path.resolve())
    resolver = CachePathResolver.from_path(params_path)

    params = read_json_dict(params_path)
    spec = read_json_dict(resolver.spec_path)

    primary_file = _find_primary_file(resolver)

    linked_entries: list[LinkedEntry] = []
    rows = flatten_params(params, linked_entries=linked_entries)

    # Identity fields come from EntryRegistry; read the hash file only for
    # fields not already in EntryRecord (spec_path, state_hash_path, etc.)
    state = read_json_dict(resolver.state_hash_path)
    class_name = state.get(JSONKeys.CLASS_NAME) or resolver.stem
    state_hash = state.get(JSONKeys.STATE_HASH)
    instance_hash = state.get(JSONKeys.INSTANCE_HASH)
    stored_dep_hash = state.get(JSONKeys.DEPENDENCY_TREE_HASH)
    co_output_hashes = state.get(JSONKeys.CO_OUTPUTS, [])
    format_version = state.get(JSONKeys.FORMAT_VERSION, FORMAT_VERSION)

    object_type = None
    live_cls = TrackedObject.find_object_class(class_name)
    if live_cls is not None:
        ot = getattr(live_cls, 'object_type', None)
        if ot is not None:
            getter = getattr(ot, 'get_class_name', None)
            object_type = str(getter()) if callable(getter) else str(ot)
    else:
        object_type = source_info_from_disk(class_name).object_type

    warnings: list[str] = []
    if not state_hash:
        warnings.append('Missing state hash in hash.json')

    return EntryInfo(
        class_name=class_name,
        object_type=object_type,
        params_path=params_path_str,
        spec_path=existing_path_str(getattr(resolver, 'spec_path', None)),
        state_hash_path=existing_path_str(getattr(resolver, 'state_hash_path', None)),
        execution_graph_path=existing_path_str(getattr(resolver, 'execution_graph_path', None)),
        state_hash=state_hash,
        instance_hash=instance_hash,
        params=params,
        spec=SpecInfo.from_spec(SpatialSpec.from_dict(spec)) if spec else SpecInfo(),
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
            return EntryInfo.from_dict(dict(cached_results[key])), True
        except (KeyError, TypeError):
            pass
    entry = _enrich_params_path(params_path)
    return entry, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def discover_entries(
    progress: dict | None = None,
) -> tuple[dict[str, EntryInfo], dict[str, GroupRecord], dict]:
    """Enrich EntryRegistry records with display-layer data.

    Uses EntryRegistry as the single source of params paths (no second rglob).
    The display cache (.dashboard_cache.json) is keyed by params_path and
    covers only the browser-specific fields (spec, rows, linked_entries, etc.).
    """
    entry_registry = EntryRegistry.instance()
    # params_path → EntryRecord, for dep_hash_stale and co_output_hashes
    record_by_params = {
        str(rec.params_path.resolve()): rec
        for rec in entry_registry.records.values()
        if rec.params_path is not None
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
    diagnostics = dict(entry_registry.diagnostics)
    diagnostics['created_entries'] = 0

    version_registry = VersionRegistry.instance()

    for entry in partial_entries:
        if entry.error is not None:
            continue

        rec = record_by_params.get(str(Path(entry.params_path).resolve()) if entry.params_path else '')
        if rec is None:
            continue

        # record_id == state_hash (the key in EntryRegistry.records)
        record_id = rec.state_hash or entry.params_path
        entry.record_id = record_id
        entry.dep_hash_stale = bool(rec.dep_hash_stale)

        # EntryRegistry only marks staleness for classes loaded in this process;
        # for unloaded classes resolve it against the version registry on disk.
        if (
            entry.dep_hash
            and not entry.dep_hash_stale
            and TrackedObject.find_object_class(entry.class_name) is None
        ):
            entry.dep_hash_stale = version_registry.is_dep_hash_stale(entry.dep_hash)

        entries[record_id] = entry

    # Second pass — resolve co_output_hashes to EntryInfo references
    for entry in entries.values():
        entry.co_outputs = [entries[h] for h in entry.co_output_hashes if h in entries]

    diagnostics['created_entries'] = len(entries)
    return entries, entry_registry.groups, diagnostics
