from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pygeodata import Artifact
from pygeodata.config import JSONKeys
from pygeodata.paths import CachePathResolver
from pygeodata.tracked_object import TrackedObject

import logging
_log = logging.getLogger(__name__)

from .class_catalog import source_info_from_disk
from .io_utils import classify_file, existing_path_str, read_json_dict
from .models import EntryInfo, FileRef, GroupInfo, LinkedEntry, ParamRow, SpecInfo
from .params_index import flatten_params


# ---------------------------------------------------------------------------
# Disk cache for processed params — keyed by params file path, invalidated
# by mtime changes on params + hash + spec files.
# ---------------------------------------------------------------------------

def _cache_file() -> Path:
    from pygeodata.config import get_config
    return get_config().path_registry / '.dashboard_cache.json'


def _cache_mtime_key(params_path: Path) -> float:
    """Combined mtime of params + hash + spec files — any change invalidates."""
    resolver = CachePathResolver.from_path(params_path)
    total = 0.0
    for p in (params_path, resolver.state_hash_path, resolver.spec_path):
        try:
            total += p.stat().st_mtime
        except OSError:
            pass
    return total


def _load_disk_cache() -> tuple[dict[str, dict], dict[str, float]]:
    """Load cache from disk. Returns (results, mtimes) or empty dicts."""
    path = _cache_file()
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data.get('results', {}), data.get('mtimes', {})
    except Exception:
        return {}, {}


def _save_disk_cache(results: dict[str, dict], mtimes: dict[str, float]) -> None:
    try:
        path = _cache_file()
        path.write_text(
            json.dumps({'results': results, 'mtimes': mtimes}, separators=(',', ':')),
            encoding='utf-8',
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# ProcessResult — typed return value from _process_params_path
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProcessResult:
    class_name: str
    object_type: str
    params_path_str: str
    spec_path: str | None
    state_hash_path: str | None
    execution_graph_path: str | None
    state_hash: str | None
    instance_hash: str | None
    stored_dep_hash: str | None
    co_output_hashes: list[str]
    params: dict[str, Any]
    spec: SpecInfo
    rows: list[ParamRow]
    linked_entries: list[LinkedEntry]
    primary_file: FileRef | None
    warnings: list[str]
    derived: bool
    error: str | None = None   # set when resolver fails; other fields will be defaults


# ---------------------------------------------------------------------------
# Serialise/deserialise ProcessResult for the disk cache (plain JSON types)
# ---------------------------------------------------------------------------

def _serialise_result(result: ProcessResult) -> dict:
    d = dataclasses.asdict(result)
    # bounds_latlon is a tuple — asdict converts it to a list, which is fine for JSON.
    # No further conversion needed.
    return d


def _deserialise_result(data: dict) -> ProcessResult:
    spec_d = data.pop('spec')
    bl = spec_d.get('bounds_latlon')
    spec = SpecInfo(
        crs=spec_d.get('crs'),
        resolution=spec_d.get('resolution'),
        shape=spec_d.get('shape'),
        bounds=spec_d.get('bounds'),
        bounds_latlon=tuple(bl) if bl else None,
    )
    rows = [ParamRow(**r) for r in data.pop('rows')]
    linked_entries = [LinkedEntry(**le) for le in data.pop('linked_entries')]
    primary_file_d = data.pop('primary_file')
    primary_file = FileRef(**primary_file_d) if primary_file_d else None
    return ProcessResult(spec=spec, rows=rows, linked_entries=linked_entries,
                         primary_file=primary_file, **data)


# ---------------------------------------------------------------------------
# Per-entry processing
# ---------------------------------------------------------------------------

def _object_type_from_class_name(class_name: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    cls = TrackedObject.find_object_class(class_name)
    if cls is not None:
        object_type = getattr(cls, 'object_type', None)
        if object_type is None:
            warnings.append('Class has no object_type')
            return 'unknown', warnings
        getter = getattr(object_type, 'get_class_name', None)
        return (str(getter()).lower() if callable(getter) else str(object_type).lower()), warnings

    object_type, _, _, _, _ = source_info_from_disk(class_name)
    return object_type, warnings


_CACHE_META_SUFFIXES = frozenset({'.params.json', '.hash.json', '.spec.json', '.graph.pdf'})
_DIR_SUFFIXES = frozenset({'.zarr'})


def _is_output_file(path: Path) -> bool:
    name = path.name
    if name.startswith('.'):
        return False
    for suffix in _CACHE_META_SUFFIXES:
        if name.endswith(suffix):
            return False
    if path.is_dir():
        return path.suffix.lower() in _DIR_SUFFIXES
    return path.is_file()


def _find_primary_file(resolver: CachePathResolver) -> FileRef | None:
    for path in resolver.directory.iterdir():
        if not _is_output_file(path):
            continue
        if path.name.split('.')[0] == resolver.stem:
            return FileRef(label=path.name, path=str(path.resolve()), kind=classify_file(path))
    return None


def _unique_record_id(state_hash: str | None, params_path_str: str, taken: set[str]) -> tuple[str, bool]:
    if state_hash:
        if state_hash not in taken:
            return state_hash, False
        stem = Path(params_path_str).stem.lstrip('.')
        candidate = f'{state_hash}/{stem}'
        suffix = 0
        base = candidate
        while candidate in taken:
            suffix += 1
            candidate = f'{base}_{suffix}'
        return candidate, True
    return params_path_str, False


_EMPTY_PROCESS_RESULT = ProcessResult(
    class_name='', object_type='', params_path_str='', spec_path=None,
    state_hash_path=None, execution_graph_path=None, state_hash=None,
    instance_hash=None, stored_dep_hash=None, co_output_hashes=[],
    params={}, spec=SpecInfo(), rows=[], linked_entries=[],
    primary_file=None, warnings=[], derived=False,
)


def _process_params_path(params_path: Path) -> ProcessResult:
    """Process one params file into a ProcessResult. Never raises."""
    params_path_str = str(params_path.resolve())
    warnings: list[str] = []

    try:
        resolver = CachePathResolver.from_path(params_path)
    except Exception as exc:
        return dataclasses.replace(_EMPTY_PROCESS_RESULT,
                                   params_path_str=params_path_str,
                                   error=str(exc))

    params = read_json_dict(params_path)
    state  = read_json_dict(resolver.state_hash_path)
    spec   = read_json_dict(resolver.spec_path)

    class_name = state.get(JSONKeys.CLASS_NAME)
    derived = False
    if not class_name:
        class_name = resolver.stem
        warnings.append('Class name derived from params path (hash.json had no class_name)')
        derived = True

    state_hash = state.get(JSONKeys.STATE_HASH)
    if not state_hash:
        warnings.append('Missing state hash in hash.json')

    instance_hash = state.get(JSONKeys.INSTANCE_HASH)
    stored_dep_hash = state.get(JSONKeys.DEPENDENCY_TREE_HASH)
    co_output_hashes = state.get(JSONKeys.CO_OUTPUTS, [])

    object_type, object_warnings = _object_type_from_class_name(class_name)
    warnings.extend(object_warnings)

    primary_file = _find_primary_file(resolver)

    linked_entries: list[LinkedEntry] = []
    rows = flatten_params(params, linked_entries=linked_entries)

    return ProcessResult(
        class_name=class_name,
        object_type=object_type,
        params_path_str=params_path_str,
        spec_path=existing_path_str(getattr(resolver, 'spec_path', None)),
        state_hash_path=existing_path_str(getattr(resolver, 'state_hash_path', None)),
        execution_graph_path=existing_path_str(getattr(resolver, 'execution_graph_path', None)),
        state_hash=state_hash,
        instance_hash=instance_hash,
        stored_dep_hash=stored_dep_hash,
        co_output_hashes=co_output_hashes,
        params=params,
        spec=SpecInfo.from_spec_json(spec),
        rows=rows,
        linked_entries=linked_entries,
        primary_file=primary_file,
        warnings=warnings,
        derived=derived,
    )


def _process_with_cache(
    params_path: Path,
    cached_results: dict[str, dict],
    cached_mtimes: dict[str, float],
) -> tuple[ProcessResult, bool]:
    """Return (result, from_cache). Uses disk cache if mtimes match."""
    key = str(params_path.resolve())
    current_mtime = _cache_mtime_key(params_path)
    if key in cached_results and cached_mtimes.get(key) == current_mtime:
        return _deserialise_result(dict(cached_results[key])), True
    result = _process_params_path(params_path)
    return result, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover_entries(
    progress: dict | None = None,
) -> tuple[dict[str, EntryInfo], dict[str, GroupInfo], dict]:
    cache_roots = [family.get_cache_root() for family in Artifact.__subclasses__()]
    params_paths = sorted(
        {path for root in cache_roots if root.exists() for path in root.rglob('*.params.json')},
    )

    diagnostics: dict = {
        'resolver_failures': [],
        'missing_state_hash': [],
        'derived_class_name': [],
        'hash_collisions': [],
        'scanned_params_paths': len(params_paths),
        'created_entries': 0,
    }

    if progress is not None:
        progress['done'] = 0
        progress['total'] = len(params_paths)

    cached_results, cached_mtimes = _load_disk_cache()

    results: list[ProcessResult] = [None] * len(params_paths)  # type: ignore[list-item]
    new_results: dict[str, dict] = {}   # path → serialised result, for cache update
    new_mtimes: dict[str, float] = {}

    def _process(i: int, p: Path) -> tuple[int, ProcessResult]:
        result, from_cache = _process_with_cache(p, cached_results, cached_mtimes)
        key = str(p.resolve())
        if not from_cache and result.error is None:
            new_results[key] = _serialise_result(result)
            new_mtimes[key] = _cache_mtime_key(p)
        elif from_cache:
            new_results[key] = cached_results[key]
            new_mtimes[key] = cached_mtimes[key]
        return i, result

    with ThreadPoolExecutor() as pool:
        future_to_idx = {pool.submit(_process, i, p): i for i, p in enumerate(params_paths)}
        for future in as_completed(future_to_idx):
            idx, result = future.result()
            results[idx] = result
            if progress is not None:
                progress['done'] += 1

    _save_disk_cache(new_results, new_mtimes)

    entries: dict[str, EntryInfo] = {}
    groups: dict[str, GroupInfo] = {}

    for result in results:
        if result.error is not None:
            diagnostics['resolver_failures'].append({'path': result.params_path_str, 'error': result.error})
            continue

        if not result.state_hash:
            diagnostics['missing_state_hash'].append(result.params_path_str)
        if result.derived:
            diagnostics['derived_class_name'].append(result.params_path_str)

        record_id, collision = _unique_record_id(result.state_hash, result.params_path_str, set(entries))
        warnings = result.warnings
        if collision:
            warnings.append(f'State hash shared with another entry; record_id disambiguated to "{record_id}"')
            diagnostics['hash_collisions'].append({
                'path': result.params_path_str,
                'state_hash': result.state_hash,
                'record_id': record_id,
            })

        class_name = result.class_name
        dep_hash_stale = False
        if result.stored_dep_hash:
            cls = TrackedObject.find_object_class(class_name)
            if cls is not None:
                try:
                    dep_hash_stale = cls.get_dependency_tree_hash() != result.stored_dep_hash
                except Exception:
                    pass

        entries[record_id] = EntryInfo(
            record_id=record_id,
            class_name=class_name,
            object_type=result.object_type,
            params_path=result.params_path_str,
            spec_path=result.spec_path,
            state_hash_path=result.state_hash_path,
            execution_graph_path=result.execution_graph_path,
            state_hash=result.state_hash,
            instance_hash=result.instance_hash,
            params=result.params,
            spec=result.spec,
            rows=result.rows,
            linked_entries=result.linked_entries,
            co_output_hashes=result.co_output_hashes,
            primary_file=result.primary_file,
            warnings=warnings,
            dep_hash_stale=dep_hash_stale,
        )
        groups.setdefault(
            class_name,
            GroupInfo(class_name=class_name, object_type=result.object_type),
        ).record_ids.append(record_id)

    # Second pass — resolve co_output_hashes to EntryInfo references
    for entry in entries.values():
        entry.co_outputs = [
            entries[h] for h in entry.co_output_hashes if h in entries
        ]

    diagnostics['created_entries'] = len(entries)
    return entries, groups, diagnostics
