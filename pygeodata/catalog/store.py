"""JSON file access layer for the catalog (:attr:`~pygeodata.config.Config.path_catalog`).

The catalog is an index of cache entries, never a source of truth: an entry only
exists because its ``meta.json`` was written successfully, and entry validity stays
defined by the JSON files on disk (see :mod:`pygeodata.cache`). The active
``FORMAT_VERSION`` and a schema fingerprint are recorded in the file — a catalog
written by a different version is treated as empty and rebuilt, never migrated.
Entries are keyed by their content-addressed ``state_hash``, so concurrent writers
producing the same hash converge on identical rows.

All write paths are no-ops while ``catalog_enabled`` is False on the config —
e.g. on shared filesystems. Incremental writes take a ``filelock`` and rewrite the
file atomically (temp file + rename), which is safe on any local filesystem and
needs no database locking protocol.

Paths are stored relative to the cache root that contains the entry, so a catalog
built on one machine resolves correctly on another that mounts the same cache at
a different absolute location; entries outside every configured cache root fall
back to absolute paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

from filelock import FileLock

from pygeodata.catalog.dimensions import dimension_rows
from pygeodata.catalog.types import FileRole
from pygeodata.config import FORMAT_VERSION, get_config
from pygeodata.fileio import write_json_atomic
from pygeodata.paths import CachePathConstructor, CachePathResolver
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.spec import SpatialSpec

if TYPE_CHECKING:
    from pygeodata.catalog.display import EntryDisplay

logger = logging.getLogger(__name__)

BACKFILL_LOG_INTERVAL = 2000

_META_FORMAT_VERSION = 'format_version'
_META_SCHEMA_VERSION = 'schema_fingerprint'
_ENTRY_KEYS = (
    'class_name',
    'object_type',
    'source_hash',
    'dependency_tree_hash',
    'instance_hash',
    'params_hash',
    'spec_hash',
    'crs_hash',
    'transform_hash',
    'bounds_hash',
    'shape_hash',
    'co_output_hashes',
    'rows',
    'linked_entries',
    'dir_relpath',
    'created_at',
    'updated_at',
)


def schema_fingerprint() -> str:
    """Fingerprint of the catalog's JSON structure — any layout change changes it.

    No schema version numbers: the catalog is disposable, so a fingerprint mismatch
    makes every load treat it as empty and the next backfill rebuilds it.
    """
    template = {
        _META_FORMAT_VERSION: 0,
        _META_SCHEMA_VERSION: '',
        'entries': dict.fromkeys(_ENTRY_KEYS),
        'files': [{'relpath': '', 'label': '', 'kind': '', 'role': ''}],
        'dimensions': {'crs': {}, 'transforms': {}, 'bounds': {}, 'shapes': {}},
    }
    return hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest()


def catalog_path() -> Path:
    return get_config().path_catalog


def _empty_catalog() -> dict[str, Any]:
    return {
        _META_FORMAT_VERSION: None,
        _META_SCHEMA_VERSION: schema_fingerprint(),
        'entries': {},
        'files': {},
        'dimensions': {'crs': {}, 'transforms': {}, 'bounds': {}, 'shapes': {}},
    }


def load_catalog() -> dict[str, Any]:
    """The catalog as a plain dict; a missing or outdated file yields an empty one."""
    path = catalog_path()
    if not path.is_file():
        return _empty_catalog()
    try:
        catalog = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f'Catalog unreadable, starting empty: {exc}', stacklevel=2)
        return _empty_catalog()
    if catalog.get(_META_FORMAT_VERSION) != FORMAT_VERSION or catalog.get(_META_SCHEMA_VERSION) != schema_fingerprint():
        return _empty_catalog()
    return catalog


def save_catalog(catalog: dict[str, Any]) -> None:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog[_META_FORMAT_VERSION] = FORMAT_VERSION
    catalog[_META_SCHEMA_VERSION] = schema_fingerprint()
    write_json_atomic(path, catalog)


@contextmanager
def _catalog_lock() -> Iterator[None]:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_name(f'{path.name}.lock')):
        yield


# ---------------------------------------------------------------------------
# Paths — cache-root relative, resolved against the local config
# ---------------------------------------------------------------------------


def _relpath(entry_dir: Path) -> str:
    """Relative to the cache root containing the entry; absolute when outside all roots."""
    for root in get_config().cache_roots:
        try:
            return str(Path(entry_dir).resolve().relative_to(Path(root).resolve()))
        except ValueError:  # noqa: PERF203 — a handful of roots; exception cost is irrelevant
            continue
    return str(Path(entry_dir).resolve())


def entry_abs_path(dir_relpath: str, cache_roots: Iterable[Path] | None = None) -> Path:
    """Resolve a stored relpath against the configured cache roots of this machine."""
    path = Path(dir_relpath)
    if path.is_absolute():
        return path.resolve()
    roots = tuple(get_config().cache_roots if cache_roots is None else cache_roots)
    for root in roots:
        if (root / path).exists():
            return (root / path).resolve()
    return (roots[0] / path).resolve() if roots else path


# ---------------------------------------------------------------------------
# Entry rows
# ---------------------------------------------------------------------------


def _entry_row(
    entry_dir: Path,
    record: EntryRecord,
    dims: dict[str, Any],
    display: EntryDisplay | None,
) -> dict[str, Any]:
    try:
        created_at = CachePathConstructor(Path(entry_dir)).state_hash_path.stat().st_mtime
    except OSError:
        created_at = None
    return {
        'state_hash': record.state_hash,
        'class_name': record.class_name,
        'object_type': record.object_type,
        'source_hash': record.source_hash,
        'dependency_tree_hash': record.dependency_tree_hash,
        'instance_hash': record.instance_hash,
        'params_hash': record.params_hash,
        'spec_hash': record.spec_hash,
        'crs_hash': dims['crs_hash'],
        'transform_hash': dims['transform_hash'],
        'bounds_hash': dims['bounds_hash'],
        'shape_hash': dims['shape_hash'],
        'co_output_hashes': list(record.co_output_hashes or []),
        'rows': display.rows if display is not None else None,
        'linked_entries': display.linked_entries if display is not None else None,
        'dir_relpath': _relpath(entry_dir),
        'created_at': created_at,
        'updated_at': created_at,
    }


def _upsert_dims(catalog: dict[str, Any], dims: dict[str, Any]) -> None:
    crs_row = dims['crs_row']
    if crs_row is not None:
        catalog['dimensions']['crs'][crs_row['crs_hash']] = crs_row['crs']
    res_row = dims['resolution_row']
    if res_row is not None:
        catalog['dimensions']['transforms'][res_row['transform_hash']] = {
            k: v for k, v in res_row.items() if k != 'transform_hash'
        }
    bounds_row = dims['bounds_row']
    if bounds_row is not None:
        catalog['dimensions']['bounds'][bounds_row['bounds_hash']] = {
            k: v for k, v in bounds_row.items() if k != 'bounds_hash'
        }
    shape_row = dims['shape_row']
    if shape_row is not None:
        catalog['dimensions']['shapes'][shape_row['shape_hash']] = shape_row['shape']


def _upsert_files(catalog: dict[str, Any], state_hash: str, display: EntryDisplay) -> None:
    refs: list[dict[str, str]] = []
    if display.primary is not None:
        refs.append({**display.primary, 'role': FileRole.PRIMARY.value})
    refs.extend({**sidecar, 'role': FileRole.SIDECAR.value} for sidecar in display.sidecars)
    if refs:
        catalog['files'][state_hash] = refs
    else:
        catalog['files'].pop(state_hash, None)


def upsert_entry(
    catalog: dict[str, Any],
    entry_dir: Path,
    record: EntryRecord,
    spec: SpatialSpec | None,
    display: EntryDisplay | None = None,
) -> None:
    """Insert or update one entry, its dimension rows, and its file refs — in memory."""
    dims = dimension_rows(spec)
    row = _entry_row(entry_dir, record, dims, display)
    if row['state_hash'] is None:
        return
    _upsert_dims(catalog, dims)
    existing = catalog['entries'].get(row['state_hash'], {'state_hash': row['state_hash']})
    existing.update({key: row[key] for key in _ENTRY_KEYS})
    catalog['entries'][row['state_hash']] = existing
    if display is not None:
        _upsert_files(catalog, row['state_hash'], display)


def iter_cache_records(
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    skip_hashes: set[str] | None = None,
) -> Iterator[tuple[Path, EntryRecord, SpatialSpec | None]]:
    """Yield ``(entry_dir, record, spec)`` for every readable cache entry on disk.

    Skips entries written by a different ``format_version``: those are stale
    by definition and are handled by clean-cache. ``skip_hashes`` skips the
    already-indexed entries — entry dirs are named by their state hash, so
    the common flat layout skips without reading ``meta.json`` at all.
    """
    resolver = CachePathResolver(roots=tuple(cache_roots))
    for meta_path in resolver.glob_meta_paths():
        if skip_hashes is not None and meta_path.parent.name in skip_hashes:
            continue
        try:
            record = EntryRecord.from_file(meta_path)
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        if skip_hashes is not None and record.state_hash in skip_hashes:
            continue
        if record.state_hash is None or record.class_name is None:
            continue
        if format_version is not None and record.format_version != format_version:
            continue
        spec = None
        spec_path = CachePathConstructor.from_path(meta_path).spec_path
        if spec_path.exists():
            try:
                spec = SpatialSpec.from_file(spec_path)
            except Exception:  # noqa: BLE001 — unparseable spec.json leaves the dims nullable
                spec = None
        yield meta_path.parent, record, spec


def backfill(
    catalog: dict[str, Any],
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    log_interval: int = BACKFILL_LOG_INTERVAL,
) -> int:
    """Index every cache entry missing from the catalog; returns the count indexed."""
    from pygeodata.catalog.display import entry_display  # noqa: PLC0415 — display imports the artifact families

    known = set(catalog['entries'])
    count = 0
    for entry_dir, record, spec in iter_cache_records(cache_roots, format_version=format_version, skip_hashes=known):
        display = None
        params_path = CachePathConstructor(entry_dir).params_path
        if params_path.exists():
            try:
                params_json = json.loads(params_path.read_text(encoding='utf-8'))
                display = entry_display(params_json, entry_dir, record.class_name, record.object_type)
            except (OSError, json.JSONDecodeError):
                display = None
        upsert_entry(catalog, entry_dir, record, spec, display)
        count += 1
        if log_interval and count % log_interval == 0:
            logger.info('Backfilled %d entries into the catalog', count)
    return count


def prune_missing(catalog: dict[str, Any]) -> int:
    """Delete entries whose entry dir (or its ``meta.json``) no longer exists."""
    stale = [
        state_hash
        for state_hash, row in catalog['entries'].items()
        if not CachePathConstructor(entry_abs_path(row['dir_relpath'])).state_hash_path.exists()
    ]
    for state_hash in stale:
        catalog['entries'].pop(state_hash, None)
        catalog['files'].pop(state_hash, None)
    return len(stale)


def bootstrap_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Most recent entry per class — the browser's instant first payload.

    ``created_at`` is the ``meta.json`` mtime, used for ordering only.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in catalog['entries'].values():
        current = latest.get(row['class_name'])
        if current is None or (row['created_at'] or 0) >= (current['created_at'] or 0):
            latest[row['class_name']] = row
    return [latest[class_name] for class_name in sorted(latest)]


# ---------------------------------------------------------------------------
# Public write paths
# ---------------------------------------------------------------------------


def index_entry(
    entry_dir: Path,
    record: EntryRecord,
    spec: SpatialSpec | None,
    params_json: dict[str, Any] | None = None,
) -> None:
    """Upsert one entry and its dimensions; a failed write warns and never fails the render.

    The display payload is derived inside the catalog from ``params_json`` —
    callers pass data only.
    """
    if not get_config().catalog_enabled:
        return
    display = None
    if params_json is not None:
        from pygeodata.catalog.display import entry_display  # noqa: PLC0415 — display imports the artifact families

        try:
            display = entry_display(params_json, entry_dir, record.class_name, record.object_type)
        except OSError as exc:
            warnings.warn(f'Catalog write failed for {record.state_hash}: {exc}', stacklevel=2)
            return
    try:
        with _catalog_lock():
            catalog = load_catalog()
            upsert_entry(catalog, entry_dir, record, spec, display)
            save_catalog(catalog)
    except OSError as exc:
        warnings.warn(f'Catalog write failed for {record.state_hash}: {exc}', stacklevel=2)


def reconcile() -> int:
    """Prune entries whose cache entry no longer exists; warns and returns 0 on failure."""
    if not get_config().catalog_enabled:
        return 0
    try:
        with _catalog_lock():
            catalog = load_catalog()
            pruned = prune_missing(catalog)
            if pruned:
                save_catalog(catalog)
    except OSError as exc:
        warnings.warn(f'Catalog prune failed: {exc}', stacklevel=2)
        return 0
    return pruned


def refresh(
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    prune: bool = True,
) -> tuple[int, int]:
    """Backfill new entries and prune entries whose entry is gone; returns (indexed, pruned).

    The reconcile pass the browser runs on open and `catalog build` shares.
    """
    if not get_config().catalog_enabled:
        return 0, 0
    with _catalog_lock():
        catalog = load_catalog()
        indexed = backfill(catalog, cache_roots, format_version=format_version)
        pruned = prune_missing(catalog) if prune else 0
        save_catalog(catalog)
    return indexed, pruned


# ---------------------------------------------------------------------------
# Pins — machine-local, next to the catalog
# ---------------------------------------------------------------------------


def pins_path() -> Path:
    """Pins are the only human-authored catalog data; they live in their own file so a
    catalog built or synced by another machine never overwrites them.
    """
    catalog = catalog_path()
    return catalog.with_name(f'{catalog.stem}.pins.json')


def _load_pins() -> list[dict[str, Any]]:
    path = pins_path()
    if not path.is_file():
        return []
    try:
        pins = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        warnings.warn(f'Pins file unreadable, starting empty: {exc}', stacklevel=2)
        return []
    return pins if isinstance(pins, list) else []


def _save_pins(pins: list[dict[str, Any]]) -> None:
    path = pins_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, pins)


def pinned_hashes() -> list[str]:
    """Pinned state hashes, oldest pin first."""
    return [pin['state_hash'] for pin in sorted(_load_pins(), key=lambda pin: pin['pinned_at'])]


def set_pinned(state_hash: str, pinned: bool) -> None:
    """Pin or unpin one entry; pinning an already pinned entry keeps its original time."""
    with FileLock(pins_path().with_suffix('.lock')):
        pins = _load_pins()
        if pinned:
            if not any(pin['state_hash'] == state_hash for pin in pins):
                pins.append({'state_hash': state_hash, 'pinned_at': time.time()})
        else:
            pins = [pin for pin in pins if pin['state_hash'] != state_hash]
        _save_pins(pins)
