"""The only writer of the catalog (:attr:`~pygeodata.config.Config.path_catalog`).

The catalog is an index of cache entries, never a source of truth: an entry only
exists because its ``meta.json`` was written successfully, and entry validity stays
defined by the files on disk (see :mod:`pygeodata.cache`). Its layout is the
:class:`~pygeodata.catalog.types.CatalogData` contract. A catalog that is missing or
outdated is rebuilt from the cache entries by the next write; one written by a newer
pygeodata is never overwritten.

All write paths are no-ops while ``catalog_enabled`` is False on the config —
e.g. on shared filesystems. Writes take a ``filelock`` and rewrite the file
atomically (temp file + rename).

Entry dirs are stored relative to the cache root of their object type, so a catalog
built on one machine resolves on another that mounts the cache elsewhere; entries
outside that root keep absolute paths.
"""

from __future__ import annotations

import json
import logging
import socket
import time
import uuid
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec

from pygeodata._version import CATALOG_VERSION, FORMAT_VERSION, __version__
from pygeodata.catalog.display import find_output_files, param_rows
from pygeodata.catalog.types import (
    CatalogData,
    CatalogEntry,
    CatalogHeader,
    CatalogNewerError,
    CatalogOutdatedError,
    CatalogWriter,
    Pin,
    schema_fingerprint,
)
from pygeodata.config import get_config
from pygeodata.enums import ObjectType
from pygeodata.fileio import file_lock, read_typed_json, write_typed_json_atomic
from pygeodata.paths import CachePathConstructor, CachePathResolver
from pygeodata.records import EntryRecord, SpecRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

BACKFILL_LOG_INTERVAL = 2000


def catalog_path() -> Path:
    return get_config().path_catalog


def read_catalog() -> CatalogData | None:
    """The catalog for readers; None when there is none or it is outdated.

    Raises :class:`~pygeodata.catalog.types.CatalogNewerError` for a catalog written by a newer pygeodata.
    """
    try:
        return CatalogData.from_file(catalog_path())
    except FileNotFoundError:
        return None
    except (OSError, CatalogOutdatedError) as exc:
        warnings.warn(f'Catalog not used, reading the cache entries instead: {exc}', stacklevel=2)
        return None


def _writer() -> CatalogWriter:
    return CatalogWriter(
        root=str(catalog_path().parent.resolve()),
        host=socket.gethostname(),
        pygeodata=__version__,
        written_at=time.time(),
    )


def _roots() -> dict[str, str]:
    config = get_config()
    return {t.value: str(config.get_cache_root(t).resolve()) for t in ObjectType}


def _new_catalog(catalog_id: str | None = None) -> CatalogData:
    header = CatalogHeader(
        catalog_version=CATALOG_VERSION,
        schema_fingerprint=schema_fingerprint(),
        catalog_id=catalog_id or uuid.uuid4().hex,
        writer=_writer(),
        roots=_roots(),
    )
    return CatalogData(header=header, entries={}, specs={})


def _load_for_write() -> tuple[CatalogData, bool]:
    """The catalog to update, and whether it must first be rebuilt from the cache entries."""
    try:
        return CatalogData.from_file(catalog_path()), False
    except FileNotFoundError:
        return _new_catalog(), True
    except CatalogOutdatedError as exc:
        logger.info('Rebuilding the catalog: %s', exc)
        return _new_catalog(exc.catalog_id), True


def _save(catalog: CatalogData) -> None:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.header.writer = _writer()
    catalog.header.roots = _roots()
    catalog.dump(path)


@contextmanager
def _catalog_lock() -> Iterator[None]:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_name(f'{path.name}.lock')):
        yield


# ---------------------------------------------------------------------------
# Entry dirs — relative to the local cache root of the entry's object type
# ---------------------------------------------------------------------------


def _cache_root(object_type: str) -> Path | None:
    try:
        return get_config().get_cache_root(ObjectType(object_type))
    except ValueError:
        return None


def dir_relpath(entry_dir: Path, object_type: str) -> str:
    resolved = Path(entry_dir).resolve()
    root = _cache_root(object_type)
    if root is not None:
        try:
            return str(resolved.relative_to(root.resolve()))
        except ValueError:
            pass
    return str(resolved)


def entry_dir(entry: CatalogEntry) -> Path:
    """The entry's directory on this machine; touches no files."""
    path = Path(entry.dir_relpath)
    root = _cache_root(entry.record.object_type)
    if path.is_absolute() or root is None:
        return path
    return (root / path).absolute()


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def catalog_entry(entry_path: Path, record: EntryRecord, params_json: dict[str, Any] | None) -> CatalogEntry:
    rows, linked = param_rows(params_json) if params_json is not None else ([], [])
    now = time.time()
    try:
        created_at = CachePathConstructor(entry_path).state_hash_path.stat().st_mtime
    except OSError:
        created_at = now
    return CatalogEntry(
        record=record,
        dir_relpath=dir_relpath(entry_path, record.object_type),
        rows=rows,
        linked_entries=linked,
        files=find_output_files(entry_path, record.class_name, record.object_type),
        created_at=created_at,
        updated_at=now,
    )


def iter_cache_records(
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    skip_hashes: set[str] | None = None,
) -> Iterator[tuple[Path, EntryRecord, SpecRecord | None]]:
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
        except (OSError, msgspec.DecodeError):
            continue
        if skip_hashes is not None and record.state_hash in skip_hashes:
            continue
        if format_version is not None and record.format_version != format_version:
            continue
        try:
            spec = SpecRecord.from_file(CachePathConstructor.from_path(meta_path).spec_path)
        except (OSError, msgspec.DecodeError):
            spec = None
        yield meta_path.parent, record, spec


def backfill(
    catalog: CatalogData,
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    log_interval: int = BACKFILL_LOG_INTERVAL,
) -> int:
    """Index every cache entry missing from the catalog; returns the count indexed."""
    count = 0
    skip = set(catalog.entries)
    for path, record, spec in iter_cache_records(cache_roots, format_version=format_version, skip_hashes=skip):
        try:
            params_json = json.loads(CachePathConstructor(path).params_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            params_json = None
        catalog.upsert(catalog_entry(path, record, params_json), spec)
        count += 1
        if log_interval and count % log_interval == 0:
            logger.info('Backfilled %d entries into the catalog', count)
    return count


def prune_missing(catalog: CatalogData) -> int:
    """Delete entries whose entry dir (or its ``meta.json``) no longer exists."""
    return catalog.prune(
        state_hash
        for state_hash, entry in catalog.entries.items()
        if not CachePathConstructor(entry_dir(entry)).state_hash_path.exists()
    )


def bootstrap_rows(catalog: CatalogData) -> list[CatalogEntry]:
    """Most recent entry per class — the browser's instant first payload.

    ``created_at`` is the ``meta.json`` mtime, used for ordering only.
    """
    latest: dict[str, CatalogEntry] = {}
    for entry in catalog.entries.values():
        current = latest.get(entry.record.class_name)
        if current is None or entry.created_at >= current.created_at:
            latest[entry.record.class_name] = entry
    return [latest[class_name] for class_name in sorted(latest)]


# ---------------------------------------------------------------------------
# Public write paths
# ---------------------------------------------------------------------------


def index_entry(
    entry_path: Path,
    record: EntryRecord,
    spec: SpecRecord,
    params_json: dict[str, Any] | None = None,
) -> None:
    """Upsert one entry and its spec; a failed write warns and never fails the render."""
    if not get_config().catalog_enabled:
        return
    try:
        entry = catalog_entry(entry_path, record, params_json)
        with _catalog_lock():
            catalog, rebuild = _load_for_write()
            if rebuild:
                backfill(catalog, get_config().cache_roots, format_version=FORMAT_VERSION)
            catalog.upsert(entry, spec)
            _save(catalog)
    except (OSError, CatalogNewerError) as exc:
        warnings.warn(f'Catalog write failed for {record.state_hash}: {exc}', stacklevel=2)


def reconcile() -> int:
    """Prune entries whose cache entry no longer exists; warns and returns 0 on failure."""
    if not get_config().catalog_enabled or not catalog_path().is_file():
        return 0
    try:
        with _catalog_lock():
            catalog, rebuild = _load_for_write()
            if rebuild:
                backfill(catalog, get_config().cache_roots, format_version=FORMAT_VERSION)
            pruned = prune_missing(catalog)
            if pruned or rebuild:
                _save(catalog)
    except (OSError, CatalogNewerError) as exc:
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
        catalog, _ = _load_for_write()
        indexed = backfill(catalog, cache_roots, format_version=format_version)
        pruned = prune_missing(catalog) if prune else 0
        _save(catalog)
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


def _load_pins() -> list[Pin]:
    path = pins_path()
    if not path.is_file():
        return []
    try:
        return read_typed_json(path, list[Pin])
    except (OSError, msgspec.DecodeError) as exc:
        warnings.warn(f'Pins file unreadable, starting empty: {exc}', stacklevel=2)
        return []


def _save_pins(pins: list[Pin]) -> None:
    path = pins_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_typed_json_atomic(path, pins)


def pinned_hashes() -> list[str]:
    """Pinned state hashes, oldest pin first."""
    return [pin.state_hash for pin in sorted(_load_pins(), key=lambda pin: pin.pinned_at)]


def set_pinned(state_hash: str, pinned: bool) -> None:
    """Pin or unpin one entry; pinning an already pinned entry keeps its original time."""
    with file_lock(pins_path().with_suffix('.lock')):
        pins = _load_pins()
        if pinned:
            if not any(pin.state_hash == state_hash for pin in pins):
                pins.append(Pin(state_hash=state_hash, pinned_at=time.time()))
        else:
            pins = [pin for pin in pins if pin.state_hash != state_hash]
        _save_pins(pins)
