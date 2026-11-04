"""The only writer of the catalog (:attr:`~pygeodata.config.Config.path_catalog`).

The catalog is an index of cache entries, never a source of truth: an entry only
exists because its ``meta.json`` was written successfully, and entry validity stays
defined by the files on disk (see :mod:`pygeodata.cache`). Its layout is the
:class:`~pygeodata.catalog.types.CatalogData` contract. A catalog that is missing or
outdated is rebuilt from the cache entries by the next write; one written by a newer
pygeodata is never overwritten.

Readers open it through :func:`open_catalog`, which checks it against the entry dirs on disk:
one directory listing per cache root, as entry dirs are named by their state hash. A catalog
written from this project root is brought up to date in place; one written elsewhere (a synced
mirror of a remote project) is read-only and never triggers a scan of the entry dirs.

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
import os
import platform
import socket
import time
import uuid
import warnings
from dataclasses import dataclass, field
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

    from filelock import BaseFileLock

logger = logging.getLogger(__name__)

BACKFILL_LOG_INTERVAL = 2000


def catalog_path() -> Path:
    return get_config().path_catalog


def _local_root() -> str:
    return str(catalog_path().parent.resolve())


def _writer() -> CatalogWriter:
    return CatalogWriter(
        root=_local_root(),
        host=socket.gethostname(),
        pygeodata=__version__,
        python=platform.python_version(),
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
    """The catalog to update, and whether it was rebuilt from the cache entries.

    A missing or outdated catalog is rebuilt in full before any change, so a reader never sees a
    partial index.
    """
    try:
        return CatalogData.from_file(catalog_path()), False
    except FileNotFoundError:
        catalog = _new_catalog()
    except CatalogOutdatedError as exc:
        logger.info('Rebuilding the catalog: %s', exc)
        catalog = _new_catalog(exc.catalog_id)
    backfill(catalog, CachePathResolver(get_config().cache_roots).iter_entry_dirs(), format_version=FORMAT_VERSION)
    return catalog, True


def _save(catalog: CatalogData) -> None:
    path = catalog_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog.header.writer = _writer()
    catalog.header.roots = _roots()
    catalog.dump(path)


def _lock_for(path: Path) -> BaseFileLock:
    """Next to the resolved file, so every symlink to it shares one lock."""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return file_lock(target.with_name(f'{target.name}.lock'))


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
        files=find_output_files(entry_path, record.output_ext),
        created_at=created_at,
        updated_at=now,
    )


@dataclass
class SkippedDirs:
    """Why entry dirs were left out of the index; each is a dir that is no usable entry."""

    # written by a different cache format, stale by definition
    other_format: int = 0
    # no meta.json: an interrupted run
    no_meta: int = 0
    # a meta.json that cannot be read or decoded
    unreadable: int = 0


def iter_cache_records(
    entry_dirs: Iterable[Path],
    format_version: int | None = None,
    skipped: SkippedDirs | None = None,
) -> Iterator[tuple[Path, EntryRecord]]:
    """Yield ``(entry_dir, record)`` for each of ``entry_dirs`` holding a readable ``meta.json``.

    Skips entries written by a different ``format_version``: those are stale
    by definition and are handled by clean-cache. The reasons for skipping are tallied in ``skipped``.
    """
    skipped = skipped if skipped is not None else SkippedDirs()
    for entry_dir in entry_dirs:
        try:
            record = EntryRecord.from_file(CachePathConstructor(entry_dir).state_hash_path)
        except FileNotFoundError:
            skipped.no_meta += 1
            continue
        except (OSError, msgspec.DecodeError):
            skipped.unreadable += 1
            continue
        if format_version is not None and record.format_version != format_version:
            skipped.other_format += 1
            continue
        yield entry_dir, record


def backfill(
    catalog: CatalogData,
    entry_dirs: Iterable[Path],
    format_version: int | None = None,
    skipped: SkippedDirs | None = None,
) -> int:
    """Index the entries in ``entry_dirs`` that the catalog lacks; returns the count indexed.

    Entry dirs are named by their state hash, so indexed entries are skipped without reading
    their ``meta.json``.
    """
    count = 0
    unknown = (d for d in entry_dirs if d.name not in catalog.entries)
    for path, record in iter_cache_records(unknown, format_version=format_version, skipped=skipped):
        if record.state_hash in catalog.entries:
            continue
        paths = CachePathConstructor(path)
        try:
            params_json = json.loads(paths.params_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            params_json = None
        try:
            spec = SpecRecord.from_file(paths.spec_path)
        except (OSError, msgspec.DecodeError):
            spec = None
        catalog.upsert(catalog_entry(path, record, params_json), spec)
        count += 1
        if count % BACKFILL_LOG_INTERVAL == 0:
            logger.info('Backfilled %d entries into the catalog', count)
    return count


# ---------------------------------------------------------------------------
# Catalog against disk — pruning only on evidence that an entry was deleted
# ---------------------------------------------------------------------------


def entry_root(entry: CatalogEntry) -> Path | None:
    """The directory whose contents are the evidence for this entry's existence."""
    path = Path(entry.dir_relpath)
    if path.is_absolute():
        return path.parent
    return _cache_root(entry.record.object_type)


def _list_dirs(root: Path) -> set[str]:
    try:
        with os.scandir(root) as it:
            return {e.name for e in it if e.is_dir()}
    except OSError:
        return set()


@dataclass
class CatalogDiff:
    """How the catalog differs from the entry dirs on disk."""

    # entry dirs the catalog lacks: new entries, or dirs that are no current entry (yet)
    unindexed: list[Path] = field(default_factory=list)
    # why those dirs are no entry; None when the dirs were not read (a read-only catalog)
    unindexed_reasons: SkippedDirs | None = None
    # entries whose dir is gone from a root that is otherwise present
    missing: list[str] = field(default_factory=list)
    # roots that are missing or empty — an unmounted share, an unsynced mirror — whose entries
    # are kept, unverified
    unavailable_roots: list[Path] = field(default_factory=list)


def _indexed_by_root(catalog: CatalogData) -> dict[Path, dict[str, str]]:
    """Entry dir name → state hash, per root (an absolute entry dir's root is its parent)."""
    by_root: dict[Path, dict[str, str]] = {}
    for entry in catalog.entries.values():
        root = entry_root(entry)
        if root is not None:
            by_root.setdefault(root, {})[Path(entry.dir_relpath).name] = entry.state_hash
    return by_root


def diff_against_disk(catalog: CatalogData) -> CatalogDiff:
    """Compare the catalog with one listing per root; touches no entry dir."""
    roots = get_config().cache_roots
    nested = {child.name for root in roots for child in roots if child.parent == root}
    by_root = _indexed_by_root(catalog)

    diff = CatalogDiff()
    for root in dict.fromkeys([*roots, *by_root]):
        names = _list_dirs(root)
        indexed = by_root.get(root, {})
        if root in roots:
            diff.unindexed.extend(root / name for name in sorted(names - indexed.keys() - nested))
        if not names:
            if indexed:
                diff.unavailable_roots.append(root)
            continue
        diff.missing.extend(state_hash for name, state_hash in indexed.items() if name not in names)
    return diff


@dataclass
class PruneResult:
    pruned: int = 0
    unavailable_roots: list[Path] = field(default_factory=list)


def prune_missing(catalog: CatalogData, diff: CatalogDiff | None = None) -> PruneResult:
    """Delete entries whose dir is gone from a root that is otherwise present."""
    diff = diff if diff is not None else diff_against_disk(catalog)
    if diff.unavailable_roots:
        warnings.warn(
            f'Cache root missing or empty, its catalog entries are kept: {", ".join(map(str, diff.unavailable_roots))}',
            stacklevel=2,
        )
    return PruneResult(pruned=catalog.prune(diff.missing), unavailable_roots=diff.unavailable_roots)


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
# Readers
# ---------------------------------------------------------------------------


@dataclass
class CatalogStatus:
    read_only: bool
    # what still differs from disk after any update
    diff: CatalogDiff = field(default_factory=CatalogDiff)


def _sync() -> tuple[CatalogData, CatalogDiff]:
    """Bring the catalog file up to date with the entry dirs; saves only on change."""
    with _lock_for(catalog_path()):
        catalog, rebuilt = _load_for_write()
        diff = diff_against_disk(catalog)
        skipped = SkippedDirs()
        indexed = backfill(catalog, diff.unindexed, format_version=FORMAT_VERSION, skipped=skipped)
        pruned = catalog.prune(diff.missing)
        if rebuilt or indexed or pruned:
            _save(catalog)
    diff.unindexed_reasons = skipped
    diff.unindexed = [d for d in diff.unindexed if d.name not in catalog.entries]
    diff.missing = []
    return catalog, diff


def _needs_sync(diff: CatalogDiff) -> bool:
    """Whether the diff holds anything to index or prune; reads only the unindexed dirs' meta.json.

    When nothing is to be indexed every unindexed dir has been read, so the reasons are set on the diff.
    """
    if diff.missing:
        return True
    skipped = SkippedDirs()
    if next(iter_cache_records(diff.unindexed, FORMAT_VERSION, skipped), None) is not None:
        return True
    diff.unindexed_reasons = skipped
    return False


def _written_elsewhere(catalog: CatalogData | None, outdated: CatalogOutdatedError | None) -> bool:
    """Whether the catalog file was written from another project root; unknown counts as yes."""
    if catalog is not None:
        return catalog.header.writer.root != _local_root()
    if outdated is not None:
        return outdated.writer_root != _local_root()
    return False


def _open_read_only(
    catalog: CatalogData | None,
    outdated: CatalogOutdatedError | None,
) -> tuple[CatalogData, CatalogStatus]:
    if outdated is not None:
        raise CatalogOutdatedError(
            f'{outdated}; it is read-only here, so run `pygeodata catalog build` where it is written',
            catalog_id=outdated.catalog_id,
            writer_root=outdated.writer_root,
        )
    if catalog is None:
        catalog = _new_catalog()
        backfill(catalog, CachePathResolver(get_config().cache_roots).iter_entry_dirs(), format_version=FORMAT_VERSION)
        return catalog, CatalogStatus(read_only=True)
    return catalog, CatalogStatus(read_only=True, diff=diff_against_disk(catalog))


def open_catalog(read_only: bool | None = None) -> tuple[CatalogData, CatalogStatus]:
    """The catalog for readers, checked against the entry dirs on disk.

    ``read_only`` defaults to True when catalog writes are disabled or the catalog was written
    from another project root (a synced mirror). A writable catalog is updated in place: missing
    or outdated → rebuilt, otherwise new entry dirs indexed and deleted ones pruned. A read-only
    one is used as it is and its difference reported; one that is outdated raises
    :class:`~pygeodata.catalog.types.CatalogOutdatedError`, since rebuilding would scan every
    entry dir. Without a catalog file a read-only reader indexes the entries in memory.
    Raises :class:`~pygeodata.catalog.types.CatalogNewerError` for a catalog written by a newer pygeodata.
    """
    outdated: CatalogOutdatedError | None = None
    try:
        catalog: CatalogData | None = CatalogData.from_file(catalog_path())
    except FileNotFoundError:
        catalog = None
    except CatalogOutdatedError as exc:
        catalog, outdated = None, exc

    if read_only is None:
        read_only = not get_config().catalog_enabled or _written_elsewhere(catalog, outdated)
    if read_only:
        return _open_read_only(catalog, outdated)

    if catalog is not None:
        diff = diff_against_disk(catalog)
        if not _needs_sync(diff):
            return catalog, CatalogStatus(read_only=False, diff=diff)
    try:
        catalog, diff = _sync()
    except OSError as exc:
        warnings.warn(f'Catalog update failed, reading it as it is: {exc}', stacklevel=2)
        return _open_read_only(catalog, outdated)
    return catalog, CatalogStatus(read_only=False, diff=diff)


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
        with _lock_for(catalog_path()):
            catalog, _ = _load_for_write()
            catalog.upsert(entry, spec)
            _save(catalog)
    except (OSError, CatalogNewerError) as exc:
        warnings.warn(f'Catalog write failed for {record.state_hash}: {exc}', stacklevel=2)


def remove_entries(state_hashes: Iterable[str]) -> int:
    """Drop the entries the caller deleted from disk; warns and returns 0 on failure."""
    hashes = set(state_hashes)
    if not hashes or not get_config().catalog_enabled or not catalog_path().is_file():
        return 0
    try:
        with _lock_for(catalog_path()):
            catalog, rebuilt = _load_for_write()
            removed = catalog.prune(hashes)
            if removed or rebuilt:
                _save(catalog)
    except (OSError, CatalogNewerError) as exc:
        warnings.warn(f'Catalog update failed: {exc}', stacklevel=2)
        return 0
    return removed


@dataclass
class RefreshResult:
    indexed: int = 0
    prune: PruneResult = field(default_factory=PruneResult)


def refresh(
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    prune: bool = True,
) -> RefreshResult:
    """Backfill new entries and prune entries whose entry is gone; saves only on change.

    The pass the browser runs on open and `catalog build` shares. Raises
    :class:`~pygeodata.catalog.types.CatalogNewerError` rather than overwrite a newer catalog;
    other write failures warn.
    """
    if not get_config().catalog_enabled:
        return RefreshResult()
    try:
        with _lock_for(catalog_path()):
            catalog, rebuilt = _load_for_write()
            indexed = len(catalog.entries) if rebuilt else 0
            indexed += backfill(
                catalog,
                CachePathResolver(tuple(cache_roots)).iter_entry_dirs(),
                format_version=format_version,
            )
            pruning = prune_missing(catalog) if prune else PruneResult()
            if rebuilt or indexed or pruning.pruned:
                _save(catalog)
    except OSError as exc:
        warnings.warn(f'Catalog refresh failed: {exc}', stacklevel=2)
        return RefreshResult()
    return RefreshResult(indexed=indexed, prune=pruning)


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
    with _lock_for(pins_path()):
        pins = _load_pins()
        if pinned:
            if not any(pin.state_hash == state_hash for pin in pins):
                pins.append(Pin(state_hash=state_hash, pinned_at=time.time()))
        else:
            pins = [pin for pin in pins if pin.state_hash != state_hash]
        _save_pins(pins)
