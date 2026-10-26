"""SQLAlchemy access layer for ``.source/catalog.db``.

The catalog is an index of cache entries, never a source of truth: a row only
exists because the entry's ``meta.json`` was written successfully, and entry
validity stays defined by the JSON files on disk (see :mod:`pygeodata.cache`).
The active ``FORMAT_VERSION`` is recorded once in the ``meta`` table — a
catalog built by a different version is wiped and rebuilt lazily, never
migrated. Entries are keyed by their content-addressed ``state_hash``, so
concurrent writers producing the same hash converge on identical rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, delete, event, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from pygeodata.catalog import models
from pygeodata.catalog.dimensions import dimension_rows
from pygeodata.catalog.types import FileRole
from pygeodata.paths import CachePathResolver
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.spec import SpatialSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from pygeodata.catalog.display import EntryDisplay

logger = logging.getLogger(__name__)

CATALOG_FILENAME = 'catalog.db'

_ENTRY_UPDATE_COLUMNS = (
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

_META_FORMAT_VERSION = 'format_version'
_META_SCHEMA_VERSION = 'schema_fingerprint'


def catalog_path(source_root: Path) -> Path:
    return Path(source_root) / CATALOG_FILENAME


def create_catalog_engine(source_root: Path) -> Engine:
    path = catalog_path(source_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f'sqlite:///{path}')

    @event.listens_for(engine, 'connect')
    def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL')
        cursor.execute('PRAGMA busy_timeout=30000')
        cursor.execute('PRAGMA synchronous=NORMAL')
        cursor.close()

    return engine


def schema_fingerprint(engine: Engine) -> str:
    """Fingerprint of the current table DDL — any model change changes it.

    No schema version numbers: the catalog is disposable, so a fingerprint
    mismatch wipes it and the next backfill rebuilds.
    """
    ddl = '\n'.join(
        str(CreateTable(table).compile(dialect=engine.dialect)).strip()
        for table in sorted(models.Base.metadata.tables.values(), key=lambda t: t.name)
    )
    return hashlib.sha256(ddl.encode()).hexdigest()


def sync_schema(engine: Engine, format_version: int) -> None:
    """Create tables and enforce the one-version contract.

    Wipes the catalog when ``FORMAT_VERSION`` changed or when the table DDL
    no longer matches the stored fingerprint; the next backfill rebuilds it.
    """
    models.Base.metadata.create_all(engine)
    with Session(engine) as session:
        format_row = session.get(models.CatalogMeta, _META_FORMAT_VERSION)
        schema_row = session.get(models.CatalogMeta, _META_SCHEMA_VERSION)
        stored_format = format_row.value if format_row is not None else None
        stored_fingerprint = schema_row.value if schema_row is not None else None
        fingerprint = schema_fingerprint(engine)
        if stored_format != str(format_version) or stored_fingerprint != fingerprint:
            if stored_format is not None or stored_fingerprint is not None:
                logger.info(
                    'Catalog built under format %s / schema %s, current is %s / %s — wiping and rebuilding',
                    stored_format,
                    stored_fingerprint[:12] if stored_fingerprint else None,
                    format_version,
                    fingerprint[:12],
                )
            session.commit()
            models.Base.metadata.drop_all(engine)
            models.Base.metadata.create_all(engine)
            with Session(engine) as fresh:
                fresh.add(models.CatalogMeta(key=_META_FORMAT_VERSION, value=str(format_version)))
                fresh.add(models.CatalogMeta(key=_META_SCHEMA_VERSION, value=str(fingerprint)))
                fresh.commit()


@cache
def get_engine(source_root: str, format_version: int | None = None) -> Engine:
    """Process-wide engine for one resolved ``.source`` path."""
    engine = create_catalog_engine(Path(source_root))
    if format_version is not None:
        sync_schema(engine, format_version)
    return engine


def _relpath(entry_dir: Path, source_root: Path) -> str:
    """Relative to the project base — the parent of ``.source``."""
    base = Path(source_root).parent
    try:
        return str(entry_dir.relative_to(base))
    except ValueError:
        return str(entry_dir.resolve())


def entry_abs_path(source_root: Path, dir_relpath: str) -> Path:
    path = Path(dir_relpath)
    if path.is_absolute():
        return path
    return Path(source_root).parent / path


def _entry_row(
    source_root: Path,
    entry_dir: Path,
    record: EntryRecord,
    dims: dict[str, Any],
    display: EntryDisplay | None,
) -> dict[str, Any]:
    try:
        created_at = (Path(entry_dir) / 'meta.json').stat().st_mtime
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
        'co_output_hashes': json.dumps(record.co_output_hashes or []),
        'rows': display.rows if display is not None else None,
        'linked_entries': display.linked_entries if display is not None else None,
        'dir_relpath': _relpath(entry_dir, source_root),
        'created_at': created_at,
        'updated_at': created_at,
    }


def _upsert_dims(session: Session, dims: dict[str, Any]) -> None:
    for table, row_key in (
        (models.Crs, 'crs_row'),
        (models.Resolution, 'resolution_row'),
        (models.Bounds, 'bounds_row'),
        (models.Shape, 'shape_row'),
    ):
        row = dims[row_key]
        if row is not None:
            session.execute(
                sqlite_insert(table).values(**row).on_conflict_do_nothing(),
            )


def upsert_entry(  # noqa: PLR0913
    session: Session,
    source_root: Path,
    entry_dir: Path,
    record: EntryRecord,
    spec: SpatialSpec | None,
    display: EntryDisplay | None = None,
) -> None:
    """Insert or update one entry and its dimension rows."""
    _upsert_entry(session, source_root, entry_dir, record, spec, display)
    session.commit()


def _upsert_entry(  # noqa: PLR0913
    session: Session,
    source_root: Path,
    entry_dir: Path,
    record: EntryRecord,
    spec: SpatialSpec | None,
    display: EntryDisplay | None,
) -> None:
    dims = dimension_rows(spec)
    row = _entry_row(source_root, entry_dir, record, dims, display)
    if row['state_hash'] is None:
        return
    _upsert_dims(session, dims)
    updates = {c: row[c] for c in _ENTRY_UPDATE_COLUMNS}
    session.execute(
        sqlite_insert(models.Entry).values(**row).on_conflict_do_update(index_elements=['state_hash'], set_=updates),
    )
    if display is not None:
        _upsert_files(session, row['state_hash'], display)


def _upsert_files(session: Session, state_hash: str, display: EntryDisplay) -> None:
    """Replace the file rows for one entry — role per file, no sidecar lists."""
    session.execute(delete(models.EntryFile).where(models.EntryFile.state_hash == state_hash))
    refs: list[tuple[dict, FileRole]] = []
    if display.primary is not None:
        refs.append((display.primary, FileRole.PRIMARY))
    refs.extend((sidecar, FileRole.SIDECAR) for sidecar in display.sidecars)
    if not refs:
        return
    session.execute(
        sqlite_insert(models.EntryFile),
        [
            {
                'state_hash': state_hash,
                'relpath': ref['relpath'],
                'label': ref['label'],
                'kind': ref['kind'],
                'role': role,
            }
            for ref, role in refs
        ],
    )


def delete_entries(session: Session, state_hashes: Iterable[str]) -> None:
    hashes = list(state_hashes)
    session.execute(delete(models.EntryFile).where(models.EntryFile.state_hash.in_(hashes)))
    session.execute(delete(models.Entry).where(models.Entry.state_hash.in_(hashes)))
    session.commit()


def prune_missing(session: Session, source_root: Path) -> int:
    """Delete rows whose entry dir (or its ``meta.json``) no longer exists.

    The DB lives in ``.source`` and survives cache deletion, so this
    reconcile keeps the index honest after cleaning or manual deletion.
    """
    rows = session.execute(select(models.Entry.state_hash, models.Entry.dir_relpath)).all()
    stale = [
        state_hash
        for state_hash, dir_relpath in rows
        if not (entry_abs_path(source_root, dir_relpath) / 'meta.json').exists()
    ]
    if stale:
        delete_entries(session, stale)
    return len(stale)


def iter_cache_records(
    cache_roots: Iterable[Path],
    format_version: int | None = None,
    skip_hashes: set[str] | None = None,
) -> Iterator[tuple[Path, EntryRecord, SpatialSpec | None]]:
    """Yield ``(entry_dir, record, spec)`` for every readable cache entry on disk.

    Skips entries written by a different ``format_version``: those are stale
    by definition and are handled by clean-cache. ``skip_hashes`` skips
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
        spec_path = meta_path.parent / 'spec.json'
        if spec_path.exists():
            try:
                spec = SpatialSpec.from_file(spec_path)
            except Exception:  # noqa: BLE001 — unparseable spec.json leaves the dims nullable
                spec = None
        yield meta_path.parent, record, spec


BACKFILL_BATCH_SIZE = 2000


def backfill(
    session: Session,
    source_root: Path,
    cache_roots: Iterable[Path],
    format_version: int | None = None,
) -> int:
    from pygeodata.catalog.display import entry_display  # noqa: PLC0415 — display imports the artifact families

    known = set(session.scalars(select(models.Entry.state_hash)))
    count = 0
    for entry_dir, record, spec in iter_cache_records(cache_roots, format_version=format_version, skip_hashes=known):
        display = None
        params_path = entry_dir / 'parameters.json'
        if params_path.exists():
            try:
                params_json = json.loads(params_path.read_text(encoding='utf-8'))
                display = entry_display(params_json, entry_dir, source_root, record.class_name, record.object_type)
            except (OSError, json.JSONDecodeError):
                display = None
        _upsert_entry(session, source_root, entry_dir, record, spec, display)
        count += 1
        if count % BACKFILL_BATCH_SIZE == 0:
            session.commit()
    session.commit()
    return count


def bootstrap_rows(engine: Engine) -> list[dict[str, Any]]:
    """Most recent entry per class — the browser's instant first payload.

    ``created_at`` is the ``meta.json`` mtime, used for ordering only.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                'SELECT * FROM entries WHERE rowid IN ('
                '  SELECT rowid FROM ('
                '    SELECT rowid, ROW_NUMBER() OVER ('
                '      PARTITION BY class_name ORDER BY created_at DESC, rowid DESC'
                '    ) AS rn FROM entries'
                '  ) WHERE rn = 1'
                ') ORDER BY class_name',
            ),
        ).mappings()
        return [dict(row) for row in rows]


def index_entry(
    source_root: Path,
    entry_dir: Path,
    record: EntryRecord,
    spec: SpatialSpec | None,
    params_json: dict[str, Any] | None = None,
) -> None:
    """Upsert one entry and its dimensions; a failed write warns and never fails the render.

    The display payload is derived inside the catalog from ``params_json`` —
    callers pass data only.
    """
    display = None
    if params_json is not None:
        from pygeodata.catalog.display import entry_display  # noqa: PLC0415 — display imports the artifact families

        display = entry_display(params_json, entry_dir, source_root, record.class_name, record.object_type)
    try:
        engine = get_engine(str(Path(source_root).resolve()), format_version=record.format_version)
        with Session(engine) as session:
            upsert_entry(session, source_root, entry_dir, record, spec, display)
    except (SQLAlchemyError, OSError) as exc:
        warnings.warn(f'Catalog write failed for {record.state_hash}: {exc}', stacklevel=2)


def reconcile(source_root: Path, format_version: int | None = None) -> int:
    """Prune rows whose cache entry no longer exists; warns and returns 0 on failure."""
    try:
        engine = get_engine(str(Path(source_root).resolve()), format_version=format_version)
        with Session(engine) as session:
            return prune_missing(session, source_root)
    except (SQLAlchemyError, OSError) as exc:
        warnings.warn(f'Catalog prune failed: {exc}', stacklevel=2)
        return 0


def refresh(
    source_root: Path,
    cache_roots: Iterable[Path],
    format_version: int | None = None,
) -> tuple[int, int]:
    """Backfill new entries and prune rows whose entry is gone; returns (indexed, pruned).

    The reconcile pass the browser runs on open and `catalog build` shares.
    """
    engine = get_engine(str(Path(source_root).resolve()), format_version=format_version)
    with Session(engine) as session:
        indexed = backfill(session, source_root, cache_roots, format_version=format_version)
        pruned = prune_missing(session, source_root)
    return indexed, pruned
