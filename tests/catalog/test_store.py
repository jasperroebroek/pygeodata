import dataclasses
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from pyproj import CRS
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from pygeodata.catalog import models, store
from pygeodata.catalog.dimensions import dimension_rows
from pygeodata.catalog.entry_catalog import _spec_info_cached
from pygeodata.catalog.types import FileRole
from pygeodata.config import FORMAT_VERSION, JSONKeys
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.spec import SpatialSpec, SpecKeys
from tests.fixtures.data import SimpleLoader


def make_record(state_hash: str = 's1', class_name: str = 'SimpleTable') -> EntryRecord:
    return EntryRecord(
        class_name=class_name,
        object_type='Table',
        source_hash='src1',
        dependency_tree_hash='tree1',
        instance_hash='inst1',
        params_hash='params1',
        spec_hash='spec1',
        state_hash=state_hash,
        co_output_hashes=['co1'],
        format_version=FORMAT_VERSION,
    )


def write_entry_dir(
    cache_root: Path,
    state_hash: str,
    class_name: str = 'SimpleTable',
    spec_dict: dict | None = None,
) -> Path:
    entry_dir = cache_root / state_hash
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / 'meta.json').write_text(
        json.dumps(dataclasses.asdict(make_record(state_hash, class_name))),
        encoding='utf-8',
    )
    if spec_dict is not None:
        (entry_dir / 'spec.json').write_text(json.dumps(spec_dict), encoding='utf-8')
    return entry_dir


@pytest.fixture
def source_root(tmp_path: Path) -> Path:
    return tmp_path / '.source'


@pytest.fixture
def session(source_root: Path) -> Iterator[Session]:
    engine = store.create_catalog_engine(source_root)
    store.sync_schema(engine, FORMAT_VERSION)
    with Session(engine) as session:
        yield session


def test_upsert_entry_inserts_dimensions_once(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(session, source_root, entry_dir, make_record(), sample_spatial_spec)
    store.upsert_entry(session, source_root, entry_dir, make_record(), sample_spatial_spec)

    assert session.scalar(select(func.count()).select_from(models.Entry)) == 1
    assert session.scalar(select(func.count()).select_from(models.Crs)) == 1
    assert session.scalar(select(func.count()).select_from(models.Resolution)) == 1
    assert session.scalar(select(func.count()).select_from(models.Bounds)) == 1
    assert session.scalar(select(func.count()).select_from(models.Shape)) == 1

    entry = session.get(models.Entry, 's1')
    assert entry.class_name == 'SimpleTable'
    assert entry.dir_relpath == str(Path('data_processed') / 's1')
    assert json.loads(entry.co_output_hashes) == ['co1']
    crs = session.get(models.Crs, entry.crs_hash)
    assert crs.crs == 'EPSG:4326'
    resolution = session.get(models.Resolution, entry.transform_hash)
    assert (resolution.res_x, resolution.res_y) == (0.1, 0.1)
    bounds = session.get(models.Bounds, entry.bounds_hash)
    assert (bounds.left, bounds.bottom, bounds.right, bounds.top) == (-180.0, -90.0, 180.0, 90.0)
    assert session.get(models.Shape, entry.shape_hash).shape == '(1800, 3600)'


def test_upsert_entry_updates_existing_row(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(session, source_root, entry_dir, make_record(), sample_spatial_spec)

    record = make_record()
    record.class_name = 'RenamedTable'
    store.upsert_entry(session, source_root, entry_dir, record, sample_spatial_spec)
    assert session.get(models.Entry, 's1').class_name == 'RenamedTable'


def test_shared_dimensions_deduplicate(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    """Entries sharing a spec share dimension rows; a different spec adds only what changed."""
    d1 = write_entry_dir(tmp_path / 'data_processed', 's1')
    d2 = write_entry_dir(tmp_path / 'data_processed', 's2')
    store.upsert_entry(session, source_root, d1, make_record('s1'), sample_spatial_spec)

    other_spec = SpatialSpec(
        crs=CRS.from_string('EPSG:3857'),
        transform=sample_spatial_spec.transform,
        shape=sample_spatial_spec.shape,
    )
    store.upsert_entry(session, source_root, d2, make_record('s2', 'SimpleTable'), other_spec)

    assert session.scalar(select(func.count()).select_from(models.Entry)) == 2
    assert session.scalar(select(func.count()).select_from(models.Crs)) == 2
    assert session.scalar(select(func.count()).select_from(models.Resolution)) == 1
    assert session.scalar(select(func.count()).select_from(models.Bounds)) == 1
    assert session.scalar(select(func.count()).select_from(models.Shape)) == 1


def test_bounds_keyed_by_transform_and_shape() -> None:
    """Identical bounds from different (transform, shape) pairs stay separate rows."""
    t1 = {'a': 0.1, 'b': 0.0, 'c': -180.0, 'd': 0.0, 'e': -0.1, 'f': 90.0}
    t2 = {'a': 0.05, 'b': 0.0, 'c': -180.0, 'd': 0.0, 'e': -0.05, 'f': 90.0}
    b1 = dimension_rows(
        SpatialSpec.from_dict({SpecKeys.CRS: 'EPSG:4326', SpecKeys.TRANSFORM: t1, SpecKeys.SHAPE: [1800, 3600]}),
    )['bounds_row']
    b2 = dimension_rows(
        SpatialSpec.from_dict({SpecKeys.CRS: 'EPSG:4326', SpecKeys.TRANSFORM: t2, SpecKeys.SHAPE: [3600, 7200]}),
    )['bounds_row']

    assert b1['bounds_hash'] != b2['bounds_hash']
    assert (b1['left'], b1['bottom'], b1['right'], b1['top']) == (b2['left'], b2['bottom'], b2['right'], b2['top'])


def test_partial_spec_leaves_nullable_keys(session: Session, source_root: Path, tmp_path: Path) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(session, source_root, entry_dir, make_record(), SpatialSpec(crs=None))

    entry = session.get(models.Entry, 's1')
    assert entry.crs_hash is None
    assert entry.transform_hash is None
    assert entry.bounds_hash is None
    assert entry.shape_hash is None


def test_spec_info_keeps_crs_of_partial_spec(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(session, source_root, entry_dir, make_record(), SpatialSpec(crs=CRS.from_string('ESRI:54012')))
    session.commit()
    entry = session.get(models.Entry, 's1')
    monkeypatch.setattr(store, 'get_engine', lambda _path: session.get_bind())

    info = _spec_info_cached.__wrapped__(entry.crs_hash, entry.transform_hash, entry.shape_hash)

    assert info.crs == 'ESRI:54012'
    assert info.resolution is None
    assert info.shape is None


def test_format_version_mismatch_wipes(source_root: Path, tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    engine = store.create_catalog_engine(source_root)
    store.sync_schema(engine, FORMAT_VERSION)
    with Session(engine) as session:
        entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
        store.upsert_entry(session, source_root, entry_dir, make_record(), sample_spatial_spec)
        assert session.scalar(select(func.count()).select_from(models.Entry)) == 1

    store.sync_schema(engine, FORMAT_VERSION + 1)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(models.Entry)) == 0
        assert session.get(models.CatalogMeta, 'format_version').value == str(FORMAT_VERSION + 1)


def test_prune_missing(session: Session, source_root: Path, tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    keep = write_entry_dir(tmp_path / 'data_processed', 's1')
    gone = write_entry_dir(tmp_path / 'data_processed', 's2')
    store.upsert_entry(session, source_root, keep, make_record('s1'), sample_spatial_spec)
    store.upsert_entry(session, source_root, gone, make_record('s2'), sample_spatial_spec)

    shutil.rmtree(gone)

    assert store.prune_missing(session, source_root) == 1
    assert session.get(models.Entry, 's1') is not None
    assert session.get(models.Entry, 's2') is None


def test_backfill_indexes_and_skips_old_format(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry_dir(tmp_path / 'data_processed', 's1', spec_dict=spec_dict)
    write_entry_dir(tmp_path / 'data_processed', 's2', 'OtherTable', spec_dict=spec_dict)

    stale_dir = tmp_path / 'data_processed' / 's_old'
    stale_dir.mkdir()
    stale_record = dataclasses.asdict(make_record('s_old'))
    stale_record[JSONKeys.FORMAT_VERSION] = FORMAT_VERSION - 1
    (stale_dir / 'meta.json').write_text(json.dumps(stale_record), encoding='utf-8')

    indexed = store.backfill(session, source_root, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION)
    assert indexed == 2
    assert session.get(models.Entry, 's_old') is None
    assert session.get(models.Entry, 's1') is not None


def test_bootstrap_rows_latest_per_class(source_root: Path, tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    engine = store.create_catalog_engine(source_root)
    store.sync_schema(engine, FORMAT_VERSION)
    with Session(engine) as session:
        for state_hash, class_name, mtime in [('a1', 'A', 100.0), ('a2', 'A', 200.0), ('b1', 'B', 150.0)]:
            entry_dir = write_entry_dir(tmp_path / 'data_processed', state_hash, class_name)
            os.utime(entry_dir / 'meta.json', (mtime, mtime))
            store.upsert_entry(
                session,
                source_root,
                entry_dir,
                make_record(state_hash, class_name),
                sample_spatial_spec,
            )

    rows = store.bootstrap_rows(engine)
    assert [r['state_hash'] for r in rows] == ['a2', 'b1']


def test_write_cache_metadata_hooks_catalog(sample_spatial_spec: SpatialSpec, source_root: Path) -> None:
    artifact = SimpleLoader()
    artifact.write_cache_metadata(sample_spatial_spec)

    engine = store.get_engine(str(source_root.resolve()), format_version=FORMAT_VERSION)
    state_hash = artifact.get_state_hash(sample_spatial_spec)
    with Session(engine) as session:
        entry = session.get(models.Entry, state_hash)
        assert entry is not None
        assert entry.class_name == 'SimpleLoader'
        assert entry.spec_hash == sample_spatial_spec.get_hash()
        assert entry.dir_relpath == str(Path('data_processed') / state_hash)
        assert session.get(models.Crs, entry.crs_hash).crs == sample_spatial_spec.crs.to_string()
        resolution = session.get(models.Resolution, entry.transform_hash)
        assert (resolution.res_x, resolution.res_y) == sample_spatial_spec.resolution
        bounds = session.get(models.Bounds, entry.bounds_hash)
        spec_bounds = sample_spatial_spec.bounds
        assert (bounds.left, bounds.bottom, bounds.right, bounds.top) == (
            spec_bounds.left,
            spec_bounds.bottom,
            spec_bounds.right,
            spec_bounds.top,
        )


def test_write_hook_warns_and_continues(
    sample_spatial_spec: SpatialSpec,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsChecker,
) -> None:
    def broken_engine(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(store, 'get_engine', broken_engine)
    SimpleLoader().write_cache_metadata(sample_spatial_spec)
    assert any('Catalog write failed' in str(w.message) for w in recwarn.list)


def test_backfill_populates_display_and_files(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1', spec_dict=sample_spatial_spec.to_dict())
    (entry_dir / 'parameters.json').write_text(json.dumps({'a': {'b': 1}}), encoding='utf-8')
    (entry_dir / 'out.html').write_text('<html/>', encoding='utf-8')
    (entry_dir / 'sidecar.csv').write_text('x', encoding='utf-8')

    store.backfill(session, source_root, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION)

    entry = session.get(models.Entry, 's1')
    assert entry.rows is not None
    assert entry.rows[0]['path'] == 'a.b'
    assert entry.rows[0]['value_text'] == '1'

    file_rows = (
        session.execute(
            select(models.EntryFile).where(models.EntryFile.state_hash == 's1').order_by(models.EntryFile.relpath),
        )
        .scalars()
        .all()
    )
    assert [(f.label, f.kind, f.role) for f in file_rows] == [
        ('out.html', 'table', FileRole.PRIMARY),
        ('sidecar.csv', 'file', FileRole.SIDECAR),
    ]


def test_write_hook_stores_display_and_files(sample_spatial_spec: SpatialSpec, source_root: Path) -> None:
    artifact = SimpleLoader()
    entry_dir = artifact.resolve_cache_paths(sample_spatial_spec).directory
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / 'dummy.tif').write_bytes(b'x')
    artifact.write_cache_metadata(sample_spatial_spec)

    engine = store.get_engine(str(source_root.resolve()), format_version=FORMAT_VERSION)
    state_hash = artifact.get_state_hash(sample_spatial_spec)
    with Session(engine) as session:
        entry = session.get(models.Entry, state_hash)
        assert entry.rows is not None
        assert (
            session.scalar(
                select(func.count()).select_from(models.EntryFile).where(models.EntryFile.state_hash == state_hash),
            )
            >= 1
        )


def test_schema_fingerprint_mismatch_wipes(source_root: Path, tmp_path: Path) -> None:
    engine = store.create_catalog_engine(source_root)
    store.sync_schema(engine, FORMAT_VERSION)
    with Session(engine) as session:
        write_entry_dir(tmp_path / 'data_processed', 's1')
        record = make_record('s1')
        store.upsert_entry(session, source_root, tmp_path / 'data_processed' / 's1', record, None)
        session.execute(
            update(models.CatalogMeta).where(models.CatalogMeta.key == 'schema_fingerprint').values(value='stale'),
        )
        session.commit()

    store.sync_schema(engine, FORMAT_VERSION)
    with Session(engine) as session:
        assert session.get(models.CatalogMeta, 'schema_fingerprint').value == store.schema_fingerprint(engine)
        assert session.get(models.Entry, 's1') is None


def test_backfill_is_incremental(
    session: Session,
    source_root: Path,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry_dir(tmp_path / 'data_processed', 's1', spec_dict=spec_dict)
    assert store.backfill(session, source_root, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION) == 1

    write_entry_dir(tmp_path / 'data_processed', 's2', 'OtherTable', spec_dict=spec_dict)
    assert store.backfill(session, source_root, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION) == 1
    assert session.get(models.Entry, 's1') is not None
    assert session.get(models.Entry, 's2') is not None


def test_pins_keep_order_and_are_idempotent(session: Session) -> None:
    engine = session.get_bind()
    store.set_pinned(engine, 'b', pinned=True)
    store.set_pinned(engine, 'a', pinned=True)
    store.set_pinned(engine, 'b', pinned=True)
    assert store.pinned_hashes(engine) == ['b', 'a']

    store.set_pinned(engine, 'b', pinned=False)
    store.set_pinned(engine, 'missing', pinned=False)
    assert store.pinned_hashes(engine) == ['a']
