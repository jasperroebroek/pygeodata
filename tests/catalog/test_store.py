import dataclasses
import json
import os
from pathlib import Path

import pytest
from pyproj import CRS

from pygeodata.catalog import store
from pygeodata.catalog.dimensions import dimension_rows
from pygeodata.catalog.entry_catalog import _spec_info_cached
from pygeodata.catalog.types import FileRole
from pygeodata.config import FORMAT_VERSION, Config, get_config, set_config
from pygeodata.enums import JSONKeys, SpecKeys
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.spec import SpatialSpec
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
def catalog() -> dict:
    return store._empty_catalog()


def test_upsert_entry_inserts_dimensions_once(
    catalog: dict,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(catalog, entry_dir, make_record(), sample_spatial_spec)
    store.upsert_entry(catalog, entry_dir, make_record(), sample_spatial_spec)

    assert list(catalog['entries']) == ['s1']
    dims = catalog['dimensions']
    assert len(dims['crs']) == 1
    assert len(dims['transforms']) == 1
    assert len(dims['bounds']) == 1
    assert len(dims['shapes']) == 1

    entry = catalog['entries']['s1']
    assert entry['class_name'] == 'SimpleTable'
    assert entry['dir_relpath'] == 's1'
    assert entry['co_output_hashes'] == ['co1']
    assert dims['crs'][entry['crs_hash']] == 'EPSG:4326'
    transform = dims['transforms'][entry['transform_hash']]
    assert (transform['res_x'], transform['res_y']) == (0.1, 0.1)
    bounds = dims['bounds'][entry['bounds_hash']]
    assert (bounds['left'], bounds['bottom'], bounds['right'], bounds['top']) == (-180.0, -90.0, 180.0, 90.0)
    assert dims['shapes'][entry['shape_hash']] == '(1800, 3600)'


def test_upsert_entry_updates_existing_row(
    catalog: dict,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(catalog, entry_dir, make_record(), sample_spatial_spec)

    record = make_record()
    record.class_name = 'RenamedTable'
    store.upsert_entry(catalog, entry_dir, record, sample_spatial_spec)
    assert catalog['entries']['s1']['class_name'] == 'RenamedTable'


def test_shared_dimensions_deduplicate(
    catalog: dict,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    """Entries sharing a spec share dimension rows; a different spec adds only what changed."""
    d1 = write_entry_dir(tmp_path / 'data_processed', 's1')
    d2 = write_entry_dir(tmp_path / 'data_processed', 's2')
    store.upsert_entry(catalog, d1, make_record('s1'), sample_spatial_spec)

    other_spec = SpatialSpec(
        crs=CRS.from_string('EPSG:3857'),
        transform=sample_spatial_spec.transform,
        shape=sample_spatial_spec.shape,
    )
    store.upsert_entry(catalog, d2, make_record('s2', 'SimpleTable'), other_spec)

    assert len(catalog['entries']) == 2
    dims = catalog['dimensions']
    assert len(dims['crs']) == 2
    assert len(dims['transforms']) == 1
    assert len(dims['bounds']) == 1
    assert len(dims['shapes']) == 1


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


def test_partial_spec_leaves_nullable_keys(catalog: dict, tmp_path: Path) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(catalog, entry_dir, make_record(), SpatialSpec(crs=None))

    entry = catalog['entries']['s1']
    assert entry['crs_hash'] is None
    assert entry['transform_hash'] is None
    assert entry['bounds_hash'] is None
    assert entry['shape_hash'] is None


def test_spec_info_keeps_crs_of_partial_spec(
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    catalog = store._empty_catalog()
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(catalog, entry_dir, make_record(), SpatialSpec(crs=CRS.from_string('ESRI:54012')))
    store.save_catalog(catalog)

    entry = store.load_catalog()['entries']['s1']
    info = _spec_info_cached.__wrapped__(entry['crs_hash'], entry['transform_hash'], entry['shape_hash'])

    assert info.crs == 'ESRI:54012'
    assert info.resolution is None
    assert info.shape is None


def test_format_version_mismatch_wipes(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    catalog = store._empty_catalog()
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(catalog, entry_dir, make_record(), sample_spatial_spec)
    store.save_catalog(catalog)
    assert store.load_catalog()['entries']

    on_disk = json.loads(get_config().path_catalog.read_text(encoding='utf-8'))
    on_disk['format_version'] = FORMAT_VERSION + 1
    get_config().path_catalog.write_text(json.dumps(on_disk), encoding='utf-8')

    assert store.load_catalog()['entries'] == {}


def test_prune_missing(catalog: dict, tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    keep = write_entry_dir(tmp_path / 'data_processed', 's1')
    gone = write_entry_dir(tmp_path / 'data_processed', 's2')
    store.upsert_entry(catalog, keep, make_record('s1'), sample_spatial_spec)
    store.upsert_entry(catalog, gone, make_record('s2'), sample_spatial_spec)

    import shutil

    shutil.rmtree(gone)

    assert store.prune_missing(catalog) == 1
    assert 's1' in catalog['entries']
    assert 's2' not in catalog['entries']


def test_backfill_indexes_and_skips_old_format(
    catalog: dict,
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

    indexed = store.backfill(catalog, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION)
    assert indexed == 2
    assert 's_old' not in catalog['entries']
    assert 's1' in catalog['entries']


def test_bootstrap_rows_latest_per_class(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    catalog = store._empty_catalog()
    for state_hash, class_name, mtime in [('a1', 'A', 100.0), ('a2', 'A', 200.0), ('b1', 'B', 150.0)]:
        entry_dir = write_entry_dir(tmp_path / 'data_processed', state_hash, class_name)
        os.utime(entry_dir / 'meta.json', (mtime, mtime))
        store.upsert_entry(catalog, entry_dir, make_record(state_hash, class_name), sample_spatial_spec)

    rows = store.bootstrap_rows(catalog)
    assert [r['state_hash'] for r in rows] == ['a2', 'b1']


def test_write_entry_hooks_catalog(sample_spatial_spec: SpatialSpec) -> None:
    artifact = SimpleLoader()
    artifact.write_entry(sample_spatial_spec)

    catalog = store.load_catalog()
    state_hash = artifact.get_state_hash(sample_spatial_spec)
    entry = catalog['entries'][state_hash]
    assert entry['class_name'] == 'SimpleLoader'
    assert entry['spec_hash'] == sample_spatial_spec.get_hash()
    assert entry['dir_relpath'] == state_hash
    dims = catalog['dimensions']
    assert dims['crs'][entry['crs_hash']] == sample_spatial_spec.crs.to_string()
    transform = dims['transforms'][entry['transform_hash']]
    assert (transform['res_x'], transform['res_y']) == sample_spatial_spec.resolution
    bounds = dims['bounds'][entry['bounds_hash']]
    spec_bounds = sample_spatial_spec.bounds
    assert (bounds['left'], bounds['bottom'], bounds['right'], bounds['top']) == (
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
    def broken_save(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(store, 'save_catalog', broken_save)
    SimpleLoader().write_entry(sample_spatial_spec)
    assert any('Catalog write failed' in str(w.message) for w in recwarn.list)


def test_catalog_disabled_makes_writes_noops(
    sample_spatial_spec: SpatialSpec,
    recwarn: pytest.WarningsChecker,
) -> None:
    with set_config(catalog_enabled=False):
        SimpleLoader().write_entry(sample_spatial_spec)
        assert store.reconcile() == 0
        assert store.refresh([get_config().path_cache]) == (0, 0)
    assert not get_config().path_catalog.exists()
    assert not [w for w in recwarn.list if 'Catalog' in str(w.message)]


def test_path_catalog_moves_the_catalog(tmp_path: Path) -> None:
    custom = tmp_path / 'elsewhere' / 'catalog.json'

    with set_config(path_catalog=custom):
        store.save_catalog(store.load_catalog())

        assert custom.exists()
        assert not (tmp_path / '.catalog.json').exists()

    assert get_config().path_catalog == tmp_path / '.catalog.json'
    assert Config().path_catalog == Path('.catalog.json')


def test_entry_abs_path_resolves_against_configured_roots(tmp_path: Path) -> None:
    cache_root = tmp_path / 'data_processed'
    entry_dir = write_entry_dir(cache_root, 's1')
    assert store.entry_abs_path('s1') == entry_dir.resolve()

    with set_config(path_cache=tmp_path / 'moved'):
        assert store.entry_abs_path('s1', cache_roots=(cache_root,)) == entry_dir.resolve()

    absolute = str(entry_dir.resolve())
    assert store.entry_abs_path(absolute) == entry_dir.resolve()


def test_backfill_populates_display_and_files(
    catalog: dict,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    entry_dir = write_entry_dir(tmp_path / 'data_processed', 's1', spec_dict=sample_spatial_spec.to_dict())
    (entry_dir / 'parameters.json').write_text(json.dumps({'a': {'b': 1}}), encoding='utf-8')
    (entry_dir / 'out.html').write_text('<html/>', encoding='utf-8')
    (entry_dir / 'sidecar.csv').write_text('x', encoding='utf-8')

    store.backfill(catalog, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION)

    entry = catalog['entries']['s1']
    assert entry['rows'] is not None
    assert entry['rows'][0]['path'] == 'a.b'
    assert entry['rows'][0]['value_text'] == '1'

    file_rows = sorted(catalog['files']['s1'], key=lambda f: f['relpath'])
    assert [(f['label'], f['kind'], f['role']) for f in file_rows] == [
        ('out.html', 'table', FileRole.PRIMARY.value),
        ('sidecar.csv', 'file', FileRole.SIDECAR.value),
    ]


def test_write_hook_stores_display_and_files(sample_spatial_spec: SpatialSpec) -> None:
    artifact = SimpleLoader()
    entry_dir = artifact.resolve_cache_paths(sample_spatial_spec).directory
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / 'dummy.tif').write_bytes(b'x')
    artifact.write_entry(sample_spatial_spec)

    catalog = store.load_catalog()
    state_hash = artifact.get_state_hash(sample_spatial_spec)
    assert catalog['entries'][state_hash]['rows'] is not None
    assert len(catalog['files'].get(state_hash, [])) >= 1


def test_schema_fingerprint_mismatch_wipes(tmp_path: Path) -> None:
    catalog = store._empty_catalog()
    write_entry_dir(tmp_path / 'data_processed', 's1')
    store.upsert_entry(catalog, tmp_path / 'data_processed' / 's1', make_record('s1'), None)
    store.save_catalog(catalog)
    assert store.load_catalog()['entries']

    on_disk = json.loads(get_config().path_catalog.read_text(encoding='utf-8'))
    on_disk['schema_fingerprint'] = 'stale'
    get_config().path_catalog.write_text(json.dumps(on_disk), encoding='utf-8')

    wiped = store.load_catalog()
    assert wiped['schema_fingerprint'] == store.schema_fingerprint()
    assert wiped['entries'] == {}


def test_backfill_is_incremental(
    catalog: dict,
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry_dir(tmp_path / 'data_processed', 's1', spec_dict=spec_dict)
    assert store.backfill(catalog, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION) == 1

    write_entry_dir(tmp_path / 'data_processed', 's2', 'OtherTable', spec_dict=spec_dict)
    assert store.backfill(catalog, [tmp_path / 'data_processed'], format_version=FORMAT_VERSION) == 1
    assert 's1' in catalog['entries']
    assert 's2' in catalog['entries']


def test_save_catalog_follows_symlinks(tmp_path: Path) -> None:
    """A symlinked catalog (e.g. into a synced tree) stays a symlink; writes land on its target."""
    target = tmp_path / 'synced' / '.catalog.json'
    target.parent.mkdir()
    link = tmp_path / '.catalog.json'
    link.symlink_to(target)

    with set_config(path_catalog=link):
        store.save_catalog(store.load_catalog())

    assert link.is_symlink()
    assert target.is_file()
    assert store.load_catalog() is not None


def test_entry_display_relpath_survives_symlinked_entry_dir(tmp_path: Path) -> None:
    """File refs must relativize even when entry dirs are reached through a symlinked
    home (e.g. eos_user_home → /mnt), which resolves differently than the raw path.
    """
    import os

    from pygeodata.catalog.display import entry_display

    real = tmp_path / 'real' / 'data_processed'
    real.mkdir(parents=True)
    entry_dir = real / 's1'
    entry_dir.mkdir()
    (entry_dir / 'out.tif').write_bytes(b'x')

    alias_root = tmp_path / 'alias' / 'data_processed'
    alias_root.parent.mkdir()
    os.symlink(real, alias_root)

    display = entry_display({'a': 1}, alias_root / 's1', 'MyLoader', 'Data')
    assert display.primary is not None
    assert display.primary['relpath'] == 'out.tif'


def test_pins_keep_order_and_are_idempotent() -> None:
    store.set_pinned('b', pinned=True)
    store.set_pinned('a', pinned=True)
    store.set_pinned('b', pinned=True)
    assert store.pinned_hashes() == ['b', 'a']

    store.set_pinned('b', pinned=False)
    store.set_pinned('missing', pinned=False)
    assert store.pinned_hashes() == ['a']
    assert store.pins_path().name == '.catalog.pins.json'
