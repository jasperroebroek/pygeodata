import json
import os
import shutil
from pathlib import Path

import msgspec
import pytest
from pyproj import CRS

from pygeodata._version import CATALOG_VERSION, FORMAT_VERSION
from pygeodata.cache_entry import write_entry
from pygeodata.catalog import store
from pygeodata.catalog.display import find_output_files
from pygeodata.catalog.types import CatalogData, CatalogNewerError, FileRef, FileRole, schema_fingerprint
from pygeodata.config import Config, get_config, set_config
from pygeodata.paths import CachePathResolver
from pygeodata.records import EntryRecord
from pygeodata.spec import SpatialSpec
from pygeodata.views.types import spec_info
from tests.fixtures.data import SimpleLoader
from tests.fixtures.records import make_record, write_meta


def record(state_hash: str = 's1', class_name: str = 'SimpleTable', **fields: object) -> EntryRecord:
    return make_record(state_hash, class_name, 'Data', **fields)


def write_entry_dir(state_hash: str, class_name: str = 'SimpleTable', spec: SpatialSpec | None = None) -> Path:
    entry_dir = get_config().path_cache / state_hash
    write_meta(entry_dir, record(state_hash, class_name))
    if spec is not None:
        (entry_dir / 'spec.json').write_text(json.dumps(spec.to_dict()), encoding='utf-8')
    return entry_dir


def index(catalog: CatalogData, entry_dir: Path, rec: EntryRecord, spec: SpatialSpec | None) -> None:
    catalog.upsert(store.catalog_entry(entry_dir, rec, None), spec.to_record() if spec is not None else None)


def all_entry_dirs() -> list[Path]:
    return list(CachePathResolver(get_config().cache_roots).iter_entry_dirs())


def write_catalog(catalog: CatalogData) -> None:
    catalog.dump(get_config().path_catalog)


def on_disk() -> dict:
    return json.loads(get_config().path_catalog.read_text(encoding='utf-8'))


@pytest.fixture
def catalog() -> CatalogData:
    return store._new_catalog()


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_round_trip(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    index(catalog, write_entry_dir('s1'), record(co_output_hashes=['co1']), sample_spatial_spec)
    write_catalog(catalog)
    assert CatalogData.from_file(get_config().path_catalog) == catalog


def test_entries_sharing_a_spec_share_its_record(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    other_spec = SpatialSpec(
        crs=CRS.from_string('EPSG:3857'),
        transform=sample_spatial_spec.transform,
        shape=sample_spatial_spec.shape,
    )
    index(catalog, write_entry_dir('s1'), record('s1'), sample_spatial_spec)
    index(catalog, write_entry_dir('s2'), record('s2'), sample_spatial_spec)
    index(catalog, write_entry_dir('s3'), record('s3', spec_hash='spec2'), other_spec)

    assert len(catalog.entries) == 3
    assert catalog.specs == {'spec1': sample_spatial_spec.to_record(), 'spec2': other_spec.to_record()}


def test_upsert_replaces_the_entry(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    entry_dir = write_entry_dir('s1')
    index(catalog, entry_dir, record(), sample_spatial_spec)
    index(catalog, entry_dir, record(class_name='RenamedTable'), sample_spatial_spec)
    assert catalog.entries['s1'].record.class_name == 'RenamedTable'


def test_prune_drops_entries_and_unused_specs(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    index(catalog, write_entry_dir('s1'), record('s1'), sample_spatial_spec)
    gone = write_entry_dir('s2')
    index(catalog, gone, record('s2', spec_hash='spec2'), SpatialSpec(crs=None))
    shutil.rmtree(gone)

    assert store.prune_missing(catalog) == store.PruneResult(pruned=1)
    assert list(catalog.entries) == ['s1']
    assert list(catalog.specs) == ['spec1']


def test_spec_info_keeps_crs_of_partial_spec() -> None:
    info = spec_info(SpatialSpec(crs=CRS.from_string('ESRI:54012')).to_record())
    assert info.crs == 'ESRI:54012'
    assert info.resolution is None
    assert info.shape is None


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------


def test_reader_rebuilds_local_outdated_catalog(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    index(catalog, write_entry_dir('s1'), record(), sample_spatial_spec)
    catalog.header.catalog_version = CATALOG_VERSION - 1
    write_catalog(catalog)

    opened, status = store.open_catalog()

    assert list(opened.entries) == ['s1']
    assert not status.read_only
    assert on_disk()['header']['catalog_version'] == CATALOG_VERSION


def test_reader_rebuilds_local_catalog_with_other_schema(catalog: CatalogData) -> None:
    catalog.header.schema_fingerprint = 'other'
    write_catalog(catalog)
    store.open_catalog()
    assert on_disk()['header']['schema_fingerprint'] == schema_fingerprint()


def test_writer_rebuilds_outdated_catalog_and_keeps_its_id(
    catalog: CatalogData,
    sample_spatial_spec: SpatialSpec,
) -> None:
    write_entry_dir('s1', spec=sample_spatial_spec)
    write_entry_dir('s2', spec=sample_spatial_spec)
    catalog.header.catalog_version = CATALOG_VERSION - 1
    write_catalog(catalog)

    store.remove_entries(['s3'])

    rebuilt = CatalogData.from_file(get_config().path_catalog)
    assert rebuilt.header.catalog_id == catalog.header.catalog_id
    assert set(rebuilt.entries) == {'s1', 's2'}


def test_writer_rebuilds_pre_contract_catalog(sample_spatial_spec: SpatialSpec) -> None:
    write_entry_dir('s1', spec=sample_spatial_spec)
    get_config().path_catalog.write_text(json.dumps({'format_version': 2, 'entries': {}}), encoding='utf-8')

    assert store.refresh(get_config().cache_roots).indexed == 1
    assert CatalogData.from_file(get_config().path_catalog).header.catalog_version == CATALOG_VERSION


def test_newer_catalog_is_never_read_or_overwritten(
    catalog: CatalogData,
    sample_spatial_spec: SpatialSpec,
    recwarn: pytest.WarningsRecorder,
) -> None:
    catalog.header.catalog_version = CATALOG_VERSION + 1
    catalog.header.writer.pygeodata = '9.9.9'
    write_catalog(catalog)
    before = get_config().path_catalog.read_bytes()

    with pytest.raises(CatalogNewerError, match='9.9.9'):
        store.open_catalog()
    with pytest.raises(CatalogNewerError):
        store.refresh(get_config().cache_roots)
    write_entry(SimpleLoader(), sample_spatial_spec)

    assert any('Catalog write failed' in str(w.message) for w in recwarn.list)
    assert get_config().path_catalog.read_bytes() == before


def test_catalog_id_survives_catalog_build(sample_spatial_spec: SpatialSpec) -> None:
    write_entry_dir('s1', spec=sample_spatial_spec)
    store.refresh(get_config().cache_roots)
    catalog_id = on_disk()['header']['catalog_id']

    store.refresh(get_config().cache_roots)
    assert on_disk()['header']['catalog_id'] == catalog_id


def test_header_records_writer_and_roots(sample_spatial_spec: SpatialSpec) -> None:
    write_entry(SimpleLoader(), sample_spatial_spec)
    header = CatalogData.from_file(get_config().path_catalog).header
    assert header.writer.root == str(get_config().path_catalog.parent.resolve())
    assert header.roots['Data'] == str(get_config().path_cache.resolve())


# ---------------------------------------------------------------------------
# Entry dirs
# ---------------------------------------------------------------------------


def test_entry_dir_resolves_against_local_root(catalog: CatalogData, tmp_path: Path) -> None:
    index(catalog, write_entry_dir('s1'), record(), None)
    entry = catalog.entries['s1']
    assert entry.dir_relpath == 's1'

    with set_config(path_cache=tmp_path / 'moved'):
        assert store.entry_dir(entry) == (tmp_path / 'moved' / 's1').absolute()


def test_entry_outside_its_root_keeps_absolute_path(catalog: CatalogData, tmp_path: Path) -> None:
    entry_dir = tmp_path / 'elsewhere' / 's1'
    write_meta(entry_dir, record())
    index(catalog, entry_dir, record(), None)
    entry = catalog.entries['s1']
    assert entry.dir_relpath == str(entry_dir.resolve())
    assert store.entry_dir(entry) == entry_dir.resolve()


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def test_backfill_indexes_and_skips_old_format(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    write_entry_dir('s1', spec=sample_spatial_spec)
    write_entry_dir('s2', 'OtherTable', spec=sample_spatial_spec)
    write_meta(get_config().path_cache / 's_old', record('s_old', format_version=FORMAT_VERSION - 1))

    assert store.backfill(catalog, all_entry_dirs(), format_version=FORMAT_VERSION) == 2
    assert set(catalog.entries) == {'s1', 's2'}
    assert catalog.specs == {'spec1': sample_spatial_spec.to_record()}


def test_backfill_is_incremental(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    write_entry_dir('s1', spec=sample_spatial_spec)
    assert store.backfill(catalog, all_entry_dirs(), format_version=FORMAT_VERSION) == 1

    write_entry_dir('s2', 'OtherTable', spec=sample_spatial_spec)
    assert store.backfill(catalog, all_entry_dirs(), format_version=FORMAT_VERSION) == 1
    assert set(catalog.entries) == {'s1', 's2'}


def test_backfill_stores_rows_and_files(catalog: CatalogData, sample_spatial_spec: SpatialSpec) -> None:
    entry_dir = write_entry_dir('s1', spec=sample_spatial_spec)
    (entry_dir / 'parameters.json').write_text(json.dumps({'a': {'b': 1}}), encoding='utf-8')
    (entry_dir / 'out.html').write_text('<html/>', encoding='utf-8')
    (entry_dir / 'sidecar.csv').write_text('x', encoding='utf-8')

    store.backfill(catalog, all_entry_dirs(), format_version=FORMAT_VERSION)

    entry = catalog.entries['s1']
    assert (entry.rows[0].path, entry.rows[0].value_text) == ('a.b', '1')
    assert entry.files == [
        FileRef(relpath='out.html', kind='table', role=FileRole.PRIMARY),
        FileRef(relpath='sidecar.csv', kind='file', role=FileRole.SIDECAR),
    ]


def test_backfill_tolerates_unreadable_spec(catalog: CatalogData) -> None:
    entry_dir = write_entry_dir('s1')
    (entry_dir / 'spec.json').write_text('{not json', encoding='utf-8')
    assert store.backfill(catalog, all_entry_dirs()) == 1
    assert catalog.specs == {}


def test_bootstrap_rows_latest_per_class(catalog: CatalogData) -> None:
    for state_hash, class_name, mtime in [('a1', 'A', 100.0), ('a2', 'A', 200.0), ('b1', 'B', 150.0)]:
        entry_dir = write_entry_dir(state_hash, class_name)
        os.utime(entry_dir / 'meta.json', (mtime, mtime))
        index(catalog, entry_dir, record(state_hash, class_name), None)

    assert [e.record.state_hash for e in store.bootstrap_rows(catalog)] == ['a2', 'b1']


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


def test_write_entry_indexes_the_entry(sample_spatial_spec: SpatialSpec) -> None:
    artifact = SimpleLoader()
    entry_dir = artifact.resolve_cache_paths(sample_spatial_spec).directory
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / 'dummy.tif').write_bytes(b'x')
    write_entry(artifact, sample_spatial_spec)

    catalog = CatalogData.from_file(get_config().path_catalog)
    state_hash = artifact.get_state_hash(sample_spatial_spec)
    entry = catalog.entries[state_hash]
    assert entry.record.class_name == 'SimpleLoader'
    assert entry.record.ext == artifact.get_ext()
    assert entry.dir_relpath == state_hash
    assert entry.primary_file == FileRef(relpath='dummy.tif', kind='cube', role=FileRole.PRIMARY)
    assert catalog.specs[entry.record.spec_hash] == sample_spatial_spec.to_record()


def test_first_write_indexes_existing_entries(sample_spatial_spec: SpatialSpec) -> None:
    """A missing catalog is rebuilt in full, so readers never see a partial index."""
    write_entry_dir('existing', spec=sample_spatial_spec)
    write_entry(SimpleLoader(), sample_spatial_spec)
    assert 'existing' in CatalogData.from_file(get_config().path_catalog).entries


def test_write_hook_warns_and_continues(
    sample_spatial_spec: SpatialSpec,
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    def broken_save(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(store, '_save', broken_save)
    write_entry(SimpleLoader(), sample_spatial_spec)
    assert any('Catalog write failed' in str(w.message) for w in recwarn.list)


def test_catalog_disabled_makes_writes_noops(
    sample_spatial_spec: SpatialSpec,
    recwarn: pytest.WarningsRecorder,
) -> None:
    with set_config(catalog_enabled=False):
        write_entry(SimpleLoader(), sample_spatial_spec)
        assert store.remove_entries(['s1']) == 0
        assert store.refresh([get_config().path_cache]) == store.RefreshResult()
    assert not get_config().path_catalog.exists()
    assert not [w for w in recwarn.list if 'Catalog' in str(w.message)]


def test_path_catalog_moves_the_catalog(tmp_path: Path) -> None:
    custom = tmp_path / 'elsewhere' / 'catalog.json'

    with set_config(path_catalog=custom):
        store.refresh(get_config().cache_roots)

        assert custom.exists()
        assert not (tmp_path / '.catalog.json').exists()

    assert get_config().path_catalog == tmp_path / '.catalog.json'
    assert Config().path_catalog == Path('.catalog.json')


def test_save_follows_symlinks(tmp_path: Path) -> None:
    """A symlinked catalog (e.g. into a synced tree) stays a symlink; writes land on its target."""
    target = tmp_path / 'synced' / '.catalog.json'
    target.parent.mkdir()
    link = tmp_path / '.catalog.json'
    link.symlink_to(target)

    with set_config(path_catalog=link):
        store.refresh(get_config().cache_roots)
        assert store.open_catalog()[0] is not None

    assert link.is_symlink()
    assert target.is_file()


def test_file_refs_survive_symlinked_entry_dir(tmp_path: Path) -> None:
    """File refs must relativize even when entry dirs are reached through a symlinked
    home (e.g. eos_user_home → /mnt), which resolves differently than the raw path.
    """
    real = tmp_path / 'real' / 'data_processed'
    (real / 's1').mkdir(parents=True)
    (real / 's1' / 'out.tif').write_bytes(b'x')

    alias_root = tmp_path / 'alias' / 'data_processed'
    alias_root.parent.mkdir()
    os.symlink(real, alias_root)

    files = find_output_files(alias_root / 's1', 'tif')
    assert files == [FileRef(relpath='out.tif', kind='cube', role=FileRole.PRIMARY)]


def test_primary_file_is_the_recorded_ext(tmp_path: Path) -> None:
    (tmp_path / 'out.csv').write_bytes(b'x')
    (tmp_path / 'out.html').write_bytes(b'x')

    assert [f.relpath for f in find_output_files(tmp_path, 'html')] == ['out.html', 'out.csv']
    assert [f.relpath for f in find_output_files(tmp_path, None)] == ['out.csv', 'out.html']


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def test_missing_root_is_never_pruned(catalog: CatalogData) -> None:
    index(catalog, write_entry_dir('s1'), record('s1'), None)
    shutil.rmtree(get_config().path_cache)

    with pytest.warns(UserWarning, match='Cache root missing or empty'):
        result = store.prune_missing(catalog)

    assert result == store.PruneResult(pruned=0, unavailable_roots=[get_config().path_cache])
    assert list(catalog.entries) == ['s1']


def test_empty_root_is_never_pruned(catalog: CatalogData) -> None:
    gone = write_entry_dir('s1')
    index(catalog, gone, record('s1'), None)
    shutil.rmtree(gone)

    with pytest.warns(UserWarning, match='Cache root missing or empty'):
        assert store.prune_missing(catalog).pruned == 0
    assert list(catalog.entries) == ['s1']


def test_single_entry_catalog_is_pruned(catalog: CatalogData) -> None:
    gone = write_entry_dir('s1')
    index(catalog, gone, record('s1'), None)
    shutil.rmtree(gone)
    write_entry_dir('s2')

    assert store.prune_missing(catalog) == store.PruneResult(pruned=1)
    assert not catalog.entries


def test_remove_entries_removes_exactly_those(catalog: CatalogData) -> None:
    for state_hash in ('s1', 's2'):
        index(catalog, write_entry_dir(state_hash), record(state_hash), None)
    write_catalog(catalog)

    assert store.remove_entries(['s1', 'unknown']) == 1
    assert set(on_disk()['entries']) == {'s2'}


def test_refresh_saves_only_on_change(sample_spatial_spec: SpatialSpec, monkeypatch: pytest.MonkeyPatch) -> None:
    write_entry_dir('s1', spec=sample_spatial_spec)
    assert store.refresh(get_config().cache_roots).indexed == 1

    saves: list[CatalogData] = []
    monkeypatch.setattr(store, '_save', saves.append)
    assert store.refresh(get_config().cache_roots) == store.RefreshResult()
    assert not saves


def test_refresh_warns_on_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_save(*_args: object) -> None:
        raise OSError('read-only file system')

    write_entry_dir('s1')
    monkeypatch.setattr(store, '_save', broken_save)
    with pytest.warns(UserWarning, match='Catalog refresh failed'):
        assert store.refresh(get_config().cache_roots) == store.RefreshResult()


def test_symlinks_to_one_catalog_share_its_lock(tmp_path: Path) -> None:
    target = tmp_path / 'synced' / '.catalog.json'
    target.parent.mkdir()
    links = [tmp_path / 'a.json', tmp_path / 'b.json']
    for link in links:
        link.symlink_to(target)

    lock_files = {store._lock_for(link).lock_file for link in links}
    assert lock_files == {str(target.with_name('.catalog.json.lock'))}


# ---------------------------------------------------------------------------
# Pins
# ---------------------------------------------------------------------------


def test_pins_keep_order_and_are_idempotent() -> None:
    store.set_pinned('b', pinned=True)
    store.set_pinned('a', pinned=True)
    store.set_pinned('b', pinned=True)
    assert store.pinned_hashes() == ['b', 'a']

    store.set_pinned('b', pinned=False)
    store.set_pinned('missing', pinned=False)
    assert store.pinned_hashes() == ['a']
    assert store.pins_path().name == '.catalog.pins.json'


def test_unreadable_pins_start_empty() -> None:
    store.pins_path().write_text(msgspec.json.encode([{'state_hash': 'a'}]).decode(), encoding='utf-8')
    with pytest.warns(UserWarning, match='Pins file unreadable'):
        assert store.pinned_hashes() == []
