"""Tests for EntryRegistry and the catalog read path it opens (store.open_catalog)."""

import json
from pathlib import Path

import pytest

from pygeodata._version import CATALOG_VERSION, FORMAT_VERSION
from pygeodata.catalog import store
from pygeodata.catalog.types import CatalogData, CatalogOutdatedError
from pygeodata.config import get_config, set_config
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.registry import EntryRegistry
from pygeodata.spec import SpatialSpec
from tests.fixtures.records import make_record, write_meta


def write_entry(state_hash: str, class_name: str = 'MyLoader', spec_hash: str = 'spec1') -> Path:
    entry_dir = get_config().path_cache / state_hash
    write_meta(entry_dir, make_record(state_hash, class_name, 'Data', spec_hash=spec_hash))
    CachePathConstructor(entry_dir).params_path.write_text('{"year": 2020}', encoding='utf-8')
    (entry_dir / 'out.tif').write_bytes(b'data')
    return entry_dir


def write_foreign_catalog(**header: object) -> bytes:
    """A catalog as synced from another machine: written from a different project root."""
    catalog = store._new_catalog()
    catalog.header.writer.root = '/eos/elsewhere'
    for key, value in header.items():
        setattr(catalog.header, key, value)
    store.backfill(catalog, [get_config().path_cache / 'a1'])
    catalog.dump(get_config().path_catalog)
    return get_config().path_catalog.read_bytes()


def test_empty() -> None:
    reg = EntryRegistry()
    assert reg.entries == {}
    assert reg.class_names == []
    assert reg.diagnostics()['entries'] == 0


def test_indexes_entries_and_writes_the_catalog() -> None:
    write_entry('a1')
    write_entry('a2')

    reg = EntryRegistry()

    assert reg.get_state_hashes('MyLoader') == ['a1', 'a2']
    assert reg.get_object_type('MyLoader') == 'Data'
    assert reg.entry_dir('a1') == (get_config().path_cache / 'a1').absolute()
    assert reg.get_entry('a1').primary_file.relpath == 'out.tif'
    assert reg.get_entry('a1').state_hash == 'a1'
    assert set(CatalogData.from_file(get_config().path_catalog).entries) == {'a1', 'a2'}


def test_prefix_lookup() -> None:
    write_entry('abc1')
    write_entry('abd2')
    reg = EntryRegistry()

    assert sorted(reg.match_prefix('ab')) == ['abc1', 'abd2']
    assert reg.resolve_hash_prefix('abc') == 'abc1'
    assert reg.resolve_hash_prefix('ab') is None


def test_built_from_data_reads_and_writes_nothing() -> None:
    write_entry('a1')
    catalog = store._new_catalog()

    reg = EntryRegistry(catalog)

    assert reg.entries == {}
    assert reg.diagnostics()['read_only'] is None
    assert not get_config().path_catalog.exists()


def test_writable_catalog_follows_the_entry_dirs() -> None:
    write_entry('a1')
    gone = write_entry('a2')
    EntryRegistry()

    write_entry('a3')
    for f in gone.iterdir():
        f.unlink()
    gone.rmdir()

    reg = EntryRegistry()
    assert set(reg.entries) == {'a1', 'a3'}
    assert set(CatalogData.from_file(get_config().path_catalog).entries) == {'a1', 'a3'}


def test_up_to_date_catalog_is_not_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    write_entry('a1')
    EntryRegistry()

    saves: list[CatalogData] = []
    monkeypatch.setattr(store, '_save', saves.append)
    EntryRegistry()
    assert not saves


def test_unreadable_dir_is_reported_without_rewriting(monkeypatch: pytest.MonkeyPatch) -> None:
    write_entry('a1')
    broken = get_config().path_cache / 'broken'
    broken.mkdir()
    CachePathConstructor(broken).state_hash_path.write_text(json.dumps({'class_name': 'X'}))
    EntryRegistry()

    saves: list[CatalogData] = []
    monkeypatch.setattr(store, '_save', saves.append)
    reg = EntryRegistry()

    assert set(reg.entries) == {'a1'}
    assert reg.diagnostics()['unindexed_dirs'] == 1
    assert not saves


def write_skipped_dirs() -> None:
    cache = get_config().path_cache
    write_meta(cache / 'old1', make_record('old1', 'MyLoader', 'Data', format_version=FORMAT_VERSION - 1))
    (cache / 'interrupted1').mkdir()
    (cache / 'interrupted2').mkdir()
    (cache / 'broken').mkdir()
    CachePathConstructor(cache / 'broken').state_hash_path.write_text('{not json')


@pytest.mark.parametrize('sync_needed', [False, True])
def test_writable_diagnostics_count_why_dirs_are_unindexed(sync_needed: bool) -> None:
    write_entry('a1')
    write_skipped_dirs()
    if not sync_needed:
        EntryRegistry()
    else:
        write_entry('a2')

    diagnostics = EntryRegistry().diagnostics()

    assert diagnostics['unindexed_dirs'] == 4
    assert diagnostics['unindexed_reasons'] == {'other_format': 1, 'no_meta': 2, 'unreadable': 1}


def test_mirror_diagnostics_do_not_know_why_dirs_are_unindexed() -> None:
    write_entry('a1')
    write_foreign_catalog()
    write_skipped_dirs()

    diagnostics = EntryRegistry().diagnostics()

    assert diagnostics['unindexed_dirs'] == 4
    assert diagnostics['unindexed_reasons'] is None


def test_mirror_is_read_only_and_reports_the_difference() -> None:
    write_entry('a1')
    before = write_foreign_catalog()
    write_entry('a2')

    reg = EntryRegistry()

    assert set(reg.entries) == {'a1'}
    assert reg.diagnostics()['read_only'] is True
    assert reg.diagnostics()['unindexed_dirs'] == 1
    assert get_config().path_catalog.read_bytes() == before


def test_outdated_mirror_catalog_raises_instead_of_scanning() -> None:
    write_entry('a1')
    before = write_foreign_catalog(catalog_version=CATALOG_VERSION - 1)

    with pytest.raises(CatalogOutdatedError, match='catalog build'):
        EntryRegistry()
    assert get_config().path_catalog.read_bytes() == before


def test_pre_header_catalog_is_never_rebuilt_by_a_reader() -> None:
    """Such a catalog does not say where it was written, so it may be a mirror."""
    write_entry('a1')
    get_config().path_catalog.write_text(json.dumps({'format_version': 2, 'entries': {}}), encoding='utf-8')

    with pytest.raises(CatalogOutdatedError, match='catalog build'):
        EntryRegistry()


def test_outdated_local_catalog_is_rebuilt() -> None:
    write_entry('a1')
    catalog = store._new_catalog()
    catalog.header.catalog_version = 1
    catalog.dump(get_config().path_catalog)

    reg = EntryRegistry()

    assert set(reg.entries) == {'a1'}
    assert CatalogData.from_file(get_config().path_catalog).header.catalog_id == catalog.header.catalog_id


def test_catalog_disabled_indexes_in_memory() -> None:
    write_entry('a1')
    with set_config(catalog_enabled=False):
        reg = EntryRegistry()
    assert set(reg.entries) == {'a1'}
    assert reg.diagnostics()['read_only'] is True
    assert not get_config().path_catalog.exists()


def test_explicit_read_only_never_writes() -> None:
    write_entry('a1')
    EntryRegistry()
    before = get_config().path_catalog.read_bytes()
    write_entry('a2')

    reg = EntryRegistry(read_only=True)

    assert set(reg.entries) == {'a1'}
    assert get_config().path_catalog.read_bytes() == before


def test_get_spec_resolves_through_the_entry(sample_spatial_spec: SpatialSpec) -> None:
    entry_dir = write_entry('a1', spec_hash=sample_spatial_spec.get_hash())
    CachePathConstructor(entry_dir).spec_path.write_text(json.dumps(sample_spatial_spec.to_dict()))

    reg = EntryRegistry()

    assert reg.get_spec('a1') == sample_spatial_spec.to_record()
    assert reg.get_spec('missing') is None


def test_unavailable_root_keeps_its_entries() -> None:
    write_entry('a1')
    EntryRegistry()
    for f in (get_config().path_cache / 'a1').iterdir():
        f.unlink()
    (get_config().path_cache / 'a1').rmdir()
    get_config().path_cache.rmdir()

    reg = EntryRegistry()

    assert set(reg.entries) == {'a1'}
    assert reg.diagnostics()['unavailable_roots'] == [str(get_config().path_cache)]
