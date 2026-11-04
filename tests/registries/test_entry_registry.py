"""Tests for EntryRegistry and the EntryInfo payload.

Covers: scan→records, catalog hydration, entry dirs, hash-collision behaviour,
and the EntryInfo.to_dict payload (including the co_outputs no-recursion guarantee).
"""

import json
from pathlib import Path

import pytest

from pygeodata._version import FORMAT_VERSION
from pygeodata.catalog import store
from pygeodata.catalog.types import FileRef, FileRole
from pygeodata.config import get_config, set_config
from pygeodata.enums import JSONKeys
from pygeodata.registries.registry import EntryRegistry
from pygeodata.tracked_object import TrackedObject
from pygeodata.views.types import EntryInfo, SpecInfo
from tests.fixtures.records import make_record, write_meta

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path):
    with set_config(
        path_cache=tmp_path / 'data_processed',
        path_figures=tmp_path / 'figures',
        path_registry=tmp_path / '.source',
    ):
        EntryRegistry._instance = None
        yield tmp_path
        EntryRegistry._instance = None


@pytest.fixture(autouse=True)
def restore_registry():
    saved = dict(TrackedObject._registry)
    yield
    TrackedObject._registry = saved
    TrackedObject.clear_function_caches()


def write_hash_file(
    directory: Path,
    stem: str,
    state: dict | None = None,
    params: dict | None = None,
    spec: dict | None = None,
    output_ext: str | None = 'tif',
) -> Path:
    """Write cache files for one entry. Returns the hash file path."""
    from pygeodata.paths import CachePathConstructor

    directory.mkdir(parents=True, exist_ok=True)
    resolver = CachePathConstructor(directory)
    state = {str(k): v for k, v in (state or {}).items()}
    if JSONKeys.STATE_HASH in state:
        write_meta(directory, make_record(**{'object_type': 'Data', **state}))
    else:
        resolver.state_hash_path.write_text(json.dumps(state))  # an incomplete, unreadable record
    resolver.params_path.write_text(json.dumps(params or {}))
    resolver.spec_path.write_text(json.dumps(spec or {}))
    if output_ext:
        (directory / f'{stem}.{output_ext}').write_bytes(b'data')
    return resolver.state_hash_path


# ---------------------------------------------------------------------------
# EntryInfo.to_dict
# ---------------------------------------------------------------------------


def _make_entry_info(**kwargs) -> EntryInfo:
    defaults = dict(
        class_name='MyLoader',
        object_type='data',
        entry_dir='/cache',
        params_path='/cache/file',
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash='abc',
        instance_hash=None,
        params_hash='params_hash',
        spec_hash='spec_hash',
        params={'year': 2020},
        spec=SpecInfo(crs='EPSG:4326', resolution='0.1°'),
        rows=[],
        record_id='abc',
        format_version=FORMAT_VERSION,
    )
    defaults.update(kwargs)
    return EntryInfo(**defaults)


def test_entry_info_to_dict_renders_file_paths():
    entry = _make_entry_info(
        primary_file=FileRef(relpath='out.tif', kind='cube', role=FileRole.PRIMARY),
        sidecar_files=[FileRef(relpath='sub/side.csv', kind='file', role=FileRole.SIDECAR)],
    )
    d = entry.to_dict()
    assert d['primary_file'] == {'label': 'out.tif', 'path': '/cache/out.tif', 'kind': 'cube'}
    assert d['sidecar_files'] == [{'label': 'side.csv', 'path': '/cache/sub/side.csv', 'kind': 'file'}]
    assert _make_entry_info(primary_file=None).to_dict()['primary_file'] is None


def test_entry_info_to_dict_excludes_co_outputs_terminates():
    """to_dict on an entry with resolved co_outputs must terminate and emit only hashes."""
    child = _make_entry_info(state_hash='child', record_id='child', co_output_hashes=[])
    parent = _make_entry_info(
        state_hash='parent',
        record_id='parent',
        co_output_hashes=['child'],
        co_outputs=[child],
    )
    d = parent.to_dict()  # must not recurse infinitely
    assert 'co_output_hashes' in d
    assert d['co_output_hashes'] == ['child']
    assert 'co_outputs' not in d  # back-references excluded


def test_entry_info_format_version_stale_property():
    entry = _make_entry_info(format_version=FORMAT_VERSION)
    assert not entry.format_version_stale
    stale = _make_entry_info(format_version=FORMAT_VERSION - 1)
    assert stale.format_version_stale


# ---------------------------------------------------------------------------
# EntryRegistry scan + assembly
# ---------------------------------------------------------------------------


def test_entry_registry_empty(tmp_path):
    reg = EntryRegistry()
    assert reg.records == {}
    assert reg.class_names == []
    assert reg.diagnostics()['created_records'] == 0


def test_entry_registry_single_entry(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(
        d,
        'abc',
        state={
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.STATE_HASH: 'abc123',
        },
    )
    reg = EntryRegistry()
    assert 'abc123' in reg.records
    assert reg.records['abc123'].class_name == 'MyLoader'


def test_entry_registry_groups_populated(tmp_path):
    base = tmp_path / 'data_processed'
    write_hash_file(base / 'abc', 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})
    write_hash_file(base / 'def', 'def', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'def'})
    reg = EntryRegistry()
    assert 'MyLoader' in reg.class_names
    assert len(reg.get_state_hashes('MyLoader')) == 2


def test_entry_registry_missing_state_hash(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader'})  # no STATE_HASH
    reg = EntryRegistry()
    assert reg.diagnostics()['unreadable_records'] == 1


def test_entry_registry_collision_identical_silently_deduplicates(tmp_path):
    """Same state_hash at two paths with identical identity fields → keep one."""
    d1 = tmp_path / 'data_processed' / 'A'
    d2 = tmp_path / 'data_processed' / 'B'
    state = {
        JSONKeys.CLASS_NAME: 'MyLoader',
        JSONKeys.OBJECT_TYPE: 'data',
        JSONKeys.STATE_HASH: 'same_hash',
        JSONKeys.INSTANCE_HASH: 'inst',
        JSONKeys.DEPENDENCY_TREE_HASH: 'dep',
        JSONKeys.CO_OUTPUTS: [],
        JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
    }
    write_hash_file(d1, 'abc', state=state)
    write_hash_file(d2, 'abc', state=state)
    reg = EntryRegistry()
    assert 'same_hash' in reg.records
    assert len(reg.records) == 1


def test_entry_registry_collision_divergent_raises(tmp_path):
    """Same state_hash but different class_name → ValueError."""
    d1 = tmp_path / 'data_processed' / 'A'
    d2 = tmp_path / 'data_processed' / 'B'
    write_hash_file(d1, 'abc', state={JSONKeys.CLASS_NAME: 'Foo', JSONKeys.STATE_HASH: 'clash'})
    write_hash_file(d2, 'xyz', state={JSONKeys.CLASS_NAME: 'Bar', JSONKeys.STATE_HASH: 'clash'})
    with pytest.raises(ValueError, match='divergent'):
        EntryRegistry()


# ---------------------------------------------------------------------------
# Catalog hydration and disk fallback
# ---------------------------------------------------------------------------


def test_entry_registry_hydrates_from_catalog(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})
    (tmp_path / '.source').mkdir(parents=True, exist_ok=True)

    store.refresh(get_config().cache_roots, FORMAT_VERSION)

    reg = EntryRegistry()
    assert 'abc' in reg.records
    assert reg.entry_dir('abc') == d.absolute()
    assert get_config().path_catalog.exists()


def test_entry_registry_disk_scan_without_catalog(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    (tmp_path / '.source').mkdir(parents=True, exist_ok=True)
    hash_path = write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})

    reg = EntryRegistry()
    assert 'abc' in reg.records
    assert reg.entry_dir('abc') == d

    write_meta(hash_path.parent, make_record('abc_new', 'MyLoader', 'Data'))

    reg2 = EntryRegistry()
    assert 'abc_new' in reg2.records
    assert 'abc' not in reg2.records
