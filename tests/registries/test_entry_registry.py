"""Tests for EntryRegistry and EntryRecord serialisation.

Covers: scan→records, disk-cache hit/miss by mtime, co-output resolution,
hash-collision behaviour, EntryInfo.to_dict/from_dict round-trips (including
the co_outputs no-recursion guarantee), and old-format cache degradation.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from pygeodata.config import FORMAT_VERSION, JSONKeys, set_config
from pygeodata.registries.registry import EntryRegistry
from pygeodata.catalog.entry_catalog import discover_entries
from pygeodata.catalog.types import EntryInfo, FileRef, SpecInfo
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.tracked_object import TrackedObject

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
    resolver.state_hash_path.write_text(json.dumps(state or {}))
    resolver.params_path.write_text(json.dumps(params or {}))
    resolver.spec_path.write_text(json.dumps(spec or {}))
    if output_ext:
        (directory / f'{stem}.{output_ext}').write_bytes(b'data')
    return resolver.state_hash_path


# ---------------------------------------------------------------------------
# EntryRecord.from_file
# ---------------------------------------------------------------------------


def test_from_file_reads_identity(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(
        d,
        'abc',
        state={
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.OBJECT_TYPE: 'data',
            JSONKeys.STATE_HASH: 'abc123',
            JSONKeys.INSTANCE_HASH: 'inst1',
            JSONKeys.DEPENDENCY_TREE_HASH: 'dep1',
            JSONKeys.CO_OUTPUTS: ['x', 'y'],
            JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
        },
    )
    rec = EntryRecord.from_file(d / 'meta.json')
    assert rec.class_name == 'MyLoader'
    assert rec.object_type == 'data'
    assert rec.state_hash == 'abc123'
    assert rec.instance_hash == 'inst1'
    assert rec.dependency_tree_hash == 'dep1'
    assert rec.co_output_hashes == ['x', 'y']
    assert rec.format_version == FORMAT_VERSION
    assert rec.params_path is not None
    assert rec.params_path.name == 'parameters.json'


def test_from_file_missing_raises(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    d.mkdir(parents=True)
    import pytest

    with pytest.raises(OSError):
        EntryRecord.from_file(d / 'meta.json')


def test_from_file_params_path_derivation(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    hash_path = write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader'})
    rec = EntryRecord.from_file(hash_path)
    assert rec.params_path is not None
    assert rec.params_path.name == 'parameters.json'


# ---------------------------------------------------------------------------
# EntryRecord.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def test_entry_record_round_trip(tmp_path):
    hash_path = tmp_path / 'meta.json'
    hash_path.write_text('{}')
    rec = EntryRecord(
        class_name='MyLoader',
        hash_path=str(hash_path),
        state_hash='abc123',
        instance_hash='inst1',
        dependency_tree_hash='dep1',
        co_output_hashes=['x'],
        object_type='data',
        format_version=FORMAT_VERSION,
    )
    recovered = EntryRecord(**dataclasses.asdict(rec))
    assert recovered.class_name == rec.class_name
    assert recovered.state_hash == rec.state_hash
    assert recovered.co_output_hashes == rec.co_output_hashes
    assert recovered.format_version == rec.format_version
    assert recovered.hash_path == str(hash_path)
    assert recovered.params_path == hash_path.parent / 'parameters.json'


# ---------------------------------------------------------------------------
# EntryInfo.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


def _make_entry_info(**kwargs) -> EntryInfo:
    defaults = dict(
        class_name='MyLoader',
        object_type='data',
        params_path='/cache/file',
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash='abc',
        instance_hash=None,
        params={'year': 2020},
        spec=SpecInfo(crs='EPSG:4326', resolution='0.1°'),
        rows=[],
        record_id='abc',
        format_version=FORMAT_VERSION,
    )
    defaults.update(kwargs)
    return EntryInfo(**defaults)


@pytest.mark.parametrize(
    'kwargs,check',
    [
        (
            {},
            lambda orig, rec: (
                rec.class_name == orig.class_name
                and rec.state_hash == orig.state_hash
                and rec.params == orig.params
                and rec.spec.crs == orig.spec.crs
            ),
        ),
        (
            {'spec': SpecInfo(bounds_latlon=(-90.0, -180.0, 90.0, 180.0))},
            lambda orig, rec: (
                isinstance(rec.spec.bounds_latlon, tuple)
                and rec.spec.bounds_latlon == (-90.0, -180.0, 90.0, 180.0)
            ),
        ),
        (
            {'primary_file': FileRef(label='out.tif', path='/data/out.tif', kind='raster')},
            lambda orig, rec: (
                rec.primary_file is not None
                and rec.primary_file.label == 'out.tif'
                and rec.primary_file.kind == 'raster'
            ),
        ),
        (
            {'primary_file': None},
            lambda orig, rec: rec.primary_file is None,
        ),
    ],
)
def test_entry_info_round_trip(kwargs, check):
    entry = _make_entry_info(**kwargs)
    recovered = EntryInfo.from_dict(entry.to_dict())
    assert check(entry, recovered)


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
# Old cache format degradation (pre-merge cache with format_version_stale bool)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('stale_flag,expected', [(True, True), (False, False)])
def test_entry_info_from_dict_old_format_version_stale(stale_flag, expected):
    """Pre-merge cache blob with format_version_stale bool must degrade gracefully."""
    old_blob = {
        'record_id': 'abc',
        'class_name': 'MyLoader',
        'object_type': None,
        'params_path': '/cache/file',
        'spec_path': None,
        'state_hash_path': None,
        'execution_graph_path': None,
        'state_hash': 'abc',
        'instance_hash': None,
        'params': {},
        'spec': {'crs': None, 'resolution': None, 'shape': None, 'bounds': None, 'bounds_latlon': None},
        'rows': [],
        'linked_entries': [],
        'co_output_hashes': [],
        'primary_file': None,
        'warnings': [],
        'error': None,
        'format_version_stale': stale_flag,
        'dep_hash': None,
        'dep_hash_stale': False,
    }
    entry = EntryInfo.from_dict(old_blob)
    assert entry.format_version_stale is expected


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
    base = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(base / 'abc', 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})
    write_hash_file(base / 'def', 'def', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'def'})
    reg = EntryRegistry()
    assert 'MyLoader' in reg.class_names
    assert len(reg.get_state_hashes('MyLoader')) == 2


def test_entry_registry_missing_state_hash(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader'})  # no STATE_HASH
    reg = EntryRegistry()
    assert reg.diagnostics()['missing_state_hash'] == 1


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
# Disk cache hit/miss by mtime
# ---------------------------------------------------------------------------


def test_entry_registry_disk_cache_hit(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})
    (tmp_path / '.source').mkdir(parents=True, exist_ok=True)

    # First scan — populates disk cache
    EntryRegistry._instance = None
    EntryRegistry()
    cache_file = tmp_path / '.source' / '.entry_registry_cache.json'
    assert cache_file.exists()

    # Second scan — should use disk cache (mtime unchanged)
    EntryRegistry._instance = None
    reg2 = EntryRegistry()
    assert 'abc' in reg2.records


def test_entry_registry_disk_cache_invalidated_on_mtime_change(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    (tmp_path / '.source').mkdir(parents=True, exist_ok=True)
    hash_path = write_hash_file(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})

    EntryRegistry._instance = None
    EntryRegistry()

    # Touch the hash file — mtime changes
    hash_path.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: 'MyLoader',
                JSONKeys.STATE_HASH: 'abc_new',
            }
        )
    )

    EntryRegistry._instance = None
    reg2 = EntryRegistry()
    assert 'abc_new' in reg2.records
    assert 'abc' not in reg2.records


# ---------------------------------------------------------------------------
# discover_entries uses EntryRegistry (no duplicate rglob)
# ---------------------------------------------------------------------------


