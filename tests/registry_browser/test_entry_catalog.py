"""Tests for pygeodata.registry_browser.entry_catalog."""
import json
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys, set_config
from pygeodata.registry_browser.entry_catalog import (
    ProcessResult,
    _cache_mtime_key,
    _deserialise_result,
    _find_primary_file,
    _is_output_file,
    _load_disk_cache,
    _object_type_from_class_name,
    _process_params_path,
    _process_with_cache,
    _save_disk_cache,
    _serialise_result,
    _unique_record_id,
    discover_entries,
)
from pygeodata.registry_browser.models import FileRef, SpecInfo
from pygeodata.paths import CachePathResolver
from pygeodata.tracked_object import TrackedObject


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path):
    """All tests get isolated cache + registry paths."""
    with set_config(
        path_cache=tmp_path / 'data_processed',
        path_figures=tmp_path / 'figures',
        path_registry=tmp_path / '.source',
    ):
        yield tmp_path


@pytest.fixture(autouse=True)
def restore_registry():
    saved = dict(TrackedObject._registry)
    yield
    TrackedObject._registry = saved
    TrackedObject.clear_function_caches()


def write_cache_entry(
    directory: Path,
    stem: str,
    params: dict | None = None,
    state: dict | None = None,
    spec: dict | None = None,
    output_ext: str = 'tif',
) -> Path:
    """Write a complete set of cache files for one entry. Returns the params path."""
    directory.mkdir(parents=True, exist_ok=True)
    params_path = directory / f'.{stem}.params.json'
    params_path.write_text(json.dumps(params or {}))
    (directory / f'.{stem}.hash.json').write_text(json.dumps(state or {}))
    (directory / f'.{stem}.spec.json').write_text(json.dumps(spec or {}))
    if output_ext:
        (directory / f'{stem}.{output_ext}').write_bytes(b'data')
    return params_path


# ---------------------------------------------------------------------------
# _is_output_file
# ---------------------------------------------------------------------------


def test_is_output_file_regular_tif(tmp_path):
    f = tmp_path / 'result.tif'
    f.write_bytes(b'x')
    assert _is_output_file(f)


def test_is_output_file_hidden_file(tmp_path):
    f = tmp_path / '.result.params.json'
    f.write_bytes(b'x')
    assert not _is_output_file(f)


def test_is_output_file_meta_suffix(tmp_path):
    for suffix in ('.params.json', '.hash.json', '.spec.json', '.graph.pdf'):
        f = tmp_path / f'file{suffix}'
        f.write_bytes(b'x')
        assert not _is_output_file(f)


def test_is_output_file_zarr_dir(tmp_path):
    d = tmp_path / 'data.zarr'
    d.mkdir()
    assert _is_output_file(d)


def test_is_output_file_regular_dir_not_output(tmp_path):
    d = tmp_path / 'subdir'
    d.mkdir()
    assert not _is_output_file(d)


# ---------------------------------------------------------------------------
# _unique_record_id
# ---------------------------------------------------------------------------


def test_unique_record_id_no_collision():
    rid, collision = _unique_record_id('abc123', '/path/to/file', set())
    assert rid == 'abc123'
    assert collision is False


def test_unique_record_id_collision_disambiguates():
    rid, collision = _unique_record_id('abc123', '/path/to/file.params.json', {'abc123'})
    assert rid != 'abc123'
    assert collision is True
    assert 'abc123' in rid


def test_unique_record_id_no_hash_uses_path():
    rid, collision = _unique_record_id(None, '/some/path', set())
    assert rid == '/some/path'
    assert collision is False


def test_unique_record_id_multiple_collisions():
    taken = {'abc123', 'abc123/file.params', 'abc123/file.params_1'}
    rid, collision = _unique_record_id('abc123', '/path/to/file.params.json', taken)
    assert rid == 'abc123/file.params_2'
    assert collision is True


# ---------------------------------------------------------------------------
# _serialise_result / _deserialise_result round-trip
# ---------------------------------------------------------------------------


def make_process_result(**kwargs):
    defaults = dict(
        class_name='MyLoader',
        object_type='data',
        params_path_str='/cache/file',
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash='abc',
        instance_hash=None,
        stored_dep_hash=None,
        co_output_hashes=[],
        params={'year': 2020},
        spec=SpecInfo(crs='EPSG:4326', resolution='0.1°'),
        rows=[],
        linked_entries=[],
        primary_file=None,
        warnings=[],
        derived=False,
        error=None,
    )
    defaults.update(kwargs)
    return ProcessResult(**defaults)


def test_serialise_deserialise_roundtrip():
    result = make_process_result()
    serialised = _serialise_result(result)
    recovered = _deserialise_result(dict(serialised))
    assert recovered.class_name == result.class_name
    assert recovered.state_hash == result.state_hash
    assert recovered.params == result.params
    assert recovered.spec.crs == result.spec.crs


def test_serialise_deserialise_bounds_latlon_tuple():
    result = make_process_result(spec=SpecInfo(bounds_latlon=(-90, -180, 90, 180)))
    serialised = _serialise_result(result)
    recovered = _deserialise_result(dict(serialised))
    assert isinstance(recovered.spec.bounds_latlon, tuple)
    assert recovered.spec.bounds_latlon == (-90, -180, 90, 180)


def test_serialise_deserialise_primary_file():
    result = make_process_result(
        primary_file=FileRef(label='out.tif', path='/data/out.tif', kind='raster')
    )
    serialised = _serialise_result(result)
    recovered = _deserialise_result(dict(serialised))
    assert recovered.primary_file.label == 'out.tif'
    assert recovered.primary_file.kind == 'raster'


def test_serialise_deserialise_no_primary_file():
    result = make_process_result(primary_file=None)
    serialised = _serialise_result(result)
    recovered = _deserialise_result(dict(serialised))
    assert recovered.primary_file is None


# ---------------------------------------------------------------------------
# _load_disk_cache / _save_disk_cache
# ---------------------------------------------------------------------------


def test_load_disk_cache_missing_file(tmp_path):
    with set_config(path_registry=tmp_path / '.source'):
        results, mtimes = _load_disk_cache()
    assert results == {}
    assert mtimes == {}


def test_save_and_load_disk_cache(tmp_path):
    with set_config(path_registry=tmp_path / '.source'):
        (tmp_path / '.source').mkdir(parents=True)
        _save_disk_cache({'key': {'a': 1}}, {'key': 1.23})
        results, mtimes = _load_disk_cache()
    assert results == {'key': {'a': 1}}
    assert mtimes == {'key': pytest.approx(1.23)}


def test_load_disk_cache_corrupted_file(tmp_path):
    reg = tmp_path / '.source'
    reg.mkdir(parents=True)
    (reg / '.dashboard_cache.json').write_text('not json')
    with set_config(path_registry=reg):
        results, mtimes = _load_disk_cache()
    assert results == {}
    assert mtimes == {}


# ---------------------------------------------------------------------------
# _cache_mtime_key
# ---------------------------------------------------------------------------


def test_cache_mtime_key_sums_existing_files(tmp_path):
    d = tmp_path / 'MyLoader'
    params_path = write_cache_entry(d, 'abc', params={'year': 2020})
    total = _cache_mtime_key(params_path)
    assert total > 0


def test_cache_mtime_key_missing_files_contribute_zero(tmp_path):
    d = tmp_path / 'MyLoader'
    d.mkdir()
    p = d / '.abc.params.json'
    p.write_text('{}')
    total = _cache_mtime_key(p)
    assert isinstance(total, float)


# ---------------------------------------------------------------------------
# _process_params_path
# ---------------------------------------------------------------------------


def test_process_params_path_basic(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        params={'year': 2020},
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc123'},
        spec={'crs': 'EPSG:4326'},
    )
    result = _process_params_path(params_path)
    assert result.error is None
    assert result.class_name == 'MyLoader'
    assert result.state_hash == 'abc123'
    assert result.spec.crs == 'EPSG:4326'
    assert result.params == {'year': 2020}


def test_process_params_path_missing_class_name_warns(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.STATE_HASH: 'abc'},  # no CLASS_NAME
    )
    result = _process_params_path(params_path)
    assert result.derived is True
    assert any('derived' in w.lower() or 'class name' in w.lower() for w in result.warnings)


def test_process_params_path_missing_state_hash_warns(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader'},  # no STATE_HASH
    )
    result = _process_params_path(params_path)
    assert result.state_hash is None
    assert any('hash' in w.lower() for w in result.warnings)


def test_process_params_path_finds_primary_file(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext='tif',
    )
    result = _process_params_path(params_path)
    assert result.primary_file is not None
    assert result.primary_file.label == 'abc.tif'


def test_process_params_path_no_primary_file_when_missing(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext=None,
    )
    result = _process_params_path(params_path)
    assert result.primary_file is None


def test_process_params_path_co_output_hashes(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.STATE_HASH: 'abc',
            JSONKeys.CO_OUTPUTS: ['hash1', 'hash2'],
        },
    )
    result = _process_params_path(params_path)
    assert result.co_output_hashes == ['hash1', 'hash2']


# ---------------------------------------------------------------------------
# _process_with_cache
# ---------------------------------------------------------------------------


def test_process_with_cache_cold_miss(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
    )
    result, from_cache = _process_with_cache(params_path, {}, {})
    assert not from_cache
    assert result.class_name == 'MyLoader'


def test_process_with_cache_hit(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
    )
    key = str(params_path.resolve())
    mtime = _cache_mtime_key(params_path)

    fresh, _ = _process_with_cache(params_path, {}, {})
    serialised = _serialise_result(fresh)

    result, from_cache = _process_with_cache(params_path, {key: serialised}, {key: mtime})
    assert from_cache
    assert result.class_name == 'MyLoader'


def test_process_with_cache_miss_on_mtime_change(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
    )
    key = str(params_path.resolve())
    fresh, _ = _process_with_cache(params_path, {}, {})
    serialised = _serialise_result(fresh)

    stale_mtime = 0.0  # definitely wrong
    result, from_cache = _process_with_cache(params_path, {key: serialised}, {key: stale_mtime})
    assert not from_cache


# ---------------------------------------------------------------------------
# discover_entries (integration)
# ---------------------------------------------------------------------------


def test_discover_entries_empty_cache(tmp_path):
    entries, groups, diag = discover_entries()
    assert entries == {}
    assert groups == {}
    assert diag['scanned_params_paths'] == 0
    assert diag['created_entries'] == 0


def test_discover_entries_single_entry(tmp_path):
    from pygeodata.data import Data

    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(
        d, 'abc',
        params={'year': 2020},
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        spec={'crs': 'EPSG:4326'},
    )

    entries, groups, diag = discover_entries()
    assert diag['created_entries'] == 1
    assert 'abc' in entries
    entry = entries['abc']
    assert entry.class_name == 'MyLoader'
    assert entry.state_hash == 'abc'
    assert entry.spec.crs == 'EPSG:4326'


def test_discover_entries_groups_populated(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})
    write_cache_entry(d, 'def', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'def'})

    entries, groups, diag = discover_entries()
    assert 'MyLoader' in groups
    assert len(groups['MyLoader'].record_ids) == 2


def test_discover_entries_missing_state_hash_diagnostic(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(
        d, 'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader'},  # no STATE_HASH
    )
    _, _, diag = discover_entries()
    assert len(diag['missing_state_hash']) == 1


def test_discover_entries_derived_class_name_diagnostic(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'abc', state={JSONKeys.STATE_HASH: 'abc'})  # no CLASS_NAME
    _, _, diag = discover_entries()
    assert len(diag['derived_class_name']) == 1


def test_discover_entries_hash_collision_diagnostic(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'same'})
    write_cache_entry(d, 'xyz', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'same'})

    entries, _, diag = discover_entries()
    assert len(diag['hash_collisions']) >= 1
    assert len(entries) == 2  # both entries created with disambiguated IDs


def test_discover_entries_co_outputs_resolved(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'child', state={JSONKeys.CLASS_NAME: 'Child', JSONKeys.STATE_HASH: 'child_hash'})
    write_cache_entry(
        d, 'parent',
        state={
            JSONKeys.CLASS_NAME: 'Parent',
            JSONKeys.STATE_HASH: 'parent_hash',
            JSONKeys.CO_OUTPUTS: ['child_hash'],
        },
    )
    entries, _, _ = discover_entries()
    parent = entries['parent_hash']
    assert any(e.record_id == 'child_hash' for e in parent.co_outputs)


def test_discover_entries_progress_tracking(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})

    progress: dict = {}
    discover_entries(progress=progress)
    assert progress['done'] == 1
    assert progress['total'] == 1


def test_discover_entries_writes_disk_cache(tmp_path):
    reg = tmp_path / '.source'
    reg.mkdir(parents=True)
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})

    discover_entries()

    cache_file = reg / '.dashboard_cache.json'
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert 'results' in data
    assert 'mtimes' in data


def test_discover_entries_uses_disk_cache_on_second_call(tmp_path):
    reg = tmp_path / '.source'
    reg.mkdir(parents=True)
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(d, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})

    entries1, _, _ = discover_entries()
    entries2, _, _ = discover_entries()

    assert set(entries1.keys()) == set(entries2.keys())
