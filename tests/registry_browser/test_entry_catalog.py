"""Tests for pygeodata.registry_browser.entry_catalog.

ProcessResult, _serialise_result, and _deserialise_result no longer exist.
Round-trip coverage is now in test_entry_registry.py (EntryInfo.to_dict/from_dict).
This file covers the display-enrichment helpers and discover_entries integration.
"""

import json
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys, set_config
from pygeodata.paths import CachePathConstructor
from pygeodata.registry import EntryRegistry
from pygeodata.registry_browser.entry_catalog import (
    _cache_mtime_key,
    _enrich_params_path,
    _enrich_with_cache,
    _is_output_file,
    _load_disk_cache,
    _save_disk_cache,
    discover_entries,
)
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
        EntryRegistry._instance = None
        yield tmp_path
        EntryRegistry._instance = None


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
    resolver = CachePathConstructor(directory)
    resolver.params_path.write_text(json.dumps(params or {}))
    resolver.state_hash_path.write_text(json.dumps(state or {}))
    resolver.spec_path.write_text(json.dumps(spec or {}))
    if output_ext:
        (directory / f'{stem}.{output_ext}').write_bytes(b'data')
    return resolver.params_path


# ---------------------------------------------------------------------------
# _is_output_file
# ---------------------------------------------------------------------------


def test_is_output_file_regular_tif(tmp_path):
    f = tmp_path / 'result.tif'
    f.write_bytes(b'x')
    assert _is_output_file(f)


def test_is_output_file_meta_files(tmp_path):
    for name in ('parameters.json', 'meta.json', 'spec.json', 'graph.pdf', 'process.lock'):
        f = tmp_path / name
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
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(d, 'abc', params={'year': 2020})
    total = _cache_mtime_key(params_path)
    assert total > 0


def test_cache_mtime_key_missing_files_contribute_zero(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    d.mkdir(parents=True)
    resolver = CachePathConstructor(d)
    resolver.params_path.write_text('{}')
    total = _cache_mtime_key(resolver.params_path)
    assert isinstance(total, float)


# ---------------------------------------------------------------------------
# _enrich_params_path  (was _process_params_path)
# ---------------------------------------------------------------------------


def test_enrich_params_path_basic(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        params={'year': 2020},
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc123'},
        spec={'crs': 'EPSG:4326'},
    )
    entry = _enrich_params_path(params_path)
    assert entry.error is None
    assert entry.class_name == 'MyLoader'
    assert entry.state_hash == 'abc123'
    assert entry.spec.crs == 'EPSG:4326'
    assert entry.params == {'year': 2020}


def test_enrich_params_path_missing_state_hash_warns(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader'},  # no STATE_HASH
    )
    entry = _enrich_params_path(params_path)
    assert entry.state_hash is None
    assert any('hash' in w.lower() for w in entry.warnings)


def test_enrich_params_path_finds_primary_file(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext='tif',
    )
    entry = _enrich_params_path(params_path)
    assert entry.primary_file is not None
    assert entry.primary_file.label == 'abc.tif'


def test_enrich_params_path_no_primary_file_when_missing(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext=None,
    )
    entry = _enrich_params_path(params_path)
    assert entry.primary_file is None


def test_enrich_params_path_co_output_hashes(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.STATE_HASH: 'abc',
            JSONKeys.CO_OUTPUTS: ['hash1', 'hash2'],
        },
    )
    entry = _enrich_params_path(params_path)
    assert entry.co_output_hashes == ['hash1', 'hash2']


# ---------------------------------------------------------------------------
# _enrich_with_cache  (was _process_with_cache)
# ---------------------------------------------------------------------------


def test_enrich_with_cache_cold_miss(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
    )
    entry, from_cache = _enrich_with_cache(params_path, {}, {})
    assert not from_cache
    assert entry.class_name == 'MyLoader'


def test_enrich_with_cache_hit(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
    )
    key = str(params_path.resolve())
    mtime = _cache_mtime_key(params_path)

    fresh, _ = _enrich_with_cache(params_path, {}, {})
    serialised = fresh.to_dict()

    entry, from_cache = _enrich_with_cache(params_path, {key: serialised}, {key: mtime})
    assert from_cache
    assert entry.class_name == 'MyLoader'


def test_enrich_with_cache_miss_on_mtime_change(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
    )
    key = str(params_path.resolve())
    fresh, _ = _enrich_with_cache(params_path, {}, {})
    serialised = fresh.to_dict()

    stale_mtime = 0.0
    entry, from_cache = _enrich_with_cache(params_path, {key: serialised}, {key: stale_mtime})
    assert not from_cache


# ---------------------------------------------------------------------------
# discover_entries (integration)
# ---------------------------------------------------------------------------


def test_discover_entries_empty_cache(tmp_path):
    entries, entry_registry, diag = discover_entries()
    assert entries == {}
    assert entry_registry.records == {}
    assert diag['created_entries'] == 0


def test_discover_entries_single_entry(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(
        d,
        'abc',
        params={'year': 2020},
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        spec={'crs': 'EPSG:4326'},
    )

    entries, entry_registry, diag = discover_entries()
    assert diag['created_entries'] == 1
    assert 'abc' in entries
    entry = entries['abc']
    assert entry.class_name == 'MyLoader'
    assert entry.state_hash == 'abc'
    assert entry.spec.crs == 'EPSG:4326'


def test_discover_entries_groups_populated(tmp_path):
    base = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(base / 'abc', 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'})
    write_cache_entry(base / 'def', 'def', state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'def'})

    entries, entry_registry, diag = discover_entries()
    assert 'MyLoader' in entry_registry.class_names
    assert len(entry_registry.get_state_hashes('MyLoader')) == 2


def test_discover_entries_missing_state_hash_diagnostic(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader'},  # no STATE_HASH
    )
    _, _, diag = discover_entries()
    assert diag['missing_state_hash'] == 1


def test_discover_entries_co_outputs_resolved(tmp_path):
    base = tmp_path / 'data_processed'
    write_cache_entry(base / 'Child', 'child', state={JSONKeys.CLASS_NAME: 'Child', JSONKeys.STATE_HASH: 'child_hash'})
    write_cache_entry(
        base / 'Parent',
        'parent',
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
