"""Tests for pygeodata.registry_browser.entry_catalog.

ProcessResult, _serialise_result, and _deserialise_result no longer exist.
Round-trip coverage is now in test_entry_registry.py (EntryInfo.to_dict/from_dict).
This file covers the display-enrichment helpers and discover_entries integration.
"""

import json
from pathlib import Path

import pytest

from pygeodata.catalog import store
from pygeodata.catalog.display import is_output_file
from pygeodata.catalog.entry_catalog import _enrich_params_path, discover_entries
from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config, set_config
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.registry import EntryRegistry
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


def _make_tif(p):
    f = p / 'result.tif'
    f.write_bytes(b'x')
    return f


def _make_zarr(p):
    d = p / 'data.zarr'
    d.mkdir()
    return d


def _make_subdir(p):
    d = p / 'subdir'
    d.mkdir()
    return d


@pytest.mark.parametrize(
    'make_path,expected',
    [
        (_make_tif, True),
        (_make_zarr, True),
        (_make_subdir, False),
    ],
)
def test_is_output_file(tmp_path, make_path, expected):
    path = make_path(tmp_path)
    assert is_output_file(path) is expected


def test_is_output_file_meta_files(tmp_path):
    for name in ('parameters.json', 'meta.json', 'spec.json', 'graph.pdf', 'process.lock'):
        f = tmp_path / name
        f.write_bytes(b'x')
        assert not is_output_file(f)


# ---------------------------------------------------------------------------
def test_enrich_params_path_fields(tmp_path):
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

    # missing STATE_HASH → warns
    d2 = tmp_path / 'data_processed' / 'MyLoader2'
    params_path2 = write_cache_entry(d2, 'abc', state={JSONKeys.CLASS_NAME: 'MyLoader'})
    entry2 = _enrich_params_path(params_path2)
    assert entry2.state_hash is None
    assert any('hash' in w.lower() for w in entry2.warnings)

    # primary file present
    d3 = tmp_path / 'data_processed' / 'MyLoader3'
    params_path3 = write_cache_entry(
        d3,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext='tif',
    )
    entry3 = _enrich_params_path(params_path3)
    assert entry3.primary_file is not None
    assert entry3.primary_file.label == 'abc.tif'

    # primary file absent
    d4 = tmp_path / 'data_processed' / 'MyLoader4'
    params_path4 = write_cache_entry(
        d4,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext=None,
    )
    entry4 = _enrich_params_path(params_path4)
    assert entry4.primary_file is None


def test_enrich_params_path_sidecar_files_surfaced(tmp_path):
    """A Table entry's declared .html output is primary; a sibling .csv is a sidecar,
    not silently dropped (guards the entry_catalog.py primary-file-selection fix).
    """
    d = tmp_path / 'tables' / 'MyTable'
    params_path = write_cache_entry(
        d,
        'my_table',
        state={JSONKeys.CLASS_NAME: 'MyTable', JSONKeys.STATE_HASH: 'abc', JSONKeys.OBJECT_TYPE: 'Table'},
        output_ext='html',
    )
    (d / 'my_table.csv').write_bytes(b'a,b\n1,2\n')

    entry = _enrich_params_path(params_path)
    assert entry.primary_file is not None
    assert entry.primary_file.label == 'my_table.html'
    assert [f.label for f in entry.sidecar_files] == ['my_table.csv']


def test_enrich_params_path_no_sidecars_when_single_output(tmp_path):
    d = tmp_path / 'data_processed' / 'MyLoader5'
    params_path = write_cache_entry(
        d,
        'abc',
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        output_ext='tif',
    )
    entry = _enrich_params_path(params_path)
    assert entry.sidecar_files == []


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


def test_discover_entries_absolute_paths_from_catalog(tmp_path, monkeypatch):
    d = tmp_path / 'data_processed' / 'MyLoader'
    write_cache_entry(
        d,
        'abc',
        params={'year': 2020},
        state={JSONKeys.CLASS_NAME: 'MyLoader', JSONKeys.STATE_HASH: 'abc'},
        spec={'crs': 'EPSG:4326'},
    )

    # A catalog built from a project opened at its root stores relative entry dirs;
    # hydrated EntryInfo paths must still be absolute (e.g. for send_file).
    monkeypatch.chdir(tmp_path)
    with set_config(path_cache=Path('data_processed'), path_figures=Path('figures'), path_registry=Path('.source')):
        EntryRegistry._instance = None
        store.refresh(get_config().path_registry, get_config().cache_roots, FORMAT_VERSION)
        entries, _, diag = discover_entries()
    assert diag['created_entries'] == 1
    entry = entries['abc']
    assert Path(entry.params_path).is_absolute()
    assert Path(entry.params_path).exists()
