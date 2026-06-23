"""Unit tests for registry_browser/export_service.py."""

import json
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys, set_config
from pygeodata.registry_browser import export_service
from pygeodata.registry_browser.models import EntryInfo, SpecInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(record_id: str, params_path: str, dep_hash: str | None = None) -> EntryInfo:
    return EntryInfo(
        record_id=record_id,
        class_name='MyLoader',
        object_type='Data',
        params_path=params_path,
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash=record_id,
        instance_hash=None,
        params={},
        spec=SpecInfo(),
        rows=[],
        dep_hash=dep_hash,
    )


_noop_allowed = lambda path_str: None


def _rejecting_allowed(path_str: str):
    raise Exception(f'path not allowed: {path_str}')


# ---------------------------------------------------------------------------
# collect_export_files — cache members
# ---------------------------------------------------------------------------


def test_collect_cache_files(tmp_path: Path):
    registry = tmp_path / '.source'
    cache_root = tmp_path / 'data'
    cache_dir = cache_root / 'abc'
    from pygeodata.paths import CachePathConstructor

    cache_dir.mkdir(parents=True)
    (cache_dir / 'output.tif').write_bytes(b'TIFF')
    resolver = CachePathConstructor(cache_dir)
    resolver.params_path.write_text('{}')

    entry = _make_entry('abc', str(resolver.params_path))

    with set_config(path_cache=cache_root, path_figures=tmp_path / 'figs', path_registry=registry):
        files = export_service.collect_export_files(['abc'], {'abc': entry}, False, _noop_allowed)

    arcnames = [a for _, a in files]
    assert 'cache/abc/output.tif' in arcnames
    assert 'cache/abc/parameters.json' in arcnames
    assert not any(a.startswith('snapshots/') for a in arcnames)
    assert not any(a.startswith('code/') for a in arcnames)


def test_collect_snapshot_and_code_members(tmp_path: Path):
    registry = tmp_path / '.source'
    cache_root = tmp_path / 'data'
    cache_dir = cache_root / 'abc'
    cache_dir.mkdir(parents=True)
    (cache_dir / 'output.tif').write_bytes(b'TIFF')

    src_hash = 'src001'
    code_dir = registry / 'code' / src_hash
    code_dir.mkdir(parents=True)
    (code_dir / 'source.py').write_text('class MyLoader: pass')
    (code_dir / 'source.json').write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: 'MyLoader',
                JSONKeys.OBJECT_TYPE: 'Data',
                JSONKeys.SOURCE_HASH: src_hash,
                JSONKeys.REGISTERED_AT: '2026-01-01T00:00:00+00:00',
            },
        ),
    )

    dep_hash = 'dep001'
    snapshot_dir = registry / 'snapshots' / dep_hash
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / 'tree.json').write_text(
        json.dumps(
            {
                JSONKeys.NODES: {'MyLoader': {'hash': src_hash, 'object_type': 'Data'}},
                JSONKeys.TREE: {},
            },
        ),
    )

    entry = _make_entry('abc', str(cache_dir / 'output.tif'), dep_hash=dep_hash)

    with set_config(path_cache=cache_root, path_figures=tmp_path / 'figs', path_registry=registry):
        files = export_service.collect_export_files(['abc'], {'abc': entry}, True, _noop_allowed)

    arcnames = [a for _, a in files]
    assert 'cache/abc/output.tif' in arcnames
    assert f'snapshots/{dep_hash}/tree.json' in arcnames
    assert f'code/{src_hash}/source.py' in arcnames
    assert f'code/{src_hash}/source.json' in arcnames


def test_collect_skips_unknown_record_id(tmp_path: Path):
    registry = tmp_path / '.source'
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        files = export_service.collect_export_files(['no_such_id'], {}, False, _noop_allowed)
    assert files == []


def test_collect_calls_assert_allowed_path(tmp_path: Path):
    """assert_allowed_path must be invoked for every cache directory."""
    registry = tmp_path / '.source'
    cache_root = tmp_path / 'data'
    cache_dir = cache_root / 'abc'
    cache_dir.mkdir(parents=True)
    (cache_dir / 'output.tif').write_bytes(b'TIFF')

    entry = _make_entry('abc', str(cache_dir / 'output.tif'))

    with set_config(path_cache=cache_root, path_figures=tmp_path / 'figs', path_registry=registry):
        with pytest.raises(Exception, match='path not allowed'):
            export_service.collect_export_files(['abc'], {'abc': entry}, False, _rejecting_allowed)
