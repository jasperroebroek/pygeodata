"""Tests for pygeodata.api.load_from_hash / pygeodata.load_from_hash."""
import json
from pathlib import Path

import pytest

import pygeodata
from pygeodata.api import load_from_hash
from pygeodata.config import FORMAT_VERSION, JSONKeys, set_config
from pygeodata.paths import CACHE_META_FILES


def _write_meta(directory: Path, state_hash: str, class_name: str, object_type: str) -> Path:
    """Write a meta.json so EntryRegistry can discover this entry."""
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / 'meta.json'
    meta_path.write_text(
        json.dumps({
            JSONKeys.CLASS_NAME: class_name,
            JSONKeys.STATE_HASH: state_hash,
            JSONKeys.OBJECT_TYPE: object_type,
            JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
        }),
        encoding='utf-8',
    )
    return meta_path


@pytest.fixture
def cache_root(tmp_path):
    """Override the config cache/registry paths and return the cache root."""
    cache = tmp_path / 'cache'
    cache.mkdir()
    with set_config(
        path_cache=cache,
        path_figures=tmp_path / 'figures',
        path_registry=tmp_path / '.source',
    ):
        yield cache


def test_load_from_hash_success(cache_root):
    """Hash resolves and loads the output file without re-running process."""
    from tests.fixtures.data import SimpleLoader

    state_hash = 'aabbcc1234567890' * 4

    cache_dir = cache_root / 'SimpleLoader' / state_hash
    _write_meta(cache_dir, state_hash, 'SimpleLoader', 'Data')
    output_path = cache_dir / 'output.tif'
    output_path.write_bytes(b'tif_content')

    visited = []

    def capturing_driver(path):
        visited.append(path)
        return path.read_bytes()

    SimpleLoader.driver = property(lambda self: capturing_driver)
    try:
        result = load_from_hash(state_hash[:8])
    finally:
        del SimpleLoader.driver

    assert result == b'tif_content'
    assert len(visited) == 1
    assert visited[0] == output_path


def test_load_from_hash_skips_meta_files(cache_root):
    """Output file resolution skips all CACHE_META_FILES entries."""
    from tests.fixtures.data import SimpleLoader

    state_hash = 'deadbeef1234abcd' * 4

    cache_dir = cache_root / 'SimpleLoader' / state_hash
    _write_meta(cache_dir, state_hash, 'SimpleLoader', 'Data')

    for name in CACHE_META_FILES - {'meta.json'}:
        (cache_dir / name).write_bytes(b'')
    (cache_dir / 'output.nc').write_bytes(b'real_output')

    visited = []

    def capturing_driver(path):
        visited.append(path)
        return path

    SimpleLoader.driver = property(lambda self: capturing_driver)
    try:
        load_from_hash(state_hash)
    finally:
        del SimpleLoader.driver

    assert len(visited) == 1
    assert visited[0].name == 'output.nc'


def test_load_from_hash_no_match_raises_key_error(cache_root):
    """No matching entry for prefix raises KeyError."""
    with pytest.raises(KeyError, match='No entry found for hash prefix'):
        load_from_hash('0000000000000000')


def test_load_from_hash_ambiguous_prefix_raises_key_error(cache_root):
    """Multiple entries matching the same prefix raise KeyError (resolve returns None)."""
    state_hash_a = 'aabbcc1234567890' * 4
    state_hash_b = 'aabbcc9876543210' * 4

    for h in (state_hash_a, state_hash_b):
        cache_dir = cache_root / 'SimpleLoader' / h
        _write_meta(cache_dir, h, 'SimpleLoader', 'Data')

    with pytest.raises(KeyError, match='No entry found for hash prefix'):
        load_from_hash('aabbcc')


def test_load_from_hash_no_hash_path_raises_file_not_found(cache_root, monkeypatch):
    """A record whose hash_path is None raises FileNotFoundError."""
    from pygeodata.registry import EntryRegistry
    from pygeodata.registry_types import EntryRecord

    state_hash = 'deadbeef' * 8

    record = EntryRecord(
        class_name='SimpleLoader',
        state_hash=state_hash,
        object_type='Data',
        hash_path=None,
    )

    monkeypatch.setattr(EntryRegistry, 'resolve_hash_prefix', lambda self, prefix: state_hash)
    monkeypatch.setattr(EntryRegistry, 'records', property(lambda self: {state_hash: record}))

    with pytest.raises(FileNotFoundError, match=state_hash):
        load_from_hash(state_hash[:8])


def test_load_from_hash_no_output_file_raises_file_not_found(cache_root):
    """An entry dir with only meta files and no output raises FileNotFoundError."""
    state_hash = 'cafebabe12345678' * 4

    cache_dir = cache_root / 'SimpleLoader' / state_hash
    _write_meta(cache_dir, state_hash, 'SimpleLoader', 'Data')

    for name in CACHE_META_FILES - {'meta.json'}:
        (cache_dir / name).write_bytes(b'')

    with pytest.raises(FileNotFoundError, match=state_hash):
        load_from_hash(state_hash)


def test_load_from_hash_unregistered_class_raises_runtime_error(cache_root):
    """A class_name not found in TrackedObject registry raises RuntimeError."""
    state_hash = 'beefcafe12345678' * 4
    unknown_class = 'NonExistentDataClass99'

    cache_dir = cache_root / unknown_class / state_hash
    _write_meta(cache_dir, state_hash, unknown_class, 'Data')
    (cache_dir / 'output.tif').write_bytes(b'data')

    with pytest.raises(RuntimeError, match=unknown_class):
        load_from_hash(state_hash)


def test_load_from_hash_exported_from_pygeodata():
    """load_from_hash is importable directly from the pygeodata package."""
    assert pygeodata.load_from_hash is load_from_hash
    assert 'load_from_hash' in pygeodata.__all__
