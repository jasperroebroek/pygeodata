import json
from pathlib import Path

import pytest

from pygeodata.cache import handle_invalid, is_zarr_root, path_matches_hash, prune_empty_dirs
from pygeodata.config import JSONKeys
from pygeodata.paths import CachePathResolver
from tests.fixtures.data import EmptyLoader


def write_hash(path: Path, hash_value: str) -> None:
    path.write_text(json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: hash_value}))


# --- path_matches_hash ---

def test_path_matches_hash_true(tmp_path: Path) -> None:
    source_hash = EmptyLoader.get_dependency_tree_hash()
    hash_file = tmp_path / '.dummy.hash.json'
    write_hash(hash_file, source_hash)
    assert path_matches_hash(hash_file, source_hash)


def test_path_matches_hash_false_wrong_hash(tmp_path: Path) -> None:
    hash_file = tmp_path / '.dummy.hash.json'
    write_hash(hash_file, 'stale')
    assert not path_matches_hash(hash_file, 'current')


def test_path_matches_hash_false_missing(tmp_path: Path) -> None:
    assert not path_matches_hash(tmp_path / 'nonexistent.hash.json', 'any')


def test_path_matches_hash_missing_key(tmp_path: Path) -> None:
    hash_file = tmp_path / '.dummy.hash.json'
    hash_file.write_text(json.dumps({'other_key': 'value'}))
    assert not path_matches_hash(hash_file, 'value')


# --- CachePathResolver hash paths ---

def test_get_hash_path_regular_file(tmp_path: Path) -> None:
    assert CachePathResolver.from_path(tmp_path / 'data.tif').state_hash_path == tmp_path / '.data.hash.json'


def test_get_hash_path_dotfile(tmp_path: Path) -> None:
    assert CachePathResolver.from_path(tmp_path / '.data.hash.json').state_hash_path == tmp_path / '.data.hash.json'


def test_get_hash_path_zarr(tmp_path: Path) -> None:
    assert CachePathResolver.from_path(tmp_path / 'archive.zarr').state_hash_path == tmp_path / '.archive.hash.json'


def test_get_hash_path_multi_extension(tmp_path: Path) -> None:
    assert CachePathResolver.from_path(tmp_path / 'file.tar.gz').state_hash_path == tmp_path / '.file.hash.json'


# --- is_zarr_root ---

def test_is_zarr_root_by_suffix(tmp_path: Path) -> None:
    zarr_dir = tmp_path / 'archive.zarr'
    zarr_dir.mkdir()
    assert is_zarr_root(zarr_dir)


def test_is_zarr_root_v3_marker(tmp_path: Path) -> None:
    zarr_dir = tmp_path / 'archive'
    zarr_dir.mkdir()
    (zarr_dir / 'zarr.json').touch()
    assert is_zarr_root(zarr_dir)


@pytest.mark.parametrize('marker', ['.zgroup', '.zarray', '.zattrs', '.zmetadata'])
def test_is_zarr_root_v2_markers(tmp_path: Path, marker: str) -> None:
    zarr_dir = tmp_path / 'archive'
    zarr_dir.mkdir()
    (zarr_dir / marker).touch()
    assert is_zarr_root(zarr_dir)


def test_is_zarr_root_false(tmp_path: Path) -> None:
    regular_dir = tmp_path / 'regular'
    regular_dir.mkdir()
    (regular_dir / 'data.tif').touch()
    assert not is_zarr_root(regular_dir)


# --- handle_invalid ---

def test_handle_invalid_dry_run_hash_missing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / 'data.tif'
    hash_path = tmp_path / '.data.hash.json'
    handle_invalid(path, dry_run=True, hash_path=hash_path)
    assert 'Hash missing' in capsys.readouterr().out


def test_handle_invalid_dry_run_hash_wrong(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / 'data.tif'
    hash_path = tmp_path / '.data.hash.json'
    hash_path.touch()
    handle_invalid(path, dry_run=True, hash_path=hash_path)
    assert 'Hash wrong' in capsys.readouterr().out


def test_handle_invalid_none_hash_path_labels_invalid(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / 'data.tif'
    handle_invalid(path, dry_run=True, hash_path=None)
    assert 'Invalid' in capsys.readouterr().out


def test_handle_invalid_non_dry_run_prints_deleting(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / 'data.tif'
    path.touch()
    handle_invalid(path, dry_run=False, hash_path=tmp_path / '.data.hash.json')
    assert 'Deleting' in capsys.readouterr().out


def test_handle_invalid_deletes_file(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / 'data.tif'
    path.touch()
    hash_path = tmp_path / '.data.hash.json'
    handle_invalid(path, dry_run=False, hash_path=hash_path)
    assert not path.exists()


def test_handle_invalid_deletes_directory(tmp_path: Path) -> None:
    directory = tmp_path / 'archive.zarr'
    directory.mkdir()
    (directory / 'zarr.json').touch()
    handle_invalid(directory, dry_run=False, hash_path=tmp_path / '.archive.hash.json')
    assert not directory.exists()


# --- prune_empty_dirs ---

def test_prune_empty_dirs_removes_empty_subdirs(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    empty = tmp_path / 'sub' / 'empty'
    empty.mkdir(parents=True)
    prune_empty_dirs(tmp_path)
    assert not empty.exists()
    assert 'Empty dir' in capsys.readouterr().out


def test_prune_empty_dirs_keeps_non_empty_dirs(tmp_path: Path) -> None:
    sub = tmp_path / 'sub'
    sub.mkdir()
    (sub / 'data.tif').touch()
    prune_empty_dirs(tmp_path)
    assert sub.exists()


def test_prune_empty_dirs_does_not_remove_root(tmp_path: Path) -> None:
    prune_empty_dirs(tmp_path)
    assert tmp_path.exists()
