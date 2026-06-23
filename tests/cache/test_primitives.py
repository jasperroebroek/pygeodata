import json
from pathlib import Path

import pytest

from pygeodata.cache import handle_invalid, hash_matches_live, is_zarr_root, prune_empty_dirs
from pygeodata.config import JSONKeys
from pygeodata.paths import CachePathConstructor
from tests.fixtures.data import EmptyLoader

# --- hash_matches_live ---


def test_hash_matches_live_true(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
    hash_file.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: EmptyLoader.get_class_name(),
                JSONKeys.DEPENDENCY_TREE_HASH: EmptyLoader.get_dependency_tree_hash(),
            },
        ),
    )
    assert hash_matches_live(hash_file) is True


def test_hash_matches_live_false_wrong_hash(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
    hash_file.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: EmptyLoader.get_class_name(),
                JSONKeys.DEPENDENCY_TREE_HASH: 'stale',
            },
        ),
    )
    assert hash_matches_live(hash_file) is False


def test_hash_matches_live_false_when_missing(tmp_path: Path) -> None:
    assert hash_matches_live(tmp_path / 'meta.json') is False


def test_hash_matches_live_missing_key(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
    hash_file.write_text(json.dumps({JSONKeys.CLASS_NAME: EmptyLoader.get_class_name()}))
    assert hash_matches_live(hash_file) is False


def test_hash_matches_live_none_when_unregistered(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
    hash_file.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: 'NoSuchClass',
                JSONKeys.DEPENDENCY_TREE_HASH: 'abc',
            },
        ),
    )
    assert hash_matches_live(hash_file) is None


# --- CachePathResolver paths ---


def test_state_hash_path(tmp_path: Path) -> None:
    assert CachePathConstructor(tmp_path).state_hash_path == tmp_path / 'meta.json'


def test_params_path(tmp_path: Path) -> None:
    assert CachePathConstructor(tmp_path).params_path == tmp_path / 'parameters.json'


def test_spec_path(tmp_path: Path) -> None:
    assert CachePathConstructor(tmp_path).spec_path == tmp_path / 'spec.json'


def test_execution_graph_path(tmp_path: Path) -> None:
    assert CachePathConstructor(tmp_path).execution_graph_path == tmp_path / 'graph.pdf'


def test_from_path_uses_parent(tmp_path: Path) -> None:
    assert CachePathConstructor.from_path(tmp_path / 'meta.json').directory == tmp_path


def test_from_state_hash(tmp_path: Path) -> None:
    resolver = CachePathConstructor.from_state_hash('abc123', tmp_path)
    assert resolver.directory == tmp_path / 'abc123'
    assert resolver.state_hash_path == tmp_path / 'abc123' / 'meta.json'


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
    handle_invalid(tmp_path / 'data.tif', dry_run=True, label='Hash missing')
    assert 'Hash missing' in capsys.readouterr().out


def test_handle_invalid_dry_run_hash_wrong(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    handle_invalid(tmp_path / 'data.tif', dry_run=True, label='Hash wrong')
    assert 'Hash wrong' in capsys.readouterr().out


def test_handle_invalid_no_hash_path_labels_invalid(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    handle_invalid(tmp_path / 'data.tif', dry_run=True, label='Invalid')
    assert 'Invalid' in capsys.readouterr().out


def test_handle_invalid_non_dry_run_prints_deleting(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / 'data.tif'
    path.touch()
    handle_invalid(path, dry_run=False, label='Hash missing')
    assert 'Deleting' in capsys.readouterr().out


def test_handle_invalid_deletes_file(tmp_path: Path) -> None:
    path = tmp_path / 'data.tif'
    path.touch()
    handle_invalid(path, dry_run=False, label='Hash missing')
    assert not path.exists()


def test_handle_invalid_deletes_directory(tmp_path: Path) -> None:
    directory = tmp_path / 'archive.zarr'
    directory.mkdir()
    (directory / 'zarr.json').touch()
    handle_invalid(directory, dry_run=False, label='Hash missing')
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
