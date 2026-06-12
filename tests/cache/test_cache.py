import json
from pathlib import Path

import pytest

from pygeodata.cache import (
    clean_cache,
    clean_registry,
    format_version_matches,
    hash_matches_live,
    read_cache_class_name,
    rebuild_registry,
)
from pygeodata.config import FORMAT_VERSION, JSONKeys
from pygeodata.spec import SpatialSpec
from tests.fixtures.data import Child, Parent, SimpleLoader


def write_stale_hash(artifact: SimpleLoader, spec: SpatialSpec) -> None:
    hash_path = artifact.resolve_cache_paths(spec).state_hash_path
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: artifact.get_class_name(),
                JSONKeys.DEPENDENCY_TREE_HASH: 'stale',
                JSONKeys.STATE_HASH: 'stale',
            },
        ),
    )


def process_touch(artifact: SimpleLoader, spec: SpatialSpec, stale: bool = False) -> None:
    path = artifact.get_processed_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if stale:
        write_stale_hash(artifact, spec)
    else:
        artifact.write_cache_metadata(spec)


def make_zarr_archive(parent: Path, stem: str, hash_value: str | None = None) -> Path:
    zarr_dir = parent / f'{stem}.zarr'
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / 'zarr.json').touch()
    (zarr_dir / '0' / '0').mkdir(parents=True)
    if hash_value is not None:
        (parent / f'.{stem}.hash.json').write_text(
            json.dumps(
                {
                    JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
                    JSONKeys.CLASS_NAME: 'SimpleLoader',
                    JSONKeys.DEPENDENCY_TREE_HASH: hash_value,
                },
            ),
        )
    return zarr_dir


# --- clean_cache: regular files ---


def test_clean_cache_stale_hash_reported(sample_spatial_spec: SpatialSpec, capsys: pytest.CaptureFixture) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash wrong' in out or 'Format version mismatch' in out


def test_clean_cache_stale_hash_dry_run_keeps_file(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    path = SimpleLoader().get_processed_path(sample_spatial_spec)
    clean_cache(dry_run=True)
    assert path.exists()


def test_clean_cache_stale_hash_deletes_dir(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    clean_cache(dry_run=False)
    assert not entry_dir.exists()


def test_clean_cache_missing_hash_reported(sample_spatial_spec: SpatialSpec, capsys: pytest.CaptureFixture) -> None:
    path = SimpleLoader().get_processed_path(sample_spatial_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    clean_cache(dry_run=True)
    assert 'Hash missing' in capsys.readouterr().out


def test_clean_cache_valid_entry_untouched(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec)
    path = SimpleLoader().get_processed_path(sample_spatial_spec)
    clean_cache(dry_run=False)
    assert path.exists()


def test_clean_cache_removes_empty_dirs(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    clean_cache(dry_run=False)
    assert not entry_dir.exists()


# --- clean_cache: zarr ---


def test_clean_cache_zarr_valid_untouched(sample_spatial_spec: SpatialSpec) -> None:
    correct_hash = SimpleLoader.get_dependency_tree_hash()
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    entry_dir.mkdir(parents=True, exist_ok=True)
    zarr = make_zarr_archive(entry_dir, 'simple_loader', hash_value=correct_hash)
    clean_cache(dry_run=False)
    assert zarr.exists()


def test_clean_cache_zarr_stale_hash_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    entry_dir.mkdir(parents=True, exist_ok=True)
    make_zarr_archive(entry_dir, 'simple_loader', hash_value='stale')
    clean_cache(dry_run=True)
    assert 'Hash wrong' in capsys.readouterr().out


def test_clean_cache_zarr_stale_hash_deletes(sample_spatial_spec: SpatialSpec) -> None:
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    entry_dir.mkdir(parents=True, exist_ok=True)
    zarr = make_zarr_archive(entry_dir, 'simple_loader', hash_value='stale')
    clean_cache(dry_run=False)
    assert not zarr.exists()


def test_clean_cache_zarr_internals_not_visited(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    correct_hash = SimpleLoader.get_dependency_tree_hash()
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    entry_dir.mkdir(parents=True, exist_ok=True)
    zarr = make_zarr_archive(entry_dir, 'simple_loader', hash_value=correct_hash)
    chunk = zarr / '0' / '0' / 'chunk.bin'
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.touch()
    clean_cache(dry_run=True)
    assert str(chunk) not in capsys.readouterr().out


def test_clean_cache_zarr_missing_hash_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    entry_dir.mkdir(parents=True, exist_ok=True)
    make_zarr_archive(entry_dir, 'simple_loader')
    clean_cache(dry_run=True)
    assert 'Hash missing' in capsys.readouterr().out


# --- read_cache_class_name ---


def test_read_cache_class_name_returns_name(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec)
    hash_path = loader.resolve_cache_paths(sample_spatial_spec).state_hash_path
    assert read_cache_class_name(hash_path) == 'SimpleLoader'


def test_read_cache_class_name_none_when_missing(tmp_path: Path) -> None:
    assert read_cache_class_name(tmp_path / '.data.hash.json') is None


def test_read_cache_class_name_none_when_key_absent(tmp_path: Path) -> None:
    hash_file = tmp_path / '.data.hash.json'
    hash_file.write_text(json.dumps({'other_key': 'value'}))
    assert read_cache_class_name(hash_file) is None


# --- hash_matches_live ---


def test_hash_matches_live_true(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec)
    hash_path = loader.resolve_cache_paths(sample_spatial_spec).state_hash_path
    assert hash_matches_live(hash_path) is True


def test_hash_matches_live_false_when_stale(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    write_stale_hash(loader, sample_spatial_spec)
    hash_path = loader.resolve_cache_paths(sample_spatial_spec).state_hash_path
    assert hash_matches_live(hash_path) is False


def test_hash_matches_live_false_when_missing(tmp_path: Path) -> None:
    assert hash_matches_live(tmp_path / '.data.hash.json') is False


def test_hash_matches_live_none_when_class_unregistered(tmp_path: Path) -> None:
    hash_file = tmp_path / '.data.hash.json'
    hash_file.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: 'NoSuchClass',
                JSONKeys.DEPENDENCY_TREE_HASH: 'abc',
            },
        ),
    )
    assert hash_matches_live(hash_file) is None


# --- format_version_matches ---


def test_format_version_matches_true(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec)
    hash_path = loader.resolve_cache_paths(sample_spatial_spec).state_hash_path
    assert format_version_matches(hash_path) is True


def test_format_version_matches_false_when_missing_key(tmp_path: Path) -> None:
    hash_file = tmp_path / '.data.hash.json'
    hash_file.write_text(json.dumps({JSONKeys.CLASS_NAME: 'SimpleLoader'}))
    assert format_version_matches(hash_file) is False


def test_format_version_matches_false_when_wrong_version(tmp_path: Path) -> None:
    hash_file = tmp_path / '.data.hash.json'
    hash_file.write_text(json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION + 1}))
    assert format_version_matches(hash_file) is False


def test_format_version_matches_false_when_file_missing(tmp_path: Path) -> None:
    assert format_version_matches(tmp_path / '.data.hash.json') is False


def test_clean_cache_format_version_mismatch_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    assert 'Format version mismatch' in capsys.readouterr().out


def test_clean_cache_format_version_mismatch_deletes(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    entry_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    clean_cache(dry_run=False)
    assert not entry_dir.exists()


# --- clean_registry ---


def test_clean_registry_removes_stale_code_entry(tmp_path: Path) -> None:
    from pygeodata.config import set_config

    with set_config(path_registry=tmp_path / '.source'):
        code_dir = tmp_path / '.source' / 'code' / 'oldhash'
        code_dir.mkdir(parents=True)
        (code_dir / 'source.py').write_text('class Foo: pass')
        (code_dir / 'source.json').write_text(json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION + 1}))
        clean_registry(dry_run=False)
        assert not code_dir.exists()


def test_clean_registry_keeps_current_code_entry(tmp_path: Path) -> None:
    from pygeodata.config import set_config

    with set_config(path_registry=tmp_path / '.source'):
        code_dir = tmp_path / '.source' / 'code' / 'currenthash'
        code_dir.mkdir(parents=True)
        (code_dir / 'source.py').write_text('class Foo: pass')
        (code_dir / 'source.json').write_text(json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION}))
        clean_registry(dry_run=False)
        assert code_dir.exists()


def test_clean_registry_removes_stale_snapshot_entry(tmp_path: Path) -> None:
    from pygeodata.config import set_config

    with set_config(path_registry=tmp_path / '.source'):
        snapshot_dir = tmp_path / '.source' / 'snapshots' / 'oldhash'
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / 'tree.json').write_text(json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION + 1}))
        clean_registry(dry_run=False)
        assert not snapshot_dir.exists()


def test_clean_registry_missing_format_version_treated_as_stale(tmp_path: Path) -> None:
    from pygeodata.config import set_config

    with set_config(path_registry=tmp_path / '.source'):
        code_dir = tmp_path / '.source' / 'code' / 'oldhash'
        code_dir.mkdir(parents=True)
        (code_dir / 'source.json').write_text(json.dumps({JSONKeys.CLASS_NAME: 'Foo'}))
        clean_registry(dry_run=False)
        assert not code_dir.exists()


def test_clean_registry_dry_run_keeps_stale(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from pygeodata.config import set_config

    with set_config(path_registry=tmp_path / '.source'):
        code_dir = tmp_path / '.source' / 'code' / 'oldhash'
        code_dir.mkdir(parents=True)
        (code_dir / 'source.json').write_text(json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION + 1}))
        clean_registry(dry_run=True)
        assert code_dir.exists()
        assert 'Format version mismatch' in capsys.readouterr().out


# --- rebuild_registry ---


def test_rebuild_registry_writes_registry_for_all_tracked_objects() -> None:
    rebuild_registry()
    assert SimpleLoader.is_registry_valid()
    assert Child.is_registry_valid()
    assert Parent.is_registry_valid()


def test_rebuild_registry_clears_and_rewrites() -> None:
    rebuild_registry()
    rebuild_registry()
    assert SimpleLoader.is_registry_valid()
