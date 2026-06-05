import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.artifact import Artifact
from pygeodata.cache import (
    clean_cache,
    path_matches_hash,
    purge_unregistered_cache,
    read_cache_class_name,
    rebuild_registry,
)
from pygeodata.config import JSONKeys, set_config
from pygeodata.paths import CachePathResolver
from pygeodata.tracked_object import TrackedObject
from pygeodata.types import SpatialSpec
from tests.fixtures.data import Child, EmptyLoader, Parent, SimpleLoader


@pytest.fixture(
    params=[
        (True, False),
        (False, False),
        (True, True),
        (False, True),
    ],
    autouse=True,
    ids=['punct-on,human-off', 'punct-off,human-off', 'punct-on,human-on', 'punct-off,human-on'],
)
def all_path_layouts(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    punct, human = request.param
    with set_config(filesystem_allows_punctuation=punct, human_readable_paths=human):
        yield


def write_hash(path: Path, hash_value: str) -> None:
    path.write_text(json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: hash_value}))


def process_touch(artifact: Artifact, hash: bool, spec: SpatialSpec, stale: bool = False) -> None:
    path = artifact.get_processed_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    if not hash:
        return

    if stale:
        hash_file = CachePathResolver.from_path(path).state_hash_path
        hash_file.write_text(json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: 'stale'}))
    else:
        artifact.write_cache_metadata(spec)


def make_zarr_archive(parent: Path, name: str, hash_value: str | None = None) -> Path:
    zarr_dir = parent / f'{name}.zarr'
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / 'zarr.json').touch()
    (zarr_dir / '0' / '0').mkdir(parents=True)
    if hash_value is not None:
        write_hash(parent / f'.{name}.hash.json', hash_value)
    return zarr_dir


# --- purge_cls_cache ---

def test_purge_cache_invalid_flags_wrong_hash(sample_spatial_spec: SpatialSpec, capsys: pytest.CaptureFixture) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(spec=sample_spatial_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    stale_hash = CachePathResolver.from_path(path).state_hash_path
    stale_hash.write_text(json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: 'stale'}))

    loader.purge_cls_cache(dry_run=True)
    captured = capsys.readouterr().out
    assert 'Hash wrong' in captured
    assert str(path) in captured
    assert path.exists()

    loader.purge_cls_cache(dry_run=False)
    assert not path.exists()


def test_purge_artifact_cache_valid_hash_untouched(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), hash=True, spec=sample_spatial_spec)
    SimpleLoader.purge_cls_cache(dry_run=False)
    assert SimpleLoader().get_processed_path(spec=sample_spatial_spec).exists()


def test_purge_artifact_cache_missing_hash_dry_run(sample_spatial_spec: SpatialSpec, capsys: pytest.CaptureFixture) -> None:
    process_touch(SimpleLoader(), hash=False, spec=sample_spatial_spec)
    path = SimpleLoader().get_processed_path(spec=sample_spatial_spec)
    SimpleLoader.purge_cls_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash missing' in out
    assert str(path) in out
    assert path.exists()


def test_purge_artifact_cache_removes_empty_dirs() -> None:
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    process_touch(SimpleLoader(), hash=False, spec=SpatialSpec())
    SimpleLoader.purge_cls_cache(dry_run=False)
    assert not path.exists()
    assert not out_dir.exists()


def test_purge_artifact_cache_dry_run_keeps_empty_dirs() -> None:
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    process_touch(SimpleLoader(), hash=False, spec=SpatialSpec(), stale=True)
    SimpleLoader.purge_cls_cache(dry_run=True)
    assert out_dir.exists()


# --- purge_cls_cache zarr ---

def test_purge_artifact_cache_zarr_valid_untouched() -> None:
    correct_hash = SimpleLoader.get_dependency_tree_hash()
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    zarr = make_zarr_archive(out_dir, SimpleLoader.get_class_name(), hash_value=correct_hash)
    SimpleLoader.purge_cls_cache(dry_run=False)
    assert zarr.exists()


def test_purge_artifact_cache_zarr_wrong_hash_dry_run(capsys: pytest.CaptureFixture) -> None:
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    zarr = make_zarr_archive(out_dir, SimpleLoader.get_class_name(), hash_value='stale')
    SimpleLoader.purge_cls_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash wrong' in out
    assert str(zarr) in out
    assert zarr.exists()


def test_purge_artifact_cache_zarr_wrong_hash_deletes() -> None:
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    zarr = make_zarr_archive(out_dir, SimpleLoader.get_class_name(), hash_value='stale')
    SimpleLoader.purge_cls_cache(dry_run=False)
    assert not zarr.exists()


def test_purge_artifact_cache_zarr_internals_not_visited(capsys: pytest.CaptureFixture) -> None:
    """Chunks and sub-dirs inside a valid zarr should never be reported."""
    correct_hash = SimpleLoader.get_dependency_tree_hash()
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    zarr = make_zarr_archive(out_dir, SimpleLoader.get_class_name(), hash_value=correct_hash)
    chunk = zarr / '0' / '0' / 'chunk.bin'
    chunk.parent.mkdir(parents=True, exist_ok=True)
    chunk.touch()
    SimpleLoader.purge_cls_cache(dry_run=True)
    out = capsys.readouterr().out
    assert str(chunk) not in out


def test_purge_artifact_cache_zarr_missing_hash_dry_run(capsys: pytest.CaptureFixture) -> None:
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent
    zarr = make_zarr_archive(out_dir, SimpleLoader.get_class_name())
    SimpleLoader.purge_cls_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash missing' in out
    assert str(zarr) in out


# --- purge_unregistered_cache ---

def test_purge_unregistered_cache_known_artifact_skipped(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(spec=sample_spatial_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    hash_path = loader.get_state_hash_path(spec=sample_spatial_spec)
    hash_path.write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: loader.get_class_name(),
                JSONKeys.STATE_HASH: loader.get_state_hash(sample_spatial_spec),
                JSONKeys.DEPENDENCY_TREE_HASH: loader.get_dependency_tree_hash(),
            },
        ),
    )

    with patch('builtins.input', return_value='y'):
        purge_unregistered_cache()

    assert path.exists()


def test_purge_unregistered_cache_unknown_prompts_and_deletes() -> None:
    path = SimpleLoader().get_processed_path(spec=SpatialSpec())
    out_dir = path.parent.with_name('UnknownLoader')
    out_dir.mkdir(parents=True, exist_ok=True)
    unknown_file = out_dir / 'UnknownLoader.tif'
    unknown_file.touch()
    with patch('builtins.input', return_value='y'):
        purge_unregistered_cache(dry_run=False)
    assert not unknown_file.exists()


def test_purge_unregistered_cache_unknown_skipped_on_no(capsys: pytest.CaptureFixture) -> None:
    base = EmptyLoader.get_cache_root()
    out_dir = base / 'UnknownLoader' / 'run1'
    out_dir.mkdir(parents=True, exist_ok=True)
    unknown_file = out_dir / 'UnknownLoader.tif'
    unknown_file.touch()
    with patch('builtins.input', return_value='n'):
        purge_unregistered_cache(dry_run=False)
    assert unknown_file.exists()
    assert 'Skipping' in capsys.readouterr().out


def test_purge_unregistered_cache_dry_run_prints_unknown(capsys: pytest.CaptureFixture) -> None:
    base = EmptyLoader.get_cache_root()
    out_dir = base / 'UnknownLoader' / 'run1'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'unknown.tif').touch()
    purge_unregistered_cache(dry_run=True)
    assert 'UnknownLoader' in capsys.readouterr().out


# --- clean_cache ---

def test_clean_cache_stale_loader_reported(sample_spatial_spec: SpatialSpec, capsys: pytest.CaptureFixture) -> None:
    process_touch(SimpleLoader(), hash=True, spec=sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    captured = capsys.readouterr().out
    assert 'Hash wrong' in captured
    assert 'simple_loader.tif' in captured


def test_clean_cache_uses_dependency_graph_nodes(sample_spatial_spec: SpatialSpec, capsys: pytest.CaptureFixture) -> None:
    nodes: set[type[Artifact]] = Child.get_all_dependencies()

    created = []
    for cls in nodes:
        if cls.object_type == TrackedObject:
            continue
        path = cls().get_processed_path(spec=sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        created.append(path)

    assert created, 'No non-TrackedObject nodes found in Child dependency graph'

    clean_cache(loader=Child, dry_run=True)
    captured = capsys.readouterr().out
    assert 'Hash missing' in captured
    for data_file in created:
        assert data_file.name in captured


def test_clean_cache_purges_unregistered_on_full_non_dry_run(sample_spatial_spec: SpatialSpec) -> None:
    """purge_unregistered_cache is called and actually deletes unknown artifacts."""
    path = SimpleLoader().get_processed_path(spec=sample_spatial_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

    with patch('builtins.input', return_value='y'):
        clean_cache(loader=None, dry_run=False)

    assert not path.exists()


def test_clean_cache_all_artifacts(capsys: pytest.CaptureFixture) -> None:
    for cls in (SimpleLoader, Child):
        process_touch(cls(), hash=True, spec=SpatialSpec(), stale=True)
    clean_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash wrong' in out
    assert 'simple_loader.tif' in out
    assert 'child.tif' in out


def test_clean_cache_respects_dependency_graph(capsys: pytest.CaptureFixture) -> None:
    nodes = Child.get_all_dependencies()
    node_names = {cls.get_class_name() for cls in nodes}
    assert 'Parent' in node_names, f'Parent not in dependency graph: {node_names}'

    for cls in nodes:
        if cls.object_type == TrackedObject:
            continue
        process_touch(cls(), hash=True, spec=SpatialSpec(), stale=True)

    clean_cache(loader=Child, dry_run=True)
    out = capsys.readouterr().out

    for cls in nodes:
        if cls.object_type == TrackedObject:
            continue
        assert cls().get_filename('tif') in out


def test_clean_cache_skips_purge_unregistered_on_dry_run() -> None:
    with patch('pygeodata.cache.purge_unregistered_cache') as mock_unregistered:
        clean_cache(dry_run=True)
        mock_unregistered.assert_not_called()


def test_clean_cache_skips_purge_unregistered_for_specific_loader() -> None:
    """purge_unregistered_cache should only run on full registry sweeps."""
    with patch('pygeodata.cache.purge_unregistered_cache') as mock_unregistered:
        clean_cache(loader=Parent, dry_run=False)
        mock_unregistered.assert_not_called()


# --- read_cache_class_name ---

def test_read_cache_class_name_returns_name(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec)
    path = loader.get_processed_path(sample_spatial_spec)
    assert read_cache_class_name(path) == 'SimpleLoader'


def test_read_cache_class_name_none_when_no_hash_file(sample_spatial_spec: SpatialSpec) -> None:
    path = SimpleLoader().get_processed_path(sample_spatial_spec)
    assert read_cache_class_name(path) is None


def test_read_cache_class_name_none_when_key_absent(tmp_path: Path) -> None:
    data_file = tmp_path / 'data.tif'
    hash_file = tmp_path / '.data.hash.json'
    hash_file.write_text(json.dumps({'other_key': 'value'}))
    assert read_cache_class_name(data_file) is None


# --- rebuild_registry ---

def test_rebuild_registry_writes_registry_for_all_tracked_objects() -> None:
    rebuild_registry()
    assert SimpleLoader.is_registry_valid()
    assert Child.is_registry_valid()
    assert Parent.is_registry_valid()
