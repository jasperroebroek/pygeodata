import json
from pathlib import Path

import pytest

from pygeodata.cache import (
    clean_cache,
    clean_registry,
    clean_source_registry,
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
    from pygeodata.paths import CachePathConstructor

    zarr_dir = parent / f'{stem}.zarr'
    zarr_dir.mkdir(parents=True, exist_ok=True)
    (zarr_dir / 'zarr.json').touch()
    (zarr_dir / '0' / '0').mkdir(parents=True)
    if hash_value is not None:
        CachePathConstructor(parent).state_hash_path.write_text(
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
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
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
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
    clean_cache(dry_run=False)
    assert not entry_dir.exists()


# --- clean_cache: zarr ---


def test_clean_cache_zarr_valid_untouched(sample_spatial_spec: SpatialSpec) -> None:
    correct_hash = SimpleLoader.get_dependency_tree_hash()
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
    entry_dir.mkdir(parents=True, exist_ok=True)
    zarr = make_zarr_archive(entry_dir, 'simple_loader', hash_value=correct_hash)
    clean_cache(dry_run=False)
    assert zarr.exists()


def test_clean_cache_zarr_stale_hash_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
    entry_dir.mkdir(parents=True, exist_ok=True)
    make_zarr_archive(entry_dir, 'simple_loader', hash_value='stale')
    clean_cache(dry_run=True)
    assert 'Hash wrong' in capsys.readouterr().out


def test_clean_cache_zarr_stale_hash_deletes(sample_spatial_spec: SpatialSpec) -> None:
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
    entry_dir.mkdir(parents=True, exist_ok=True)
    zarr = make_zarr_archive(entry_dir, 'simple_loader', hash_value='stale')
    clean_cache(dry_run=False)
    assert not zarr.exists()


def test_clean_cache_zarr_internals_not_visited(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    correct_hash = SimpleLoader.get_dependency_tree_hash()
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
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
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
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
    assert read_cache_class_name(tmp_path / 'meta.json') is None


def test_read_cache_class_name_none_when_key_absent(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
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
    assert hash_matches_live(tmp_path / 'meta.json') is False


def test_hash_matches_live_none_when_class_unregistered(tmp_path: Path) -> None:
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


# --- format_version_matches ---


def test_format_version_matches_true(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec)
    hash_path = loader.resolve_cache_paths(sample_spatial_spec).state_hash_path
    assert format_version_matches(hash_path) is True


def test_format_version_matches_false_when_missing_key(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
    hash_file.write_text(json.dumps({JSONKeys.CLASS_NAME: 'SimpleLoader'}))
    assert format_version_matches(hash_file) is False


def test_format_version_matches_false_when_wrong_version(tmp_path: Path) -> None:
    hash_file = tmp_path / 'meta.json'
    hash_file.write_text(json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION + 1}))
    assert format_version_matches(hash_file) is False


def test_format_version_matches_false_when_file_missing(tmp_path: Path) -> None:
    assert format_version_matches(tmp_path / 'meta.json') is False


def test_clean_cache_format_version_mismatch_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    assert 'Format version mismatch' in capsys.readouterr().out


def test_clean_cache_format_version_mismatch_deletes(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleLoader(), sample_spatial_spec, stale=True)
    entry_dir = SimpleLoader().resolve_cache_paths(sample_spatial_spec).directory
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


# ===========================================================================
# clean_source_registry (Unit 9b)
# ===========================================================================


def _write_code(registry: Path, source_hash: str, class_name: str, mtime: str) -> None:
    d = registry / 'code' / source_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / 'source.py').write_text(f'class {class_name}: pass\n', encoding='utf-8')
    (d / 'source.json').write_text(
        json.dumps({
            JSONKeys.CLASS_NAME: class_name,
            JSONKeys.OBJECT_TYPE: 'Data',
            JSONKeys.SOURCE_HASH: source_hash,
            JSONKeys.REGISTERED_AT: mtime,
        }),
        encoding='utf-8',
    )


def _write_snap(registry: Path, dep_hash: str, nodes: dict) -> None:
    d = registry / 'snapshots' / dep_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / 'tree.json').write_text(
        json.dumps({JSONKeys.NODES: nodes, JSONKeys.TREE: {}}),
        encoding='utf-8',
    )


def _write_entry(cache_dir: Path, state_hash: str, dep_tree_hash: str, class_name: str) -> None:
    """Write a minimal meta.json that EntryRegistry can scan."""
    from pygeodata.paths import CachePathConstructor

    entry_dir = cache_dir / class_name / state_hash
    entry_dir.mkdir(parents=True, exist_ok=True)
    CachePathConstructor(entry_dir).state_hash_path.write_text(
        json.dumps({
            JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
            JSONKeys.CLASS_NAME: class_name,
            JSONKeys.STATE_HASH: state_hash,
            JSONKeys.DEPENDENCY_TREE_HASH: dep_tree_hash,
            JSONKeys.INSTANCE_HASH: 'inst1',
        })
    )
    CachePathConstructor(entry_dir).params_path.write_text(json.dumps({}))
    CachePathConstructor(entry_dir).spec_path.write_text(json.dumps({}))


class _CleanSourceSetup:
    """Shared fixture: one referenced snapshot + one orphan each for code and snapshots."""

    def __init__(self, tmp_path: Path):
        from pygeodata.config import set_config as _sc

        self.tmp = tmp_path
        self.registry = tmp_path / '.source'
        self.cache = tmp_path / 'cache'
        self._sc = _sc

        # Two code snapshots for MyLoader: h1 (initial) and h2 (latest/changed)
        _write_code(self.registry, 'h1', 'MyLoader', '2026-01-01T00:00:00+00:00')
        _write_code(self.registry, 'h2', 'MyLoader', '2026-06-01T00:00:00+00:00')
        # An orphan snapshot for OtherClass that has no entry and is not latest
        _write_code(self.registry, 'old_orphan', 'OtherClass', '2025-01-01T00:00:00+00:00')
        # Latest for OtherClass (no entry — unrun class, must be kept)
        _write_code(self.registry, 'other_latest', 'OtherClass', '2026-03-01T00:00:00+00:00')

        # One tree snapshot referenced by an entry (contains h2)
        _write_snap(self.registry, 'snap_ref', {'MyLoader': {'hash': 'h2'}})
        # An orphan tree snapshot (no entry points to it)
        _write_snap(self.registry, 'snap_orphan', {'MyLoader': {'hash': 'h1'}})

        # One live entry that references snap_ref (and through it, h2)
        _write_entry(self.cache, 'state_abc', 'snap_ref', 'MyLoader')

    def context(self):
        return self._sc(
            path_cache=self.cache,
            path_figures=self.tmp / 'figures',
            path_registry=self.registry,
        )


@pytest.fixture
def src_setup(tmp_path: Path):
    return _CleanSourceSetup(tmp_path)


def test_clean_source_dry_run_deletes_nothing(src_setup) -> None:
    """Dry run must not delete any files."""
    s = src_setup
    with s.context():
        clean_source_registry(dry_run=True)

    # All dirs still present
    assert (s.registry / 'code' / 'old_orphan').exists()
    assert (s.registry / 'snapshots' / 'snap_orphan').exists()


def test_clean_source_prunes_orphan_code(src_setup) -> None:
    """Unreferenced, non-latest code snapshot is deleted."""
    s = src_setup
    with s.context():
        clean_source_registry(dry_run=False)

    assert not (s.registry / 'code' / 'old_orphan').exists(), 'orphan code should be pruned'


def test_clean_source_keeps_referenced_code(src_setup) -> None:
    """Code snapshot referenced by a live entry is kept."""
    s = src_setup
    with s.context():
        clean_source_registry(dry_run=False)

    assert (s.registry / 'code' / 'h2').exists(), 'referenced code h2 must be kept'


def test_clean_source_keeps_latest_for_unrun_class(src_setup) -> None:
    """Latest snapshot for a class with no entry is kept (unrun class protection)."""
    s = src_setup
    with s.context():
        clean_source_registry(dry_run=False)

    assert (s.registry / 'code' / 'other_latest').exists(), 'latest-for-class must be kept'


def test_clean_source_prunes_orphan_tree(src_setup) -> None:
    """Tree snapshot not referenced by any entry is deleted."""
    s = src_setup
    with s.context():
        clean_source_registry(dry_run=False)

    assert not (s.registry / 'snapshots' / 'snap_orphan').exists(), 'orphan tree should be pruned'


def test_clean_source_keeps_referenced_tree(src_setup) -> None:
    """Tree snapshot referenced by a live entry is kept."""
    s = src_setup
    with s.context():
        clean_source_registry(dry_run=False)

    assert (s.registry / 'snapshots' / 'snap_ref').exists(), 'referenced tree must be kept'
