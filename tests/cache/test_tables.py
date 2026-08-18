import pytest

from pygeodata.cache import clean_cache
from pygeodata.config import JSONKeys
from pygeodata.spec import SpatialSpec
from tests.fixtures.table import SimpleTable


def process_touch(artifact: SimpleTable, spec: SpatialSpec, stale: bool = False) -> None:
    path = artifact.get_processed_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if stale:
        hash_file = artifact.resolve_cache_paths(spec).state_hash_path
        hash_file.write_text('{"' + JSONKeys.DEPENDENCY_TREE_HASH + '": "stale"}')
    else:
        artifact.write_cache_metadata(spec)


def test_clean_cache_stale_table_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    process_touch(SimpleTable(), sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash wrong' in out or 'Format version mismatch' in out


def test_clean_cache_valid_table_untouched(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleTable(), sample_spatial_spec)
    path = SimpleTable().get_processed_path(spec=sample_spatial_spec)
    clean_cache(dry_run=False)
    assert path.exists()
