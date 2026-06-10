import pytest

from pygeodata.cache import clean_cache
from pygeodata.config import JSONKeys
from pygeodata.types import SpatialSpec
from tests.fixtures.figures import SimpleFigure


def process_touch(artifact: SimpleFigure, spec: SpatialSpec, stale: bool = False) -> None:
    path = artifact.get_processed_path(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if stale:
        hash_file = artifact.resolve_cache_paths(spec).state_hash_path
        hash_file.write_text('{"' + JSONKeys.DEPENDENCY_TREE_HASH + '": "stale"}')
    else:
        artifact.write_cache_metadata(spec)


def test_clean_cache_stale_figure_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    process_touch(SimpleFigure(), sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    out = capsys.readouterr().out
    assert 'Hash wrong' in out or 'Format version mismatch' in out


def test_clean_cache_valid_figure_untouched(sample_spatial_spec: SpatialSpec) -> None:
    process_touch(SimpleFigure(), sample_spatial_spec)
    path = SimpleFigure().get_processed_path(spec=sample_spatial_spec)
    clean_cache(dry_run=False)
    assert path.exists()
