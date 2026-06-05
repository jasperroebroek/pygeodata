import json
from collections.abc import Generator
from unittest.mock import patch

import pytest

from pygeodata.artifact import Artifact
from pygeodata.cache import clean_cache, purge_unregistered_cache
from pygeodata.config import JSONKeys, set_config
from pygeodata.paths import CachePathResolver
from pygeodata.types import SpatialSpec
from tests.fixtures.figures import SimpleFigure


@pytest.fixture(
    params=[
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (True, False, True),
        (False, False, True),
        (True, True, True),
        (False, True, True),
    ],
    autouse=True,
    ids=[
        'punct-on,human-off,flat-off',
        'punct-off,human-off,flat-off',
        'punct-on,human-on,flat-off',
        'punct-off,human-on,flat-off',
        'punct-on,human-off,flat-on',
        'punct-off,human-off,flat-on',
        'punct-on,human-on,flat-on',
        'punct-off,human-on,flat-on',
    ],
)
def all_figure_layouts(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    punct, human, flat = request.param
    with set_config(filesystem_allows_punctuation=punct, human_readable_paths=human, flatten_figures=flat):
        yield


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


def test_clean_cache_stale_figure_reported(
    sample_spatial_spec: SpatialSpec,
    capsys: pytest.CaptureFixture,
) -> None:
    process_touch(SimpleFigure(), hash=True, spec=sample_spatial_spec, stale=True)
    clean_cache(dry_run=True)
    captured = capsys.readouterr().out
    assert 'Hash wrong' in captured or 'Hash missing' in captured


def test_purge_unregistered_cache_figure_with_params_skipped(
    sample_spatial_spec: SpatialSpec,
) -> None:
    """Figure files must not be deleted by purge_unregistered_cache regardless of naming mode."""
    process_touch(SimpleFigure(), hash=True, spec=sample_spatial_spec)
    file = SimpleFigure().get_processed_path(spec=sample_spatial_spec)
    with patch('builtins.input', return_value='y'):
        purge_unregistered_cache()
    assert file.exists()
