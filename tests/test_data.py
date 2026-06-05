from collections.abc import Generator

import pytest

from pygeodata import load
from pygeodata.config import get_config, set_config
from pygeodata.data import Data
from pygeodata.types import SpatialSpec
from tests.fixtures.data import (
    DummyScenario,
    EnumLoader,
    LoaderA,
    LoaderC,
    SimpleLoader,
)


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


def test_load_returns_data(simple_data_loader: Data, sample_spatial_spec: SpatialSpec) -> None:
    data = load(simple_data_loader, spec=sample_spatial_spec)
    assert data.rio.crs == sample_spatial_spec.crs


def test_cls_cache_pattern_hash_based() -> None:
    with set_config(human_readable_paths=False):
        assert SimpleLoader.get_cls_cache_pattern() == 'SimpleLoader/*'


def test_cls_cache_pattern_human_readable() -> None:
    with set_config(human_readable_paths=True):
        assert SimpleLoader.get_cls_cache_pattern() == '*/*/' + SimpleLoader.get_class_name()


def test_matches_cache_path_true(sample_spatial_spec: SpatialSpec) -> None:
    # matches_cache_path checks path.name == class name, so pass the dir named after the class
    processed_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    cls_dir = processed_dir if get_config().human_readable_paths else processed_dir.parent
    assert SimpleLoader.matches_cache_path(cls_dir)


def test_matches_cache_path_false(sample_spatial_spec: SpatialSpec) -> None:
    processed_dir = SimpleLoader().get_processed_dir(sample_spatial_spec)
    cls_dir = processed_dir if get_config().human_readable_paths else processed_dir.parent
    assert not LoaderA.matches_cache_path(cls_dir)


def test_get_cache_root_matches_config(tmp_path: pytest.TempdirFactory) -> None:
    with set_config(path_cache=tmp_path):
        assert SimpleLoader.get_cache_root() == tmp_path


def test_processed_dir_with_params_differs_by_param(sample_spatial_spec: SpatialSpec) -> None:
    p1 = LoaderA(year=2000).get_processed_dir(sample_spatial_spec)
    p2 = LoaderA(year=2001).get_processed_dir(sample_spatial_spec)
    assert p1 != p2


def test_processed_dir_exclude_params_from_path(sample_spatial_spec: SpatialSpec) -> None:
    with set_config(human_readable_paths=False):
        p_included = LoaderC(target='forest', n_jobs=1).get_processed_dir(sample_spatial_spec)
        p_excluded = LoaderC(target='forest', n_jobs=99).get_processed_dir(sample_spatial_spec)
    assert p_included == p_excluded


def test_processed_path_human_readable_contains_crs_and_geo(
    sample_spatial_spec: SpatialSpec,
) -> None:
    with set_config(human_readable_paths=True):
        path = SimpleLoader().get_processed_path(sample_spatial_spec)
    parts = path.parts
    assert 'EPSG_4326' in parts
    assert 'SimpleLoader' in parts


def test_processed_path_hash_based_contains_class_and_hash(
    sample_spatial_spec: SpatialSpec,
) -> None:
    with set_config(human_readable_paths=False):
        path = SimpleLoader().get_processed_path(sample_spatial_spec)
    parts = path.parts
    assert 'SimpleLoader' in parts
    hash_part = parts[parts.index('SimpleLoader') + 1]
    assert len(hash_part) == 64
    assert all(c in '0123456789abcdef' for c in hash_part)


def test_formatting_data_with_enum() -> None:
    loader = EnumLoader(DummyScenario.SSP126)
    formatted_enum = get_config().format_path_fn(loader.scenario)
    expected_enum = 'DummyScenario[SSP126]' if get_config().filesystem_allows_punctuation else 'DummyScenario--SSP126--'
    assert formatted_enum == expected_enum


def test_formatting_data_with_enum_human_readable(sample_spatial_spec: SpatialSpec) -> None:
    loader = EnumLoader(DummyScenario.SSP126)
    with set_config(human_readable_paths=True):
        formatted_enum = get_config().format_path_fn(loader.scenario)
        assert formatted_enum in str(loader.get_processed_path(sample_spatial_spec))
