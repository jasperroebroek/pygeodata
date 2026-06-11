import pytest

from pygeodata import load
from pygeodata.config import set_config
from pygeodata.data import Data
from pygeodata.spec import SpatialSpec
from tests.fixtures.data import (
    LoaderA,
    SimpleLoader,
)


def test_load_returns_data(simple_data_loader: Data, sample_spatial_spec: SpatialSpec) -> None:
    data = load(simple_data_loader, spec=sample_spatial_spec)
    assert data.rio.crs == sample_spatial_spec.crs


def test_get_cache_root_matches_config(tmp_path: pytest.TempdirFactory) -> None:
    with set_config(path_cache=tmp_path):
        assert SimpleLoader.get_cache_root() == tmp_path


def test_processed_dir_with_params_differs_by_param(sample_spatial_spec: SpatialSpec) -> None:
    p1 = LoaderA(year=2000).get_processed_dir(sample_spatial_spec)
    p2 = LoaderA(year=2001).get_processed_dir(sample_spatial_spec)
    assert p1 != p2


def test_processed_path_contains_class_and_hash(
    sample_spatial_spec: SpatialSpec,
) -> None:
    path = SimpleLoader().get_processed_path(sample_spatial_spec)
    hash_part = path.parent.name
    assert len(hash_part) == 64
    assert all(c in '0123456789abcdef' for c in hash_part)
