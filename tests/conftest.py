import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS

from pygeodata.config import JSONKeys, set_config
from pygeodata.types import SpatialSpec
from tests.fixtures.data import MultiOutputLoader, NestedLoader, SampleLoader


@pytest.fixture
def sample_spatial_spec():
    """Create a sample spatial specification for testing."""
    return SpatialSpec(
        crs=CRS.from_epsg(4326),  # WGS84
        transform=Affine(0.1, 0.0, -180.0, 0.0, -0.1, 90.0),  # 0.1 degree resolution
        shape=(1800, 3600),  # 180 degrees lat, 360 degrees lon at 0.1 degree resolution
    )


@pytest.fixture
def sample_raster_data() -> xr.DataArray:
    """Create a sample raster dataset for testing."""
    # Create a 10x10 array with a gradient
    data = np.linspace(0, 1, 100).reshape(10, 10)

    # Create coordinates
    x = np.linspace(-180, 180, 10)
    y = np.linspace(-90, 90, 10)

    return xr.DataArray(data, dims=('y', 'x'), coords={'x': x, 'y': y}, name='test_data')


@pytest.fixture
def sample_geotiff(tmp_path: Path, sample_raster_data: xr.DataArray) -> Path:
    """Create a sample GeoTIFF file for testing."""
    sample_raster_data.rio.write_crs('EPSG:4326', inplace=True)
    output_path = tmp_path / 'test_raster.tif'
    sample_raster_data.rio.to_raster(output_path)
    return output_path


@pytest.fixture
def mock_spec() -> MagicMock:
    """Mock SpatialSpec to bypass geographic math during tests."""
    spec = MagicMock()
    spec.is_fully_defined = True
    spec.shape = (100, 100)
    spec.transform.a = 1.0
    spec.transform.b = 0.0
    spec.transform.c = 0.0
    spec.transform.d = 0.0
    spec.transform.e = -1.0
    spec.transform.f = 0.0
    spec.crs.to_string.return_value = 'EPSG:4326'
    return spec


@pytest.fixture
def simple_data_loader(sample_geotiff: Path) -> SampleLoader:
    return SampleLoader(sample_geotiff, scale=2.0)


@pytest.fixture
def nested_data_loader(simple_data_loader: SampleLoader) -> NestedLoader:
    return NestedLoader(inner=simple_data_loader, tag='test')


@pytest.fixture
def multi_output_data_loader(sample_geotiff: Path) -> MultiOutputLoader:
    return MultiOutputLoader(loader_1=SampleLoader(sample_geotiff), loader_2=SampleLoader(sample_geotiff, scale=2))


@pytest.fixture
def cache_tree(tmp_path):
    """
    root/
      region1/
        SimpleLoader/
          tile.tif
          tile.hash.json          <- correct hash
      region2/
        SimpleLoader/
          other.tif
          other.hash.json         <- wrong hash
      region3/
        SimpleLoader/
          no_hash.tif             <- hash file absent
      .source/
        SimpleLoader/
          ignored.tif             <- should be skipped entirely
    """

    def write_hash(path: Path, hash_value: str) -> None:
        path.write_text(json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: hash_value}))

    correct_hash = 'abc123'

    # region1 — valid
    r1 = tmp_path / 'region1' / 'SimpleLoader'
    r1.mkdir(parents=True)
    (r1 / 'tile.tif').write_bytes(b'data')
    write_hash(r1 / 'tile.hash.json', correct_hash)

    # region2 — wrong hash
    r2 = tmp_path / 'region2' / 'SimpleLoader'
    r2.mkdir(parents=True)
    (r2 / 'other.tif').write_bytes(b'data')
    write_hash(r2 / 'other.hash.json', 'stale_hash')

    # region3 — missing hash
    r3 = tmp_path / 'region3' / 'SimpleLoader'
    r3.mkdir(parents=True)
    (r3 / 'no_hash.tif').write_bytes(b'data')

    # .source — must be ignored
    rs = tmp_path / '.source' / 'SimpleLoader'
    rs.mkdir(parents=True)
    (rs / 'ignored.tif').write_bytes(b'data')

    return tmp_path


@pytest.fixture(autouse=True)
def registry_tmp_path(tmp_path):
    with set_config(
        path_registry=tmp_path / '.source',
        path_cache=tmp_path / 'data_processed',
        path_figures=tmp_path / 'figures',
    ):
        yield
