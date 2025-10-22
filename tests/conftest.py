from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS

from pygeodata.drivers import RioXArrayDriver
from pygeodata.loader import DataLoader
from pygeodata.processors import Reprojector
from pygeodata.types import SpatialSpec

# Test data paths
TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
WTD_TIF = TEST_DATA_DIR / 'wtd.tif'
LUH2_NC = TEST_DATA_DIR / 'luh2.nc'
COUNTRIES_SHP = TEST_DATA_DIR / 'countries' / 'ne_110m_admin_0_map_units.shp'


@pytest.fixture
def sample_spatial_spec():
    """Create a sample spatial specification for testing."""
    return SpatialSpec(
        crs=CRS.from_epsg(4326),  # WGS84
        transform=Affine(0.1, 0.0, -180.0, 0.0, -0.1, 90.0),  # 0.1 degree resolution
        shape=(1800, 3600),  # 180 degrees lat, 360 degrees lon at 0.1 degree resolution
    )


@pytest.fixture
def sample_raster_data():
    """Create a sample raster dataset for testing."""
    # Create a 10x10 array with a gradient
    data = np.linspace(0, 1, 100).reshape(10, 10)

    # Create coordinates
    x = np.linspace(-180, 180, 10)
    y = np.linspace(-90, 90, 10)

    return xr.DataArray(data, dims=('y', 'x'), coords={'x': x, 'y': y}, name='test_data')


@pytest.fixture
def sample_geotiff(tmp_path, sample_raster_data):
    """Create a sample GeoTIFF file for testing."""

    # Add spatial attributes
    sample_raster_data.rio.write_crs('EPSG:4326', inplace=True)

    # Save to a temporary GeoTIFF
    output_path = tmp_path / 'test_raster.tif'
    sample_raster_data.rio.to_raster(output_path)

    return output_path


@pytest.fixture
def sample_loader_class(sample_spatial_spec, sample_geotiff):
    class SampleLoader(DataLoader):
        processor = Reprojector(sample_geotiff)
        driver = RioXArrayDriver()

    return SampleLoader


@pytest.fixture
def sample_loader_class_complex(sample_spatial_spec, sample_geotiff):
    @dataclass
    class ComplexSampleLoader(DataLoader):
        time: int
        resolution: int
        __slots__ = dict(processor=Reprojector(sample_geotiff), driver=RioXArrayDriver())

    return ComplexSampleLoader
