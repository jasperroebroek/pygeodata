import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS

import pygeodata.loader as loader_module
from pygeodata.drivers import RioXArrayDriver
from pygeodata.loader import DataLoader
from pygeodata.processors import Reprojector
from pygeodata.types import SpatialSpec
from tests.dummy_loaders import NestedLoader, SimpleLoader


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
    @dataclass(repr=False)
    class ComplexSampleLoader(DataLoader):
        time: int
        resolution: int
        __slots__ = dict(processor=Reprojector(sample_geotiff), driver=RioXArrayDriver())

    return ComplexSampleLoader


@pytest.fixture
def mock_spec():
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
def mock_config(tmp_path):
    """Mock get_config() so paths are generated in a temporary test directory."""
    with patch.object(loader_module, 'get_config') as mock_conf:
        mock_conf.return_value.path_data_processed = tmp_path
        yield mock_conf


@pytest.fixture
def simple_loader():
    return SimpleLoader('a.tif', scale=2.0)


@pytest.fixture
def nested_loader(simple_loader):
    return NestedLoader(inner=simple_loader, tag='test')


@pytest.fixture
def secondary_loader_class():
    """A second simple loader to use as a co-output."""

    class SecondaryLoader(DataLoader):
        driver = RioXArrayDriver()

    return SecondaryLoader


@pytest.fixture
def multi_output_loader_class(sample_loader_class, secondary_loader_class):
    """
    A loader whose _process yields two loaders.
    The yielded loaders (not self) get their state hashes written.
    """
    calls = []

    class MultiOutputLoader(DataLoader):
        ext = 'tif'
        driver = RioXArrayDriver()

        def _process(self, spec):
            calls.append(spec)
            yield sample_loader_class()
            yield secondary_loader_class()

    MultiOutputLoader._calls = calls
    return MultiOutputLoader


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
      _source/
        SimpleLoader/
          ignored.tif             <- should be skipped entirely
    """

    def write_hash(path: Path, hash_value: str) -> None:
        path.write_text(json.dumps({'source_hierarchy_hash': hash_value}))

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

    # _source — must be ignored
    rs = tmp_path / '_source' / 'SimpleLoader'
    rs.mkdir(parents=True)
    (rs / 'ignored.tif').write_bytes(b'data')

    return tmp_path
