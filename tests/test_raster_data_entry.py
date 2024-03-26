import shutil
from pathlib import Path

import pytest
from affine import Affine
from rasterio import CRS

from data_framework.data_entry import RasterDataEntry


@pytest.fixture
def wtd_entry():
    wtd_entry = RasterDataEntry('wtd', caller_name='test_caller', path=Path('data/wtd.tif'))
    return wtd_entry


@pytest.fixture(autouse=True)
def run_around_tests():
    shutil.rmtree("data_reprojected", ignore_errors=True)
    yield
    shutil.rmtree("data_reprojected", ignore_errors=True)


def test_raster_data_entry(wtd_entry):
    assert wtd_entry.get_path() == Path('data/wtd.tif')
    assert wtd_entry.get_reprojected_path(*wtd_entry.get_crs_transform_shape()) == Path(
        'data_reprojected/EPSG_4326/affine_0.0083_0.0000_110.0000_0.0000_-0.0083_-9.0000/4320-5400/test_caller/wtd.tif')


def test_get_path(wtd_entry):
    path = wtd_entry.get_path()
    assert path.exists()  # Assuming the path exists


def test_get_reprojected_paths(wtd_entry):
    crs = CRS.from_epsg(3857)  # Assuming a different CRS
    transform = Affine.scale(0.1, 0.1)  # Assuming a different transform
    shape = (1000, 1000)

    path = wtd_entry.get_reprojected_path(crs, transform, shape)
    assert path == Path(
        'data_reprojected/EPSG_3857/affine_0.1000_0.0000_0.0000_0.0000_0.1000_0.0000/1000-1000/test_caller/wtd.tif')
    assert not path.exists()
    assert not path.is_symlink()

    da = wtd_entry.load()
    path = wtd_entry.get_reprojected_path(*wtd_entry.get_crs_transform_shape())
    assert da.encoding['source'] == str(path)


def test_generate(wtd_entry):
    with pytest.raises(NotImplementedError):
        wtd_entry.generate()


def test_reproject(wtd_entry):
    crs, transform, shape = wtd_entry.parse_crs_transform_shape(CRS.from_authority('ESRI', '54012'))
    wtd_entry.reproject(crs)

    path = wtd_entry.get_reprojected_path(crs, transform, shape)
    assert not path.is_symlink()
    assert path.exists()

    assert wtd_entry._fp.crs.to_string() == "EPSG:4326"
