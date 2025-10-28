import pytest
from rasterio.shutil import RasterioIOError
from rioxarray.exceptions import TooManyDimensions

from pygeodata.drivers import RioXArrayDriver
from pygeodata.drivers.rioxarray import RioXArrayDriver
from tests.conftest import LUH2_NC, WTD_TIF


def test_load_tiff_keep_band():
    da = RioXArrayDriver(flatten=False).load(WTD_TIF)
    assert da.shape == (1, 4320, 5400)
    for dim in da.dims:
        assert dim in ('band', 'x', 'y')


def test_load_tiff_flat():
    da = RioXArrayDriver().load(WTD_TIF)
    assert da.shape == (4320, 5400)
    for dim in da.dims:
        assert dim in ('x', 'y')
    assert da.sum() == pytest.approx(3.01499e8, rel=1e-5)


def test_load_netcdf():
    with pytest.raises(TooManyDimensions):
        RioXArrayDriver().load(LUH2_NC)

    da = RioXArrayDriver().load(f'netcdf:{LUH2_NC}:primf')
    assert da.shape == (86, 720, 1440)
    for dim in da.dims:
        assert dim in ('time', 'y', 'x')


def test_load_nonexistent_file_raises_error():
    """Test that loading a non-existent file raises an appropriate error."""
    driver = RioXArrayDriver()
    with pytest.raises(RasterioIOError):
        driver.load('nonexistent_file.tif')
