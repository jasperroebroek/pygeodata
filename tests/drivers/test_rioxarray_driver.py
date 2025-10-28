import pytest
from pygeodata.drivers.rioxarray import RioXArrayDriver
from rasterio.shutil import RasterioIOError
from rioxarray.exceptions import TooManyDimensions
from tests.conftest import LUH2_NC, WTD_TIF


def test_load_tiff_keep_band():
    da = RioXArrayDriver(flatten=False)(WTD_TIF)
    assert da.shape == (1, 4320, 5400)
    for dim in da.dims:
        assert dim in ('band', 'x', 'y')


def test_load_tiff_flat():
    da = RioXArrayDriver()(WTD_TIF)
    assert da.shape == (4320, 5400)
    for dim in da.dims:
        assert dim in ('x', 'y')
    assert da.sum() == pytest.approx(3.01499e8, rel=1e-5)


def test_load_netcdf():
    with pytest.raises(TooManyDimensions):
        RioXArrayDriver()(LUH2_NC)

    da = RioXArrayDriver()(f'netcdf:{LUH2_NC}:primf')
    assert da.shape == (86, 720, 1440)
    for dim in da.dims:
        assert dim in ('time', 'y', 'x')


def test_load_nonexistent_file_raises_error():
    """Test that loading a non-existent file raises an appropriate error."""
    driver = RioXArrayDriver()
    with pytest.raises(RasterioIOError):
        driver('nonexistent_file.tif')
