import pytest

from data_framework.xarray import load_file, load_flat_file


def test_load_file():
    da = load_file('data/wtd.tif')
    assert da.shape == (1, 4320, 5400)
    for dim in da.dims:
        assert dim in ('band', 'x', 'y')
    assert da.sum() == pytest.approx(3.01499e8, rel=1e-5)


def test_load_flat_file():
    da = load_flat_file('data/wtd.tif')
    assert da.shape == (4320, 5400)
    for dim in da.dims:
        assert dim in ('x', 'y')
    assert da.sum() == pytest.approx(3.01499e8, rel=1e-5)
