import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from pyproj import CRS

from data_framework import load_raster_data
from data_framework.data_entry import RasterDataEntry
from data_framework.errors import StackError
from data_framework.raster_loader import RasterLoaderSingle, RasterLoaderMultiple
from data_framework.types import Shape


@pytest.fixture(autouse=True)
def run_around_tests():
    shutil.rmtree("data_reprojected", ignore_errors=True)
    yield
    shutil.rmtree("data_reprojected", ignore_errors=True)


class WTDLoader(RasterLoaderSingle):
    data_entry: RasterDataEntry = RasterDataEntry(
        name='wtd',
        path=Path('data/wtd.tif')
    )


@dataclass
class WTDLoaderStackable(RasterLoaderSingle):
    i: int
    stackable: bool = True

    @property
    def data_entry(self) -> RasterDataEntry:
        return RasterDataEntry(
            name='wtd',
            path=Path('data/wtd.tif')
        )

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        da = self.get_data_entry().load(crs, transform, shape)
        return da.expand_dims({"band": [self.i]})


class WTDLoaderMultiple(RasterLoaderMultiple):
    data_entries: List[RasterDataEntry] = [
        RasterDataEntry(
            name='wtd',
            path=Path('data/wtd.tif')
        )
    ]

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        return self.get_data_entries()[0].load(crs, transform, shape)


@pytest.fixture
def wtd_loader():
    return WTDLoader()


@pytest.fixture
def wtd_loader_multiple():
    return WTDLoaderMultiple()


@pytest.fixture
def wtd_loader_stackable():
    return WTDLoaderStackable()


def test_get_crs_transform(wtd_loader):
    crs, transform, shape = wtd_loader.parse_crs_transform_shape()
    assert crs == CRS.from_epsg(4326)
    assert transform == Affine(0.008333333532777779, 0.0, 109.999999342,
                               0.0, -0.008333333112499999, -8.999999499)
    assert shape == (4320, 5400)


def test_is_generated(wtd_loader):
    assert wtd_loader.is_generated()


def test_load(wtd_loader):
    da = wtd_loader.load()
    assert isinstance(da, xr.DataArray)
    assert da.shape == (4320, 5400)
    assert da.rio.crs == CRS.from_epsg(4326)
    assert da.dtype == np.float64

    path = Path("data_reprojected/EPSG_4326/affine_0.0083_0.0000_110.0000_0.0000_-0.0083_-9.0000/4320-5400/wtd/wtd.tif")
    assert path.is_symlink()


def test_load_multiple(wtd_loader_multiple):
    da = wtd_loader_multiple.load()
    assert isinstance(da, xr.DataArray)
    assert da.shape == (4320, 5400)
    assert da.rio.crs == CRS.from_epsg(4326)
    assert da.dtype == np.float64

    path = Path(
        "data_reprojected/EPSG_4326/affine_0.0083_0.0000_110.0000_0.0000_-0.0083_-9.0000/4320-5400/wtdmultiple/wtd.tif")
    assert path.is_symlink()


def test_not_stackable(wtd_loader):
    with pytest.raises(StackError):
        load_raster_data((wtd_loader, wtd_loader))


def test_stackable():
    wtd_loader_stackable_1 = WTDLoaderStackable(1)
    wtd_loader_stackable_2 = WTDLoaderStackable(2)

    da_1 = wtd_loader_stackable_1.load()
    da_2 = wtd_loader_stackable_2.load()
    assert da_1.shape == (1, 4320, 5400)
    assert da_2.shape == (1, 4320, 5400)

    da_stacked = load_raster_data((wtd_loader_stackable_1, wtd_loader_stackable_2))
    assert da_stacked.shape == (2, 4320, 5400)

    da_not_stacked = load_raster_data((wtd_loader_stackable_1, wtd_loader_stackable_1))
    assert da_not_stacked.shape == (1, 4320, 5400)
