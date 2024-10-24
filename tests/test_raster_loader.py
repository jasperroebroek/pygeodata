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
from data_framework.paths import pc
from data_framework.raster_loader import RasterLoaderMultiple, RasterLoaderSingle
from data_framework.types import Shape


@pytest.fixture(autouse=True)
def run_around_tests(tmp_path):
    pc.path_data_processed = tmp_path


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
    name = 'wtdmultiple'

    data_entries: List[RasterDataEntry] = [
        RasterDataEntry(
            name='wtd',
            path=Path('data/wtd.tif')
        )
    ]

    data_entry_processed: RasterDataEntry = RasterDataEntry(
        name='wtd',
        path=Path('data/wtd.tif')
    )

    def generate(self, crs: CRS, transform: Affine, shape: Shape) -> xr.DataArray:
        pass


@pytest.fixture
def wtd_loader():
    return WTDLoader()


@pytest.fixture
def wtd_loader_multiple():
    return WTDLoaderMultiple()


@pytest.fixture
def wtd_loader_stackable():
    return WTDLoaderStackable()


def test_load(wtd_loader):
    da = wtd_loader.load()
    assert isinstance(da, xr.DataArray)
    assert da.shape == (4320, 5400)
    assert da.rio.crs == CRS.from_epsg(4326)
    assert da.dtype == np.float64

    path = Path(
        pc.path_data_processed,
        "EPSG_4326",
        "affine_0.0083_0.0000_110.0000_0.0000_-0.0083_-9.0000",
        "4320-5400",
        "WTD",
        "wtd.tif"
    )
    assert path == Path(da.encoding['source'])


def test_load_multiple(wtd_loader_multiple):
    da = wtd_loader_multiple.load()
    assert isinstance(da, xr.DataArray)
    assert da.shape == (4320, 5400)
    assert da.rio.crs == CRS.from_epsg(4326)
    assert da.dtype == np.float64

    path = Path(
        pc.path_data_processed,
        "EPSG_4326",
        "affine_0.0083_0.0000_110.0000_0.0000_-0.0083_-9.0000",
        "4320-5400",
        "WTDMultiple",
        "wtd.tif"
    )
    assert path == Path(da.encoding['source'])


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
