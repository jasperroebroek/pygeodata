from typing import Callable, Optional, Protocol, Union, runtime_checkable

import xarray as xr
from affine import Affine
from rasterio import CRS

type GenerateFunc = Callable[['RasterDataEntry', Optional[CRS], Optional[Affine], Optional[Shape]], None]
type RasterData = Union[xr.DataArray, xr.Dataset]
type Shape = tuple[int, int]


@runtime_checkable
class RasterLoader(Protocol):
    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        ...

    @classmethod
    def load_stack(cls, loaders: list['RasterLoader'], crs: Optional[CRS] = None,
                   transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> RasterData:
        ...
