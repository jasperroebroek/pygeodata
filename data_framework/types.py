from typing import Union, runtime_checkable, Protocol, Optional, Tuple

import xarray as xr
from affine import Affine
from rasterio import CRS

type RasterData = Union[xr.DataArray, xr.Dataset]
type Shape = tuple[int, int]


@runtime_checkable
class RasterLoader(Protocol):
    def parse_crs_transform_shape(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                                  shape: Optional[Shape] = None) -> Tuple[CRS, Affine, Shape]:
        ...

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        ...

    @classmethod
    def load_stack(cls, loaders: list['RasterLoader'], crs: Optional[CRS] = None,
                   transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> xr.DataArray:
        ...
