from typing import Union, Iterable, Optional, List

import xarray as xr
from affine import Affine
from rasterio import CRS
from rasterio.errors import CRSError, TransformError

from data_framework.loader_registry import LoaderRegistry
from data_framework.types import Shape, RasterData, RasterLoader


def _load_raster_stack(loaders: List[RasterLoader], crs: Optional[CRS] = None,
                       transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> RasterData:
    if not isinstance(loaders[0], RasterLoader):
        raise TypeError(f"No RasterLoader {loaders[0]=}")

    for loader in loaders:
        if not isinstance(loader, type(loaders[0])):
            print(f"Not all loaders are matching {loader=} {loaders[0]=}")

    base_loader = type(loaders[0])
    return base_loader.load_stack(loaders, crs, transform, shape)


def load_raster_data(loaders: Union[str, RasterLoader, Iterable[RasterLoader]],
                     template: Optional[xr.DataArray] = None, *, crs: Optional[CRS] = None,
                     transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> RasterData:
    if template is not None:
        if template.rio.crs is None:
            raise CRSError("No CRS")
        if template.rio.transform is None:
            raise TransformError("No transform")

        crs = template.rio.crs
        transform = template.rio.transform(recalc=True)
        shape = template.rio.shape

    if isinstance(loaders, (list, tuple)):
        return _load_raster_stack(loaders, crs, transform, shape)

    if isinstance(loaders, str):
        lr = LoaderRegistry()
        loaders = lr.get_loader(loaders)()

    return loaders.load(crs, transform, shape)
