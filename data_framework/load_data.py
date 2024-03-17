from typing import Union, Iterable, Optional, List

import xarray as xr
from rasterio.errors import CRSError

from data_framework.loader_registry import RasterLoader, LoaderRegistry

type RasterData = Union[xr.DataArray, xr.Dataset]


def _load_raster(loader: RasterLoader, template: Optional[xr.DataArray] = None) -> RasterData:
    if template is not None:
        if template.rio.crs is None:
            raise CRSError("No CRS")
        if template.rio.transform is None:
            raise CRSError("No transform")

        crs = template.rio.crs
        transform = template.rio.transform()
    else:
        crs, transform = loader.parse_crs_transform()

    return loader.load(crs, transform)


def _load_raster_stack(loaders: List[RasterLoader], template: Optional[xr.DataArray] = None) -> RasterData:
    if not isinstance(loaders[0], RasterLoader):
        raise TypeError(f"No RasterLoader {loaders[0]=}")

    for loader in loaders:
        if not isinstance(loader, type(loaders[0])):
            print(f"Not all loaders are matching {loader=} {loaders[0]=}")

    if template is not None:
        if template.rio.crs is None:
            raise CRSError("No CRS")
        if template.rio.transform is None:
            raise CRSError("No transform")

        crs = template.rio.crs
        transform = template.rio.transform()
    else:
        crs, transform = loaders[0].parse_crs_transform()

    base_loader = type(loaders[0])
    return base_loader.load_stack(loaders, crs, transform)


def load_raster_data(loaders: Union[str, RasterLoader, Iterable[RasterLoader]],
                     template: Optional[xr.DataArray] = None) -> RasterData:
    if isinstance(loaders, (list, tuple)):
        return _load_raster_stack(loaders, template)

    if isinstance(loaders, str):
        lr = LoaderRegistry()
        loaders = lr.get_loader(loaders)()

    return _load_raster(loaders, template)
