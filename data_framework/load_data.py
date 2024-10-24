from typing import Iterable, List, Optional, Union

from affine import Affine
from rasterio import CRS

from data_framework.loader_registry import LoaderRegistry
from data_framework.types import RasterData, RasterLoader, Shape


def _load_raster_stack(loaders: List[RasterLoader], crs: Optional[CRS] = None,
                       transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> RasterData:
    if not isinstance(loaders[0], RasterLoader):
        raise TypeError(f"No RasterLoader {loaders[0]=}")

    for loader in loaders:
        if not isinstance(loader, type(loaders[0])):
            print(f"Not all loaders are matching {loader=} {loaders[0]=}")

    base_loader = type(loaders[0])
    return base_loader.load_stack(loaders, crs, transform, shape)


def load_raster_data(loaders: Union[str, RasterLoader, Iterable[RasterLoader]], *, crs: Optional[CRS] = None,
                     transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> RasterData:
    if isinstance(loaders, (list, tuple)):
        return _load_raster_stack(loaders, crs, transform, shape)

    if isinstance(loaders, str):
        lr = LoaderRegistry()
        loaders = lr.get_loader(loaders)()

    return loaders.load(crs, transform, shape)
