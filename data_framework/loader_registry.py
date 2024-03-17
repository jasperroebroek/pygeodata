from typing import Protocol, Optional, Dict, Tuple, runtime_checkable

import xarray as xr
from affine import Affine
from rasterio import CRS


@runtime_checkable
class RasterLoader(Protocol):
    def parse_crs_transform(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> Tuple[CRS, Affine]:
        ...

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> xr.DataArray:
        ...

    @classmethod
    def load_stack(cls, loaders: list['RasterLoader'], crs: Optional[CRS] = None,
                   transform: Optional[Affine] = None) -> xr.DataArray:
        ...


class LoaderRegistry:
    _instance: Optional['LoaderRegistry'] = None
    _loaders: Dict[str, RasterLoader] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def register_loader(self, name: str, loader: RasterLoader) -> None:
        if name in self._loaders:
            raise ValueError(f"Name already registered {name=}")
        if loader in self._loaders.values():
            raise ValueError(f"Loader already registered {loader=}")
        self._loaders[name] = loader

    def get_loader(self, name: str):
        if name not in self._loaders:
            raise ValueError(f"No such loader: {name=}")
        return self._loaders.get(name)
