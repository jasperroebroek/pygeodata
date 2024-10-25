from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import xarray as xr
from affine import Affine
from rasterio import CRS

from data_framework.data_entry import RasterDataEntry
from data_framework.errors import StackError
from data_framework.types import Shape


class BaseRasterLoader(ABC):
    stackable: bool = False

    @property
    def class_name(self) -> str:
        return self.__class__.__name__.replace("Loader", "")

    def get_params(self) -> Dict[str, Any]:
        params = {}
        for key in self.__dict__:
            if key in ('stackable', 'name', 'class_name'):
                continue
            params.update({key: self.__dict__[key]})
        return params

    def get_params_str(self) -> Optional[str]:
        params = self.get_params()
        if len(params) == 0:
            return

        parts = []
        for key in sorted(params.keys()):
            parts.append(f"{key}={params[key]}")

        return "_".join(parts)

    @abstractmethod
    def is_processed(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        pass

    @abstractmethod
    def process(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        pass

    @abstractmethod
    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        pass

    @classmethod
    def load_stack(cls, loaders: list['RasterLoaderSingle'], crs: CRS, transform: Affine, shape: Shape) -> xr.DataArray:
        if not cls.stackable:
            raise StackError("Attempting to stack non-stackable rasters")

        rasters = [loader.load(crs, transform, shape) for loader in loaders]
        da_combined = xr.merge(rasters).to_array(dim='variable')

        if da_combined.coords['variable'].size == 1:
            da_combined = da_combined.isel(variable=0).drop_vars('variable')

        return da_combined.rename(loaders[0].data_entry.name)


class RasterLoaderSingle(BaseRasterLoader):
    @property
    @abstractmethod
    def data_entry(self) -> RasterDataEntry:
        pass

    def get_data_entry(self) -> RasterDataEntry:
        de = self.data_entry
        de.caller_name = self.class_name
        de.params = self.get_params()
        return de

    def is_processed(self, crs: CRS, transform: Affine, shape: Shape) -> bool:
        return self.get_data_entry().is_processed(crs, transform, shape)

    def process(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        self.get_data_entry().process(crs, transform, shape)

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        return self.get_data_entry().load(crs=crs, transform=transform, shape=shape)


class RasterLoaderMultiple(BaseRasterLoader):
    @property
    @abstractmethod
    def data_entries(self) -> List[RasterDataEntry]:
        pass

    def get_data_entries(self) -> List[RasterDataEntry]:
        des: List[RasterDataEntry] = []
        names = []
        for de in self.data_entries:
            de.caller_name = self.class_name
            de.params = self.get_params()
            des.append(de)
            if de.name in names:
                raise ValueError(f"several entries with the same name: {de.name}")
            names.append(de.name)
        return des

    @property
    @abstractmethod
    def data_entry_processed(self) -> RasterDataEntry:
        pass

    def get_data_entry_processed(self) -> RasterDataEntry:
        de = self.data_entry_processed
        de.caller_name = self.class_name
        de.params = self.get_params()
        return de

    def is_processed(self, crs: CRS, transform: Affine, shape: Shape) -> bool:
        return self.get_data_entry_processed().is_processed(crs, transform, shape)

    @abstractmethod
    def generate(self, crs: CRS, transform: Affine, shape: Shape) -> xr.DataArray:
        pass

    def process(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        if self.is_processed(crs, transform, shape):
            return

        for de in self.get_data_entries():
            de.process(crs, transform, shape)

        self.generate(crs, transform, shape)

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        crs, transform, shape = (
            self
            .get_data_entry_processed()
            .parse_crs_transform_shape(crs, transform, shape)
        )

        self.process(crs, transform, shape)

        return self.get_data_entry_processed().load(crs=crs, transform=transform, shape=shape)
