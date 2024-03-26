from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import xarray as xr
from affine import Affine
from rasterio import CRS

from data_framework.data_entry import RasterDataEntry
from data_framework.errors import StackError
from data_framework.types import Shape


class BaseRasterLoader(ABC):
    stackable: bool = False

    @abstractmethod
    def get_name(self) -> str:
        pass

    def get_class_name(self) -> str:
        return self.__class__.__name__.replace("Loader", "")


    def get_params(self) -> Optional[str]:
        parts = []
        for key in self.__dict__:
            if key in ('stackable', 'name'):
                continue
            parts.append(f"{key}={self.__dict__[key]}")

        if len(parts) == 0:
            return

        return "_".join(parts)

    @abstractmethod
    def parse_crs_transform_shape(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                                  shape: Optional[Shape] = None) -> Tuple[CRS, Affine, Shape]:
        pass

    @abstractmethod
    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        pass

    @classmethod
    def load_stack(cls, loaders: list['RasterLoaderSingle'], crs: Optional[CRS] = None,
                   transform: Optional[Affine] = None, shape: Optional[Shape] = None) -> xr.DataArray:
        if not cls.stackable:
            raise StackError("Attempting to stack non-stackable rasters")

        crs, transform, shape = loaders[0].parse_crs_transform_shape(crs, transform, shape)

        bl = loaders[0]
        rasters = [loader.load(crs, transform, shape) for loader in loaders]
        da_combined = xr.merge(rasters).to_array(dim='variable')

        if da_combined.coords['variable'].size == 1:
            da_combined = da_combined.isel(variable=0).drop_vars('variable')

        return da_combined.rename(bl.data_entry.name)


class RasterLoaderSingle(BaseRasterLoader):
    @property
    @abstractmethod
    def data_entry(self) -> RasterDataEntry:
        pass

    def get_name(self) -> str:
        return self.get_data_entry().name

    def get_data_entry(self) -> RasterDataEntry:
        de = self.data_entry
        de.caller_name = self.get_class_name()
        de.params = self.get_params()
        return de

    def parse_crs_transform_shape(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                                  shape: Optional[Shape] = None) -> Tuple[CRS, Affine, Shape]:
        return self.get_data_entry().parse_crs_transform_shape(crs, transform, shape)

    def is_generated(self) -> bool:
        return self.get_data_entry().is_generated()

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        return self.get_data_entry().load(crs=crs, transform=transform, shape=shape)


class RasterLoaderMultiple(BaseRasterLoader):
    name: str

    @property
    @abstractmethod
    def data_entries(self) -> List[RasterDataEntry]:
        pass

    def get_data_entries(self) -> List[RasterDataEntry]:
        des: List[RasterDataEntry] = []
        names: List[str] = []
        for entry in self.data_entries:
            entry.caller_name = self.get_class_name()
            entry.params = self.get_params()
            des.append(entry)
            if entry.name in names:
                raise ValueError(f"several entries with the same name: {entry.name}")
            names.append(entry.name)
        return des

    def get_name(self) -> str:
        return self.name

    def parse_crs_transform_shape(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                                  shape: Optional[Shape] = None) -> Tuple[CRS, Affine, Shape]:
        crs_list = []
        transform_list = []
        shape_list = []

        for de in self.get_data_entries():
            ccrs, ctransform, cshape = de.parse_crs_transform_shape(crs, transform, shape)
            crs_list.append(ccrs)
            transform_list.append(ctransform)
            shape_list.append(cshape)

        for ccrs in crs_list:
            if ccrs != crs_list[0]:
                print(f"CRSs are not all the same {ccrs=} {crs_list[0]=}. Taking the first one")

        for ctransform in transform_list:
            if ctransform != transform_list[0]:
                print(f"Transforms are not all the same {ctransform=} {transform_list[0]=}. Taking the first one")

        for cshape in shape_list:
            if cshape != shape_list[0]:
                print(f"Shapes are not all the same {cshape=} {shape_list[0]=}. Taking the first one")

        return crs_list[0], transform_list[0], shape_list[0]

    def is_generated(self) -> bool:
        return all((e.is_generated() for e in self.get_data_entries()))

    @abstractmethod
    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        pass
