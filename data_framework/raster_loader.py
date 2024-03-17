from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List

import rasterio as rio
import xarray as xr
from affine import Affine
from rasterio import CRS

from data_framework.data_entry import RasterDataEntry
from data_framework.xarray import load_flat_file


class BaseRasterLoader(ABC):
    stackable: bool

    def get_class_name(self):
        parts = []
        for key in self.__dict__:
            if key == 'stackable':
                continue
            parts.append(f"{key}={self.__dict__[key]}")
        return self.__class__.__name__.replace("Loader", "") + "_".join(parts)

    @abstractmethod
    def get_crs_transform(self) -> Tuple[CRS, Affine]:
        pass

    def parse_crs_transform(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> Tuple[CRS, Affine]:
        if crs is None or transform is None:
            print("using file CRS and transform")
            crs, transform = self.get_crs_transform()
        return crs, transform

    @abstractmethod
    def is_generated(self) -> bool: pass

    @abstractmethod
    def is_reprojected(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> bool: pass

    @abstractmethod
    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> xr.DataArray:
        pass

    @classmethod
    def load_stack(cls, loaders: list['RasterLoaderSingle'], crs: Optional[CRS] = None,
                   transform: Optional[Affine] = None) -> xr.DataArray:
        if not cls.stackable:
            raise IndexError("Attempting to stack non-stackable rasters")

        crs, transform = loaders[0].parse_crs_transform(crs, transform)

        bl = loaders[0]
        rasters = [loader.load(crs, transform) for loader in loaders]
        return xr.merge(rasters).to_array(dim='variable').rename(bl.data_entry.name)


class RasterLoaderSingle(BaseRasterLoader):
    @property
    @abstractmethod
    def data_entry(self) -> RasterDataEntry:
        pass

    def get_data_entry(self) -> RasterDataEntry:
        de = self.data_entry
        de.caller_name = self.get_class_name()
        return de

    def get_crs_transform(self) -> Tuple[CRS, Affine]:
        fp = rio.open(self.get_path())
        crs = fp.crs
        transform = fp.transform
        fp.close()
        return crs, transform

    def get_path(self) -> Path:
        path = self.get_data_entry().get_path()
        if path is None or not path.exists():
            raise OSError(f"File does not exist: {path}")
        return path

    def get_reprojected_path(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> Path:
        crs, transform = self.parse_crs_transform(crs, transform)
        path = self.get_data_entry().get_reprojected_path(crs=crs, transform=transform)
        if path is None or (not path.exists() and not path.is_symlink()):
            raise OSError(f"File does not exist: {path}")
        return path

    def generate(self) -> None:
        self.get_data_entry().generate()

    def reproject(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> None:
        crs, transform = self.parse_crs_transform(crs, transform)
        if not self.is_generated():
            self.generate()

        self.get_data_entry().reproject(crs=crs, transform=transform)

    def is_generated(self) -> bool:
        return self.get_data_entry().is_generated()

    def is_reprojected(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> bool:
        crs, transform = self.parse_crs_transform(crs, transform)
        return self.get_data_entry().is_reprojected(crs=crs, transform=transform)

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> xr.DataArray:
        crs, transform = self.parse_crs_transform(crs, transform)
        de = self.get_data_entry()
        reprojected_path = de.get_reprojected_path(crs=crs, transform=transform)

        if not self.is_reprojected(crs, transform):
            self.reproject(crs, transform)

        return load_flat_file(reprojected_path, name=de.name)


class RasterLoaderMultiple(BaseRasterLoader):
    @property
    @abstractmethod
    def data_entries(self) -> List[RasterDataEntry]:
        pass

    def get_data_entries(self) -> List[RasterDataEntry]:
        des: List[RasterDataEntry] = []
        names: List[str] = []
        for entry in self.data_entries:
            entry.caller_name = self.get_class_name()
            des.append(entry)
            if entry.name in names:
                raise ValueError(f"several entries with the same name: {entry.name}")
            names.append(entry.name)
        return des

    def get_paths(self) -> List[Path]:
        paths = []

        for de in self.get_data_entries():
            path = de.get_path()
            if path is None or not path.exists():
                raise OSError(f"File does not exist {path=}")
            paths.append(path)

        return paths

    def get_reprojected_paths(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> List[Path]:
        paths = []
        crs, transform = self.parse_crs_transform(crs, transform)

        for de in self.get_data_entries():
            path = de.get_reprojected_path(crs=crs, transform=transform)
            if path is None or (not path.exists() and not path.is_symlink()):
                raise OSError(f"File does not exist {path=}")
            paths.append(path)

        return paths

    def get_crs_transform(self) -> Tuple[CRS, Affine]:
        crs_list = []
        transform_list = []

        for de in self.get_data_entries():
            fp = rio.open(de.get_path())
            crs_list.append(fp.crs)
            transform_list.append(fp.transform)
            fp.close()

        for crs in crs_list:
            if crs != crs_list[0]:
                print(f"CRSs are not all the same {crs=} {crs[0]=}. Taking the first one")

        for transform in transform_list:
            if transform != transform_list[0]:
                print(f"Transforms are not all the same {transform=} {transform[0]=}. Taking the first one")

        return crs_list[0], transform_list[0]

    def is_generated(self) -> bool:
        return all((e.is_generated() for e in self.get_data_entries()))

    def is_reprojected(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> bool:
        crs, transform = self.parse_crs_transform(crs, transform)
        return all((e.is_reprojected(crs, transform) for e in self.get_data_entries()))

    @abstractmethod
    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None) -> xr.DataArray:
        pass
