import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Callable, Tuple

import rasterio as rio
import xarray as xr
from affine import Affine
from data_framework.paths import path_data_reprojected
from data_framework.reproject import reproject
from data_framework.types import Shape
from data_framework.utils import transform_to_str
from data_framework.xarray import load_flat_file, load_file
from rasterio import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform


@dataclass
class RasterDataEntry:
    """class that holds the info to deal with any 2D raster data"""
    name: str
    caller_name: Optional[str] = None
    params: Optional[str] = None
    subset: Optional[Union[str, int, dict]] = None
    path: Optional[Path] = None
    path_reprojected: Optional[Path] = None
    resampling: Resampling = Resampling.nearest
    reprojection_kwargs: dict = field(default_factory=dict)
    crs: Optional[CRS] = None,
    generate_func: Optional[Callable[['RasterDataEntry'], None]] = None

    def __post_init__(self):
        self._fp = None

        if self.is_generated():
            self._fp = rio.open(self.get_path())
            self._fp.close()

    def get_fp(self):
        if self._fp is None:
            raise FileNotFoundError(f'file not found for {self.get_path()}')
        return self._fp

    def convert_path(self, crs: CRS, transform: Affine, shape: Shape) -> Path:
        """Function that converts a path of the data to the reprojected data.
        NOTE: This is not always guaranteed to work"""
        p = (self.caller_name, self.params) if self.params is not None else (self.caller_name,)
        return Path(path_data_reprojected,
                    f"{crs.to_string().replace(':', '_')}",
                    transform_to_str(transform),
                    f"{shape[0]}-{shape[1]}",
                    *p,
                    f"{self.name}.tif")

    def get_path(self) -> Optional[Path]:
        return self.path

    def get_crs_transform_shape(self) -> Tuple[CRS, Affine, Shape]:
        fp = self.get_fp()
        crs = fp.crs if fp.crs is not None else self.crs

        if crs is None:
            raise rio.RasterioIOError("Can't find CRS in file. It needs to be set on the DataEntry")

        return crs, fp.transform, fp.shape

    def parse_crs_transform_shape(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                                  shape: Optional[Shape] = None) -> Tuple[CRS, Affine, Shape]:
        if crs is None:
            crs, transform, shape = self.get_crs_transform_shape()

        if transform is None or shape is None:
            fp = self.get_fp()
            transform, width, height = calculate_default_transform(
                fp.crs,
                crs,
                fp.width,
                fp.height,
                *fp.bounds
            )
            shape = (height, width)

        return crs, transform, shape

    def get_reprojected_path(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                             shape: Optional[Shape] = None) -> Path:
        crs, transform, shape = self.parse_crs_transform_shape(crs, transform, shape)

        if self.path_reprojected is None:
            path = self.get_path()
            if path is None:
                raise ValueError(f"Neither path nor path_reprojected specified: {self}")
            if self.caller_name is None:
                raise ValueError(f"No caller name specified: {self}")
            return self.convert_path(crs, transform, shape)

        return self.path_reprojected

    def is_generated(self):
        return self.path.exists()

    def generate(self) -> None:
        if self.generate_func is None:
            raise NotImplementedError
        print(f"generating {self.name}")
        self.generate_func(self)

    def is_reprojected(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                       shape: Optional[Shape] = None) -> bool:
        p = self.get_reprojected_path(crs=crs, transform=transform, shape=shape)
        return p.exists() or p.is_symlink()

    def _reproject(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        path = self.get_path()

        if path is None:
            raise ValueError(f"Can't process without a path being specified: {self.name}")

        output_path = self.get_reprojected_path(crs=crs, transform=transform, shape=shape)

        da = load_file(path, cache=False)

        dim_name = da.rio._check_dimensions()
        if dim_name == "band" and da[dim_name].size == 1:
            da = da.sel(band=1).drop_vars('band')

        if isinstance(self.subset, (str, int)):
            da = da.sel(band=self.subset)
        elif isinstance(self.subset, dict):
            da = da.sel(**self.subset)

        src_crs, _, _ = self.get_crs_transform_shape()

        reproject(da, output_path, src_crs=src_crs, dst_crs=crs, dst_transform=transform, dst_shape=shape,
                  resampling=self.resampling, **self.reprojection_kwargs)
        da.close()

    def reproject(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                  shape: Optional[Shape] = None) -> None:
        if not self.is_generated():
            self.generate()

        crs, transform, shape = self.parse_crs_transform_shape(crs, transform, shape)

        reprojected_path = self.get_reprojected_path(crs, transform, shape)
        reprojected_path.parent.mkdir(exist_ok=True, parents=True)

        self._reproject(crs=crs, transform=transform, shape=shape)
        gc.collect()

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        crs, transform, shape = self.parse_crs_transform_shape(crs, transform, shape)

        if not self.is_reprojected(crs, transform, shape):
            self.reproject(crs, transform, shape)

        reprojected_path = self.get_reprojected_path(crs=crs, transform=transform, shape=shape)

        return load_flat_file(reprojected_path, cache=False, name=self.name)
