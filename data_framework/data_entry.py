import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import rasterio as rio
import xarray as xr
from affine import Affine
from rasterio import CRS, DatasetReader
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform

from data_framework.paths import generate_path
from data_framework.reproject import reproject
from data_framework.types import GenerateFunc, Shape
from data_framework.xarray import load_file, load_flat_file


@dataclass
class RasterDataEntry:
    """class that holds the info to deal with any 2D raster data"""
    name: str
    subset: Optional[Union[str, int, dict]] = None
    path: Optional[Path] = None
    resampling: Resampling = Resampling.nearest
    reprojection_kwargs: dict = field(default_factory=dict)
    crs: Optional[CRS] = None,
    generate_func: Optional[GenerateFunc] = None

    def __post_init__(self):
        self.caller_name: str = 'RasterDataEntry'
        self.params: dict[str, Any] = {}

    def path_exists(self) -> bool:
        if self.path is None:
            return False
        return self.path.exists() or self.path.is_symlink()

    def get_fp(self) -> DatasetReader:
        if not self.path_exists():
            raise FileNotFoundError(f'file not found for {self.path=}')

        fp = rio.open(self.path)
        fp.close()

        return fp

    def get_processed_path(self, crs: CRS, transform: Affine, shape: Shape) -> Path:
        """Function that converts a path of the data to the reprojected data."""
        return generate_path(
            crs,
            transform,
            shape,
            self.caller_name,
            self.name,
            **self.params
        )

    def is_processed(self, crs: CRS, transform: Affine, shape: Shape) -> bool:
        p = self.get_processed_path(crs=crs, transform=transform, shape=shape)
        return p.exists() or p.is_symlink()

    def get_crs_transform_shape_from_file(self) -> Tuple[CRS, Affine, Shape]:
        fp = self.get_fp()
        crs = fp.crs if fp.crs is not None else self.crs

        if crs is None:
            raise rio.RasterioIOError("Can't find CRS in file. It needs to be set on the DataEntry")

        return crs, fp.transform, fp.shape

    def parse_crs_transform_shape(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
                                  shape: Optional[Shape] = None) -> Tuple[CRS, Affine, Shape]:
        if crs is None:
            crs, transform, shape = self.get_crs_transform_shape_from_file()

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

    def reproject(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        output_path = self.get_processed_path(crs=crs, transform=transform, shape=shape)
        output_path.parent.mkdir(exist_ok=True, parents=True)

        da = load_file(self.path, cache=False)

        dim_name = da.rio._check_dimensions()
        if dim_name == "band" and da[dim_name].size == 1:
            da = da.sel(band=1).drop_vars('band')

        if isinstance(self.subset, (str, int)):
            da = da.sel(band=self.subset)
        elif isinstance(self.subset, dict):
            da = da.sel(**self.subset)

        src_crs, _, _ = self.get_crs_transform_shape_from_file()

        reproject(da, output_path, src_crs=src_crs, dst_crs=crs, dst_transform=transform, dst_shape=shape,
                  resampling=self.resampling, **self.reprojection_kwargs)
        da.close()
        gc.collect()

    def process(self, crs: CRS, transform: Affine, shape: Shape) -> None:
        if self.is_processed(crs, transform, shape):
            return

        if self.path_exists():
            self.reproject(crs, transform, shape)
            return

        if self.generate_func is None:
            raise NotImplementedError(f"Path not provided or not found and no function specified for generating the "
                                      f"data: {self.name}")

        print(f"Generating {self.name}")
        self.generate_func(self, crs, transform, shape)

    def load(self, crs: Optional[CRS] = None, transform: Optional[Affine] = None,
             shape: Optional[Shape] = None) -> xr.DataArray:
        crs, transform, shape = self.parse_crs_transform_shape(crs, transform, shape)
        self.process(crs, transform, shape)
        path_processed = self.get_processed_path(crs=crs, transform=transform, shape=shape)

        return load_flat_file(path_processed, cache=False, name=self.name)
