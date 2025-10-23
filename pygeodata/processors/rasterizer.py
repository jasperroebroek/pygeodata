from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from pygeodata.config import get_config
from pygeodata.drivers import RioXArrayDriver
from pygeodata.options import RasterCreationOptions
from pygeodata.types import SpatialSpec


@dataclass
class Rasterizer:
    """
    Rasterize a vector dataset to a single-band raster.

    Parameters
    ----------
    path : Path
        Path to the vector dataset (e.g., shapefile, GeoPackage).
    column : str
        Name of the column to use as raster values (must be numeric).
    load_df_func : Callable[[str | Path, SpatialSpec], gpd.GeoDataFrame], optional
        Function to load the vector data. By default, reads with geopandas and reprojects to `spec.crs`.
    all_touched : bool, default=True
        Whether to burn all pixels touched by geometries.
    dtype : np.dtype, optional
        Data type for raster. Defaults to the dtype of `column`.
    fill_value : float, optional
        Nodata value for raster. Defaults to `_NODATA_DTYPE_MAP` based on dtype.
    rasterize_kw : dict, optional
        Additional keyword arguments passed to `rasterio.features.rasterize`.
    raster_creation_options : RasterCreationOptions, optional
        Optional raster creation profile (compression, tiling, etc.).
    """

    path: Path
    column: str = 'index'
    load_df_func: Callable[[str | Path, SpatialSpec], gpd.GeoDataFrame] = field(
        default_factory=lambda: lambda path, spec: gpd.read_file(path).to_crs(spec.crs).reset_index(),
    )
    all_touched: bool = True
    dtype: np.dtype | None = None
    fill_value: float | None = None
    rasterize_kw: dict[str, Any] = field(default_factory=dict)
    raster_creation_options: RasterCreationOptions | None = None

    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None:
        df = self.load_df_func(self.path, spec)

        if df.crs != spec.crs:
            raise ValueError(f'GeoDataFrame CRS ({df.crs}) does not match target spec CRS ({spec.crs}).')

        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in GeoDataFrame.")

        dtype = self.dtype or df[self.column].dtype

        if not np.issubdtype(dtype, np.number):
            raise TypeError(f"Column '{self.column}' must be numeric, got {dtype}.")

        fill_value = self.fill_value or (np.nan if np.issubdtype(dtype, np.floating) else 0)

        if fill_value in df[self.column]:
            raise ValueError(f'Fill value {fill_value} is present in the data. Overwrite with a different value.')

        raster = rasterize(
            ((row.geometry, row[self.column]) for i, row in df.iterrows()),
            out_shape=spec.shape,
            transform=spec.transform,
            fill=fill_value,
            all_touched=self.all_touched,
            dtype=dtype,
            **self.rasterize_kw,
        )

        raster_creation_options = self.raster_creation_options or get_config().raster_creation_options

        with rasterio.open(
            dst_path,
            'w',
            driver='GTiff',
            height=spec.shape[0],
            width=spec.shape[1],
            count=1,
            dtype=dtype,
            crs=spec.crs,
            transform=spec.transform,
            **raster_creation_options.to_dict(),
        ) as dst:
            dst.write(raster, 1)

    default_driver = RioXArrayDriver()
