from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import DTypeLike
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
    path : Path, optional
        Path to the vector dataset (e.g., shapefile, GeoPackage).
    load_df : Callable[[SpatialSpec], gpd.GeoDataFrame], optional
        Function to load the vector data. ``path`` is preferred if provided.
    values : str or int, default='index'
        Values imprinted on the raster. A numeric value will be used for all polygons, while a string will be
        interpreted as the name of the column in the data with the values. A special value is `index`, which refers to
        the index of the dataframe.
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

    path: Path | None = None
    load_df: Callable[[SpatialSpec], gpd.GeoDataFrame] | None = None
    values: str | float = 'index'
    all_touched: bool = True
    dtype: DTypeLike | None = None
    fill_value: float | None = None
    rasterize_kw: dict[str, Any] = field(default_factory=dict)
    raster_creation_options: RasterCreationOptions | None = None

    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None:
        df = gpd.read_file(self.path).to_crs(spec.crs).reset_index() if self.path is not None else self.load_df(spec)

        if df.crs != spec.crs:
            raise ValueError(f'GeoDataFrame CRS ({df.crs}) does not match target spec CRS ({spec.crs}).')

        if self.values == 'index':
            raster_values = df.index.values
        elif isinstance(self.values, (int, float)):
            raster_values = np.full(df.shape[0], fill_value=self.values)
        else:
            raster_values = df[self.values].values

        dtype = self.dtype if self.dtype is not None else df[self.column].dtype

        if not np.issubdtype(dtype, np.number):
            raise TypeError(f'dtype must be numeric, got {dtype}.')

        default_fill_value = np.nan if np.issubdtype(dtype, np.floating) else 0
        fill_value = self.fill_value if self.fill_value is not None else default_fill_value

        if fill_value in raster_values:
            raise ValueError(f'Fill value {fill_value} is present in the data. Overwrite with a different value.')

        raster = rasterize(
            ((geom, val) for geom, val in zip(df.geometry.values, raster_values)),
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
            nodata=fill_value,
            crs=spec.crs,
            transform=spec.transform,
            **raster_creation_options.to_dict(),
        ) as dst:
            dst.write(raster, 1)

    default_driver = RioXArrayDriver()
    ext = 'tif'
