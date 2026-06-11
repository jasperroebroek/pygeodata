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
from pygeodata.rasters import RasterCreationOptions
from pygeodata.spec import SpatialSpec


@dataclass
class Rasterizer:
    """
    Rasterize a vector dataset to a single-band raster.

    Parameters
    ----------
    src_path : Path, optional
        Path to the vector dataset (e.g., shapefile, GeoPackage).
    load_df : Callable[[SpatialSpec], gpd.GeoDataFrame], optional
        Function to load the vector data. ``path`` is preferred if provided.
    values : str or int, default='index'
        Values imprinted on the raster. A numeric value will be used for all polygons, while a string will be
        interpreted as the name of the column in the data with the values. A special value is `index`, which refers to
        the index of the dataframe.
    column : str, optional
        Column to use for dtype if not provided.
    all_touched : bool, default=True
        Whether to burn all pixels touched by geometries.
    dtype : np.dtype, optional
        Data type for raster. Defaults to the dtype of `column`.
    fill_value : float, optional
        Fill value for raster. Defaults to 0 for integer dtypes and np.nan for floating point dtypes.
    nodata_value : float, optional
        Nodata value for raster.
    rasterize_kw : dict, optional
        Additional keyword arguments passed to `rasterio.features.rasterize`.
    raster_creation_options : RasterCreationOptions, optional
        Optional raster creation profile (compression, tiling, etc.).
    """

    src_path: Path | None = None
    load_df: Callable[[SpatialSpec], gpd.GeoDataFrame] | None = None
    values: str | float = 'index'
    column: str | None = None
    all_touched: bool = True
    dtype: DTypeLike | None = None
    fill_value: float | None = None
    nodata_value: float | None = None
    rasterize_kw: dict[str, Any] = field(default_factory=dict)
    raster_creation_options: RasterCreationOptions | None = None

    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None:
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if not spec.is_fully_defined:
            raise ValueError('Shape and transform must be defined in spec for rasterization.')

        if self.src_path is not None:
            df = gpd.read_file(self.src_path).to_crs(spec.crs).reset_index()
        elif self.load_df is not None:
            df = self.load_df(spec)
        else:
            raise ValueError('Either src_path or load_df must be provided.')

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
            ((geom, val) for geom, val in zip(df.geometry.values, raster_values, strict=True)),
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
            nodata=self.nodata_value,
            crs=spec.crs,
            transform=spec.transform,
            **raster_creation_options.to_dict(),
        ) as dst:
            dst.write(raster, 1)

    default_driver = RioXArrayDriver()
    ext = 'tif'
