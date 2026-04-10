import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
import rasterio.warp
from numpy.typing import DTypeLike
from rasterio import CRS, RasterioIOError
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform

from pygeodata.config import get_config
from pygeodata.drivers import RioXArrayDriver
from pygeodata.options import RasterCreationOptions
from pygeodata.types import SpatialSpec


@dataclass
class Reprojector:
    """Reprojects raster data to GeoTIFF format.

    Parameters
    ----------
    src_path : str | Path
        Path to source raster file
    bands : int or tuple of int, optional
        Band indices to reproject (1-indexed). If None, reprojects all bands
    src_crs : CRS, optional
        Override source CRS if not defined in file
    src_nodata : float, optional
        Override source nodata value if not defined in file
    resampling : Resampling, default=Resampling.nearest
        Resampling method
    dst_dtype : DTypeLike, optional
        Output data type. If None, uses source dtype
    dst_nodata : float, optional
        Output nodata value. If None, uses source nodata
    nbits : int, optional
        Number of bits per pixel
    warp_kw : dict, optional
        Additional keyword arguments for rasterio.warp.reproject
    scales : float or sequence of floats, optional
        Scale factor for each band
    offsets : float or sequence of floats, optional
        Offset for each band
    raster_creation_options : RasterCreationOptions, optional
        GeoTIFF creation profile options. If None, uses defaults
    """

    src_path: str | Path
    bands: int | Sequence[int] | None = None
    src_crs: CRS | None = None
    src_nodata: float | None = None
    resampling: Resampling = Resampling.nearest
    dst_dtype: DTypeLike | None = None
    dst_nodata: float | None = None
    nbits: int | None = None
    warp_kw: dict[str, Any] = field(default_factory=dict)
    warp_mem_limit: int | None = None
    num_threads: int | None = None
    scales: float | Sequence[float] | None = None
    offsets: float | Sequence[float] | None = None
    raster_creation_options: RasterCreationOptions | None = None

    def __post_init__(self):
        if self.dst_dtype == np.bool_:
            self.dst_dtype = 'uint8'
            self.nbits = 1

    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None:
        """Reproject raster to specified spatial configuration.

        Parameters
        ----------
        dst_path : str | Path
            Destination file path
        spec : SpatialSpec
            Target spatial specification (CRS, transform, shape)
        """
        if not spec.is_fully_defined:
            raise ValueError('Shape and transform must be defined in spec for reprojection.')

        dst_path = Path(dst_path)
        if dst_path.exists():
            raise FileExistsError(f'Destination already exists: {dst_path}')

        print(f'Reprojecting: {self.src_path} -> {dst_path}')

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / f'~{"_".join(dst_path.parts[1:])}'

            with rio.open(self.src_path) as src:
                if len(src.subdatasets) > 1:
                    sub_str = '\n'.join(src.subdatasets)
                    raise RasterioIOError(
                        f'Cannot reproject multi-variable dataset: {self.src_path}\nSubdatasets:\n{sub_str}',
                    )

                src_crs = src.crs if src.crs is not None else self.src_crs

                if src_crs is None:
                    raise ValueError(f'Cannot determine CRS for {self.src_path}. Provide src_crs parameter.')

                src_dtype = src.dtypes[0]
                src_transform = src.transform

                src_nodata = self.src_nodata if self.src_nodata is not None else src.nodata
                if src_nodata is None and np.issubdtype(src_dtype, np.floating):
                    src_nodata = np.nan

                src_bands = src.indexes if self.bands is None else self.bands
                if isinstance(src_bands, Number):
                    src_bands = (src_bands,)
                count = len(src_bands)

                dst_dtype = src_dtype if self.dst_dtype is None else self.dst_dtype
                dst_nodata = src_nodata if self.dst_nodata is None else self.dst_nodata

                raster_creation_options = self.raster_creation_options or get_config().raster_creation_options

                rio_profile = {
                    'driver': 'GTiff',
                    'height': spec.shape[0],
                    'width': spec.shape[1],
                    'dtype': dst_dtype,
                    'nodata': dst_nodata,
                    'count': count,
                    'crs': spec.crs,
                    'transform': spec.transform,
                    **raster_creation_options.to_dict(),
                }

                if self.nbits is not None:
                    rio_profile['nbits'] = self.nbits

                warp_mem_limit = self.warp_mem_limit if self.warp_mem_limit is not None else get_config().warp_mem_limit
                num_threads = self.num_threads if self.num_threads is not None else get_config().num_threads

                with rio.open(temp_path, 'w', **rio_profile) as dst:
                    rasterio.warp.reproject(
                        source=rio.band(src, src_bands),
                        destination=rio.band(dst, dst.indexes),
                        src_crs=src_crs,
                        dst_crs=spec.crs,
                        src_transform=src_transform,
                        dst_transform=spec.transform,
                        src_nodata=src_nodata,
                        dst_nodata=dst_nodata,
                        resampling=self.resampling,
                        warp_mem_limit=warp_mem_limit,
                        num_threads=num_threads,
                        **self.warp_kw,
                    )

                    scales = self.scales if self.scales is not None else tuple(src.scales[i - 1] for i in src_bands)
                    offsets = self.offsets if self.offsets is not None else tuple(src.offsets[i - 1] for i in src_bands)

                    if scales is not None:
                        scales = scales if isinstance(scales, Sequence) else [scales] * dst.count
                        dst._set_all_scales(scales)  # noqa: SLF001

                    if offsets is not None:
                        offsets = offsets if isinstance(offsets, Sequence) else [offsets] * dst.count
                        dst._set_all_offsets(offsets)  # noqa: SLF001

            shutil.move(temp_path, dst_path)

    def resolve_spec(self, spec: SpatialSpec) -> SpatialSpec:
        if spec.is_fully_defined:
            return spec

        with rio.open(self.src_path) as src:
            src_crs = src.crs if src.crs is not None else self.src_crs
            src_transform = src.transform
            src_shape = src.shape
            src_bounds = src.bounds

        if src_crs is None:
            raise ValueError(f'Cannot determine CRS for {self.src_path}. Provide src_crs parameter.')

        bounds_dict = {
            'left': src_bounds.left,
            'right': src_bounds.right,
            'top': src_bounds.top,
            'bottom': src_bounds.bottom,
        }

        transform, width, height = calculate_default_transform(
            src_crs=src_crs,
            dst_crs=spec.crs,
            width=src_shape[1],
            height=src_shape[0],
            transform=src_transform,
            **bounds_dict,
        )

        return SpatialSpec(crs=spec.crs, shape=(height, width), transform=transform)

    default_driver = RioXArrayDriver()
    ext = 'tif'
