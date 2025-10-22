import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import rasterio as rio
import rasterio.warp
from numpy.typing import DTypeLike
from rasterio import CRS, RasterioIOError
from rasterio.enums import Resampling

from pygeodata.config import get_config
from pygeodata.options import RasterioProfile
from pygeodata.types import SpatialSpec


@dataclass
class Reprojector:
    """Reprojects raster data to GeoTIFF format.

    Parameters
    ----------
    src_path : str | Path
        Path to source raster file
    bands : int | Sequence[int], optional
        Band indices to reproject (1-indexed). If None, reprojects all bands
    src_crs : CRS, optional
        Override source CRS if not defined in file
    resampling : Resampling, default=Resampling.nearest
        Resampling method
    dst_dtype : DTypeLike, optional
        Output data type. If None, uses source dtype
    dst_nodata : float, optional
        Output nodata value. If None, uses source nodata
    profile : RasterioProfile, optional
        GeoTIFF creation profile options. If None, uses defaults
    warp_kw : dict, optional
        Additional keyword arguments for rasterio.warp.reproject
    scale : float or sequence of floats, optional
        Scale factor for each band
    offset : float or sequence of floats, optional
        Offset for each band
    """

    src_path: str | Path
    src_crs: Optional[CRS] = None
    bands: Optional[int | Sequence[int]] = None
    resampling: Resampling = Resampling.nearest
    dst_dtype: Optional[DTypeLike] = None
    dst_nodata: Optional[float] = None
    warp_kw: dict[str, Any] = field(default_factory=dict)
    warp_mem_limit: Optional[int] = None
    num_threads: Optional[int] = None
    profile: RasterioProfile = field(default_factory=RasterioProfile)
    scale: Optional[float | Sequence[float]] = None
    offset: Optional[float | Sequence[float]] = None

    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None:
        """Reproject raster to specified spatial configuration.

        Parameters
        ----------
        dst_path : str | Path
            Destination file path
        spec : SpatialSpec
            Target spatial specification (CRS, transform, shape)
        """
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
                        f'Cannot reproject multi-variable dataset: {self.src_path}\nSubdatasets:\n{sub_str}'
                    )

                src_crs = src.crs if src.crs is not None else self.src_crs

                if src_crs is None:
                    raise ValueError(f'Cannot determine CRS for {self.src_path}. Provide src_crs parameter.')

                src_dtype = src.dtypes[0]
                src_transform = src.transform

                src_nodata = src.nodata or (np.nan if np.issubdtype(src_dtype, np.floating) else 0)

                src_bands = src.indexes if self.bands is None else self.bands

                dst_dtype = src_dtype if self.dst_dtype is None else self.dst_dtype
                dst_nodata = src_nodata if self.dst_nodata is None else self.dst_nodata

                rio_profile = {
                    'driver': 'GTiff',
                    'height': spec.shape[0],
                    'width': spec.shape[1],
                    'dtype': dst_dtype,
                    'nodata': dst_nodata,
                    'count': len(src_bands),
                    'crs': spec.crs,
                    'transform': spec.transform,
                    **self.profile.to_dict(),
                }

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
                        warp_mem_limit=self.warp_mem_limit or get_config().warp_mem_limit,
                        num_threads=self.num_threads or get_config().num_threads,
                        **self.warp_kw,
                    )

                    if self.scale is not None:
                        scales = self.scale if isinstance(self.scale, Sequence) else [self.scale] * len(src_bands)
                        dst._set_all_scales(scales)

                    if self.offset is not None:
                        offsets = self.offset if isinstance(self.offset, Sequence) else [self.offset] * len(src_bands)
                        dst._set_all_offsets(offsets)

            shutil.move(temp_path, dst_path)
