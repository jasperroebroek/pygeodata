from pathlib import Path

import xarray as xr
from affine import Affine
from numpy.typing import DTypeLike
from rasterio import CRS
from rasterio.enums import Resampling

from data_framework.data_entry import RasterDataEntry
from data_framework.xarray import load_file


def reproject(da: xr.DataArray, path: Path, crs: CRS, transform: Affine, dtype: DTypeLike,
              resampling: Resampling = Resampling.nearest, **resampling_kwargs) -> None:
    """da is reprojected and stored in path"""
    if path.exists():
        return

    print(f"Reprojecting: da -> {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    da_reprojected = da.rio.reproject(dst_crs=crs, transform=transform, resampling=resampling, **resampling_kwargs)
    da_reprojected.attrs.pop('long_name', None)

    da_reprojected.rio.to_raster(path, dtype=dtype, compress='DEFLATE')

    da_reprojected.close()


def reproject_data_entry(de: RasterDataEntry, crs: CRS, transform: Affine) -> None:
    path = de.get_path()
    if path is None:
        raise ValueError(f"Can't process without a path being specified: {de.name}")

    output_path = de.get_reprojected_path(crs=crs, transform=transform)

    da = load_file(path, cache=False)

    if isinstance(de.subset, (str, int)):
        da = da.sel(band=de.subset)
    elif isinstance(de.subset, dict):
        da = da.sel(**de.subset)

    reproject(da, output_path, crs=crs, transform=transform, dtype=de.dtype, resampling=de.resampling,
              **de.resampling_kwargs)
    da.close()
