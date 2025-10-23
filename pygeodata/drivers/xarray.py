from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
from rioxarray.exceptions import TooManyDimensions


@dataclass
class RioXArrayDriver:
    """Load a raster file using rioxarray.

    Parameters
    ----------
    parse_coordinates : bool, optional
        Whether to parse coordinates
    mask_and_scale : bool, optional
        Whether to mask and scale. By default, this parameter is None, which causes it to be inferred from the data
        type. Floating data will default to True, integer data to False.
    decode_times : bool, optional
        Whether to decode times, by default False
    cache : bool, optional
        Whether to cache, by default False
    open_kw : dict, optional
        Additional keyword arguments to pass to rioxarray.open_rasterio
    flatten : bool, optional
        By default 2D rasters will be returned as 3D, with a band dimension of size 1. If True, which is the
        default, this dimension is removed.
    """

    parse_coordinates: bool = True
    mask_and_scale: bool | None = None  # if None, inferred from data -> float yes, int no
    decode_times: bool = False
    cache: bool = False
    open_kw: dict = field(default_factory=dict)
    flatten: bool = True

    @staticmethod
    def _subdatasets(path: Path) -> list[str]:
        with rio.open(path) as fp:
            return fp.subdatasets

    def _has_subdatasets(self, path: Path) -> bool:
        return len(self._subdatasets(path)) > 0

    @staticmethod
    def _infer_mask_and_scale(path: Path) -> bool:
        with rio.open(path) as fp:
            return np.issubdtype(fp.dtypes[0], np.floating)

    @staticmethod
    def _assert_exists(path: Path) -> None:
        with rio.open(path):
            pass

    def load(self, path: str | Path) -> xr.DataArray:
        path = Path(path)

        self._assert_exists(path)

        if self._has_subdatasets(path):
            subs_str = '\n'.join(self._subdatasets(path))
            raise TooManyDimensions(f'Multiple variables found in {path}. Use one of the subdatasets:\n{subs_str}')

        mask_and_scale = self.mask_and_scale if self.mask_and_scale is not None else self._infer_mask_and_scale(path)

        try:
            da = rxr.open_rasterio(
                path,
                parse_coordinates=self.parse_coordinates,
                decode_times=self.decode_times,
                cache=self.cache,
                mask_and_scale=mask_and_scale,
                **self.open_kw,
            )
        except Exception as e:
            raise type(e)(f'Failed to load {path}: {e}') from e

        if isinstance(da, xr.Dataset):
            raise TooManyDimensions(f'Dataset found in {path}. Use one of the subdatasets.')

        if self.flatten and 'band' in da.dims and da.sizes['band'] == 1:
            da = da.isel(band=0).drop_vars('band')

        return da
