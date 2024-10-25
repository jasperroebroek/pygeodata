from pathlib import Path
from typing import Union

import numpy as np
import rioxarray as rxr
import xarray as xr
import rasterio as rio
from rasterio.errors import CRSError, NotGeoreferencedWarning


def load_file(f: Union[str, Path], cache: bool = True, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    try:
        with rio.open(f) as fp:
            if np.issubdtype(fp.dtypes[0], np.floating):
                kwargs.update(mask_and_scale=True)
        da = rxr.open_rasterio(f, parse_coordinates=True, cache=cache, **kwargs)
    except NotGeoreferencedWarning:
        print(f)
        raise CRSError

    return da


def load_flat_file(f: Union[str, Path], cache: bool = True, name: str = 'data') -> xr.DataArray:
    da = load_file(f, cache=cache).sel(band=1).drop_vars('band')

    if isinstance(da, xr.Dataset):
        raise TypeError

    return da.rename(name)
