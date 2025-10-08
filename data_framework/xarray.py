from pathlib import Path
from typing import Union

import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr


def load_file(f: Union[str, Path], cache: bool = True, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    try:
        with rio.open(f) as fp:
            if np.issubdtype(fp.dtypes[0], np.floating):
                kwargs.update(mask_and_scale=True)
            window = rio.windows.Window(0, 0, 1, 1)
            fp.read(1, window=window)
        da = rxr.open_rasterio(f, parse_coordinates=True, cache=cache, **kwargs)
    except Exception as e:
        print(f'path={f}')
        raise e

    return da


def load_flat_file(f: Union[str, Path], cache: bool = True, name: str = 'data') -> xr.DataArray:
    da = load_file(f, cache=cache).sel(band=1).drop_vars('band')

    if isinstance(da, xr.Dataset):
        raise TypeError

    return da.rename(name)
