from pathlib import Path
from typing import Union

import rioxarray as rxr
import xarray as xr
from rasterio.errors import NotGeoreferencedWarning, CRSError


def load_file(f: Union[str, Path], cache: bool = True, **kwargs) -> Union[xr.DataArray, xr.Dataset]:
    try:
        da = rxr.open_rasterio(f, masked=True, parse_coordinates=True, cache=cache, **kwargs)
    except NotGeoreferencedWarning:
        print(f)
        raise CRSError

    return da


def load_flat_file(f: Union[str, Path], cache: bool = True, name: str = 'data') -> xr.DataArray:
    da = load_file(f, cache=cache).sel(band=1).drop_vars('band')

    if isinstance(da, xr.Dataset):
        raise TypeError

    return da.rename(name)
