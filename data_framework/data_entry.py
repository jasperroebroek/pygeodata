import gc
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Callable

import numpy as np
import rasterio as rio
from affine import Affine
from numpy._typing import DTypeLike
from rasterio import CRS
from rasterio.enums import Resampling

from data_framework.paths import path_data_reprojected
from data_framework.utils import transform_to_str


@dataclass
class RasterDataEntry:
    """class that holds the info to deal with any 2D raster data"""
    name: str = "data"
    caller_name: Optional[str] = None
    subset: Optional[Union[str, int, dict]] = None
    path: Optional[Path] = None
    path_reprojected: Optional[Path] = None
    resampling: Resampling = Resampling.nearest
    resampling_kwargs: dict = field(default_factory=dict)
    dtype: Optional[DTypeLike] = np.float32
    generate_func: Optional[Callable[['RasterDataEntry'], None]] = None

    def convert_path(self, crs: CRS, transform: Affine) -> Path:
        """Function that converts a path of the data to the reprojected data.
        NOTE: This is not always guaranteed to work"""
        return Path(path_data_reprojected,
                    f"{crs.to_string().replace(':', '_')}",
                    transform_to_str(transform),
                    self.caller_name,
                    f"{self.name}.tif")

    def get_path(self) -> Optional[Path]:
        return self.path

    def get_reprojected_path(self, crs: CRS, transform: Affine) -> Path:
        if self.path_reprojected is None:
            path = self.get_path()
            if path is None:
                raise ValueError("Neither path nor path_reprojected specified")
            if self.caller_name is None:
                raise ValueError("No caller name specified")
            return self.convert_path(crs, transform)

        return self.path_reprojected

    def is_reprojected(self, crs: CRS, transform: Affine) -> bool:
        p = self.get_reprojected_path(crs=crs, transform=transform)
        return p.exists() or p.is_symlink()

    def is_generated(self):
        return self.path.exists()

    def reproject(self, crs: Optional[CRS], transform: Optional[Affine]) -> None:
        from data_framework.reproject import reproject_data_entry

        path = self.get_path()
        reprojected_path = self.get_reprojected_path(crs, transform)
        reprojected_path.parent.mkdir(exist_ok=True, parents=True)

        fp = rio.open(path)

        if crs == fp.crs and transform == fp.transform:
            os.symlink(path, reprojected_path)
        else:
            reproject_data_entry(self, crs=crs, transform=transform)

        fp.close()
        gc.collect()

    def generate(self) -> None:
        if self.generate_func is None:
            raise NotImplementedError

        print(f"generating {self.name}")
        self.generate_func(self)
