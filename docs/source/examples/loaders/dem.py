from dataclasses import dataclass

import numpy as np
from pygeodata.api import load
from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.processors.reprojection import Reprojector
from pygeodata.spec import SpatialSpec
from rasterio.enums import Resampling


@dataclass
class ElevationLoader(Data):
    """Reproject the source DEM to the target spec."""

    src: str = 'data/srtm_30m.tif'

    @property
    def processor(self):
        return Reprojector(src_path=self.src, resampling=Resampling.bilinear, dst_dtype=np.float32)


@dataclass
class SlopeLoader(Data):
    """Terrain slope derived from ElevationLoader, in degrees."""

    elevation: ElevationLoader = None

    ext = 'tif'
    driver = RioXArrayDriver()

    def __post_init__(self):
        if self.elevation is None:
            self.elevation = ElevationLoader()

    def _process(self, spec: SpatialSpec) -> None:
        import xrspatial

        dem = load(self.elevation, spec)
        slope = xrspatial.slope(dem)
        slope.rio.to_raster(self.get_processed_path(spec))
