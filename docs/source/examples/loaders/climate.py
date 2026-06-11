from dataclasses import dataclass

from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.processors.reprojection import Reprojector
from pygeodata.spec import SpatialSpec


@dataclass
class LSTLoader(Data):
    """Land surface temperature for a given year, reprojected to spec."""

    year: int = 2020
    src_dir: str = 'data/lst'

    @property
    def processor(self):
        return Reprojector(src_path=f'{self.src_dir}/lst_{self.year}.tif')

    driver = RioXArrayDriver()


@dataclass
class NDVILoader(Data):
    """NDVI composite for a given year."""

    year: int = 2020
    src_dir: str = 'data/ndvi'

    @property
    def processor(self):
        return Reprojector(src_path=f'{self.src_dir}/ndvi_{self.year}.tif')

    driver = RioXArrayDriver()
