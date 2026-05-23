from dataclasses import dataclass

from pygeodata.api import load
from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.processors.reprojection import Reprojector


@dataclass
class ElevationLoader(Data):
    """Reproject a raw DEM to the target spec."""

    src: str = 'data/elevation.tif'

    @property
    def processor(self):
        return Reprojector(src_path=self.src)


@dataclass
class Red(Data):
    year: int

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        return 0.05

    driver = RioXArrayDriver()


@dataclass
class NIR(Data):
    year: int

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        return 0.5

    driver = RioXArrayDriver()


@dataclass
class NDVI(Data):
    year: int

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        red = load(Red(year=self.year), spec)
        nir = load(NIR(year=self.year), spec)
        return (nir - red) / (nir + red)

    driver = RioXArrayDriver()


@dataclass
class NDVIInjection(Data):
    red: Red
    nir: NIR

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        red = load(self.red, spec)
        nir = load(self.nir, spec)
        return (nir - red) / (nir + red)

    driver = RioXArrayDriver()
