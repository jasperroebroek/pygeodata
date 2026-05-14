from dataclasses import dataclass

from pygeodata.base import load
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.loader import DataLoader
from pygeodata.processors.reprojection import Reprojector


@dataclass
class ElevationLoader(DataLoader):
    """Reproject a raw DEM to the target spec."""

    src: str = 'data/elevation.tif'

    @property
    def processor(self):
        return Reprojector(src_path=self.src)


@dataclass
class Red(DataLoader):
    year: int

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        return 0.05

    driver = RioXArrayDriver()


@dataclass
class NIR(DataLoader):
    year: int

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        return 0.5

    driver = RioXArrayDriver()


@dataclass
class NDVI(DataLoader):
    year: int

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        red = load(Red(year=self.year), spec)
        nir = load(NIR(year=self.year), spec)
        return (nir - red) / (nir + red)

    driver = RioXArrayDriver()


@dataclass
class NDVIInjection(DataLoader):
    red: Red
    nir: NIR

    def _process(self, spec) -> None:
        return

    def load(self, spec):
        red = load(self.red, spec)
        nir = load(self.nir, spec)
        return (nir - red) / (nir + red)

    driver = RioXArrayDriver()
