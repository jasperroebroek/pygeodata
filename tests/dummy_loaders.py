from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.figure import Figure
from pygeodata.processors.reprojection import Reprojector
from pygeodata.types import SpatialSpec

TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
WTD_TIF = TEST_DATA_DIR / 'wtd.tif'
LUH2_NC = TEST_DATA_DIR / 'luh2.nc'
COUNTRIES_SHP = TEST_DATA_DIR / 'countries' / 'ne_110m_admin_0_map_units.shp'


class DummyScenario(Enum):
    SSP126 = 'ssp126'
    SSP585 = 'ssp585'


@dataclass
class LoaderA(Data):
    year: int


@dataclass
class LoaderB(Data):
    features: list[Data]
    _sort_params = ('features',)


@dataclass
class LoaderC(Data):
    target: str
    n_jobs: int = 4
    _private_state: bool = False
    _exclude_params = ('n_jobs',)


@dataclass
class LoaderD(Data):
    """Loader with a mixed list: a Data AND a plain string."""

    items: list


class EmptyLoader(Data):
    driver = RioXArrayDriver()


@dataclass
class SampleLoader(Data):
    path: Path
    scale: float = 1

    @property
    def processor(self) -> Reprojector:
        return Reprojector(self.path)

    driver = RioXArrayDriver()


class NestedLoader(Data):
    def __init__(self, inner: Data, tag: str = 'default'):
        self.inner = inner
        self.tag = tag


class HardcodedDependencyLoader(Data):
    def random_helper_method(self) -> Data:
        return LoaderA(year=2020)


@dataclass
class MultiOutputLoader(Data):
    loader_1: Data
    loader_2: Data
    ext = 'tif'
    _calls = []

    def _process(self, spec):
        self._calls.append(spec)
        yield self.loader_1
        yield self.loader_2


@dataclass
class SimpleFigure(Figure):
    a: int = 1


@dataclass
class TwoParamFigure(Figure):
    a: int
    b: str


@dataclass
class DummyFigure(Figure):
    a: int

    def _process(self, spec: SpatialSpec) -> None:
        out = self.get_processed_path(spec)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
