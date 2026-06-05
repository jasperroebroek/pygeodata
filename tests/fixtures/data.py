from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.processors.reprojection import Reprojector

TEST_DATA_DIR = Path(__file__).parent.parent.parent / 'data'
WTD_TIF = TEST_DATA_DIR / 'wtd.tif'
LUH2_NC = TEST_DATA_DIR / 'luh2.nc'
COUNTRIES_SHP = TEST_DATA_DIR / 'countries' / 'ne_110m_admin_0_map_units.shp'


class DummyScenario(Enum):
    SSP126 = 'ssp126'
    SSP585 = 'ssp585'


@dataclass
class LoaderA(Data):
    year: int
    ext = 'tif'


@dataclass
class LoaderB(Data):
    features: list[Data]
    ext = 'tif'
    _sort_params = ('features',)


@dataclass
class LoaderC(Data):
    target: str
    n_jobs: int = 4
    ext = 'tif'
    _private_state: bool = False
    _exclude_params = ('n_jobs',)


@dataclass
class LoaderD(Data):
    items: list
    ext = 'tif'


class EmptyLoader(Data):
    pass


class SimpleLoader(Data):
    ext = 'tif'
    processor = Reprojector(src_path=WTD_TIF)


class XMLHTTPLoader(Data):
    ext = 'xml'


class USGSElevationLoader(Data):
    ext = 'tif'


@dataclass
class SampleLoader(Data):
    path: Path
    scale: float = 1
    ext = 'tif'

    @property
    def processor(self) -> Reprojector:
        return Reprojector(self.path)

    driver = RioXArrayDriver()


class NestedLoader(Data):
    ext = 'tif'

    def __init__(self, inner: Data, tag: str = 'default'):
        self.inner = inner
        self.tag = tag


class HardcodedDependencyLoader(Data):
    ext = 'tif'

    def random_helper_method(self) -> Data:
        return LoaderA(year=2020)


@dataclass
class MultiOutputLoader(Data):
    loader_1: Data
    loader_2: Data

    _calls = []
    ext = 'tif'

    def _process(self, spec):
        self._calls.append(spec)
        yield self.loader_1
        yield self.loader_2


class Parent(Data):
    ext = 'tif'


class Child(Parent):
    pass


@dataclass
class DictLoader(Data):
    mapping: dict


@dataclass
class EnumLoader(Data):
    ext = 'tif'
    scenario: DummyScenario


class DummyProcessor:
    _calls: list
    default_driver = None
    ext = 'tif'

    def __new__(self):
        self._calls = []
        return super().__new__(self)

    def __call__(self, path, spec):
        self._calls.append((path, spec))


class DummyLoader(Data):
    processor: DummyProcessor = None

    def __init__(self):
        self.processor = DummyProcessor()


class CircularLoader(Data):
    ext = 'tif'

    def _process(self, spec):
        return CircularLoader()
