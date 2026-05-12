from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pygeodata.loader import DataLoader

TEST_DATA_DIR = Path(__file__).parent.parent / 'data'
WTD_TIF = TEST_DATA_DIR / 'wtd.tif'
LUH2_NC = TEST_DATA_DIR / 'luh2.nc'
COUNTRIES_SHP = TEST_DATA_DIR / 'countries' / 'ne_110m_admin_0_map_units.shp'


class DummyScenario(Enum):
    SSP126 = 'ssp126'
    SSP585 = 'ssp585'


@dataclass
class LoaderA(DataLoader):
    year: int


@dataclass
class LoaderB(DataLoader):
    features: list[DataLoader]
    _sort_params = ('features',)


@dataclass
class LoaderC(DataLoader):
    target: str
    n_jobs: int = 4
    _private_state: bool = False
    _exclude_params = ('n_jobs',)


@dataclass
class LoaderD(DataLoader):
    """Loader with a mixed list: a DataLoader AND a plain string."""

    items: list


@dataclass
class UpstreamLoader(DataLoader):
    """A simple parameterless upstream loader for DAG testing."""

    def process(self, spec):
        p = self.get_processed_path(spec)
        p.write_text('upstream')
        self.write_state_hash(spec)


@dataclass
class DownstreamLoader(DataLoader):
    """Uses UpstreamLoader as an explicit dependency for DAG testing."""

    upstream: DataLoader

    def process(self, spec):
        p = self.get_processed_path(spec)
        p.write_text('downstream')
        self.write_state_hash(spec)


class SimpleLoader(DataLoader):
    def __init__(self, path: str, scale: float = 1.0):
        self.path = path
        self.scale = scale


class NestedLoader(DataLoader):
    def __init__(self, inner: DataLoader, tag: str = 'default'):
        self.inner = inner
        self.tag = tag


class HardcodedDependencyLoader(DataLoader):
    """A loader that hides a dependency inside a random method."""

    def random_helper_method(self):
        hidden_loader = LoaderA(year=2020)
        return hidden_loader


def make_simple_loader(name: str, write_fn=None):
    """Creates a minimal DataLoader subclass with a controllable process() method."""

    @dataclass(repr=False)
    class _Loader(DataLoader):
        pass

    _Loader.__name__ = name
    _Loader.__qualname__ = name
    if write_fn:
        _Loader.process = lambda self, spec: write_fn(self, spec)
    return _Loader
