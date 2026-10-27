from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('pygeodata')
except PackageNotFoundError:  # imported from a source tree that is not installed
    __version__ = '0+unknown'

from pygeodata.api import load, load_from_hash, process
from pygeodata.artifact import Artifact
from pygeodata.cache import clean_cache, clean_registry
from pygeodata.config import get_config, set_config
from pygeodata.data import Data
from pygeodata.figure import Figure
from pygeodata.spec import SpatialSpec
from pygeodata.table import Table
from pygeodata.tracked_object import TrackedObject

__all__ = [
    'Artifact',
    'Data',
    'Figure',
    'SpatialSpec',
    'Table',
    'TrackedObject',
    '__version__',
    'clean_cache',
    'clean_registry',
    'get_config',
    'load',
    'load_from_hash',
    'process',
    'set_config',
]
