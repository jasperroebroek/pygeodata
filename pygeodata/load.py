from typing import Any, Optional

from pygeodata.config import get_config
from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec


def load_data(loader: DataLoader, spec: Optional[SpatialSpec] = None) -> Any:
    spec = spec or get_config().spec
    if spec is None:
        raise ValueError('No spatial specification (spec) provided or present in config')
    return loader.load(spec)
