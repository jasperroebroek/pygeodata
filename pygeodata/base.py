from typing import Any

from pygeodata.config import get_config
from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec


def load(loader: DataLoader, spec: SpatialSpec | None = None) -> Any:
    spec = spec or get_config().spec
    if spec is None:
        raise ValueError('No spatial specification (spec) provided or present in config')
    return loader(spec)


def process(loader: DataLoader, spec: SpatialSpec | None = None) -> None:
    spec = spec or get_config().spec
    if spec is None:
        raise ValueError('No spatial specification (spec) provided or present in config')
    if loader.is_processed(spec):
        return
    loader.process(spec)
