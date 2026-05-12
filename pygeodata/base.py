from pygeodata.config import get_config
from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec, T


def process(loader: DataLoader, spec: SpatialSpec | None = None) -> SpatialSpec:
    spec = spec or get_config().spec
    if spec is None:
        raise ValueError('No spatial specification (spec) provided')
    spec = loader.resolve_spec(spec)

    if not loader.is_processed(spec):
        loader.process(spec)

    return spec


def load(loader: DataLoader[T], spec: SpatialSpec | None = None) -> T:
    spec = process(loader, spec)
    return loader.load(spec)
