from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec, T


def load(loader: DataLoader[T], spec: SpatialSpec | None = None) -> T:
    spec = spec or loader.get_default_spec()
    if spec is None:
        raise ValueError('No spatial specification (spec) provided or present in config')
    return loader(spec)


def process(loader: DataLoader, spec: SpatialSpec | None = None) -> None:
    spec = spec or load.get_default_spec()
    if spec is None:
        raise ValueError('No spatial specification (spec) provided or present in config')
    if loader.is_processed(spec):
        return
    loader.process(spec)
