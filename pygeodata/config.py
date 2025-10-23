from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from pygeodata.options import RasterCreationOptions
from pygeodata.types import SpatialSpec


@dataclass
class Config:
    path_data_processed: Path = Path('data_processed')
    num_threads: int = 1
    warp_mem_limit: int = 0  # GDAL default, indicates 64 MB
    spec: SpatialSpec | None = None
    raster_creation_options: RasterCreationOptions = field(default_factory=RasterCreationOptions)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f'Invalid config key: {key}')
            setattr(self, key, value)


CONFIG = Config()


def get_config() -> Config:
    return CONFIG


@contextmanager
def set_config(**overrides: Any) -> Iterator[Config]:
    old_values = {k.name: getattr(CONFIG, k.name) for k in fields(CONFIG)}
    CONFIG.update(**overrides)
    try:
        yield CONFIG
    finally:
        CONFIG.update(**old_values)
