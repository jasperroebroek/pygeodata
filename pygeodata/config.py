from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from pygeodata.types import SpatialSpec


@dataclass
class Config:
    path_data_processed: Path = Path('data_processed')
    num_threads: int = 1
    warp_mem_limit: int = 0  # GDAL default, indicates 64 MB
    spec: Optional[SpatialSpec] = None

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
    old_values = asdict(CONFIG)
    CONFIG.update(**overrides)
    try:
        yield CONFIG
    finally:
        CONFIG.update(**old_values)
