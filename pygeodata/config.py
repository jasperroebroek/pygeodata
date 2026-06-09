from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

from pygeodata.rasters import RasterCreationOptions
from pygeodata.types import SpatialSpec


@dataclass
class Config:
    path_cache: Path = Path('data_processed')
    path_figures: Path = Path('figures')
    path_registry: Path = Path('.source')
    num_threads: int = 1
    warp_mem_limit: int = 0  # GDAL default, indicates 64 MB
    spec: SpatialSpec | None = None
    raster_creation_options: RasterCreationOptions = field(default_factory=RasterCreationOptions)
    removable_system_files: tuple[str] = (
        '.DS_Store',
        'Thumbs.db',
        'desktop.ini',
        '.Trash-1000',
    )
    removable_system_suffixes: tuple[str] = (
        '.tmp',
        '~',
    )

    def update(self, **kwargs) -> None:
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


class JSONKeys(StrEnum):
    CLASS_NAME = 'class_name'
    OBJECT_TYPE = 'object_type'
    PARAMS = 'params'
    SOURCE_HASH = 'source_hash'
    PROCESSOR_HASH = 'processor_hash'
    INSTANCE_HASH = 'instance_hash'
    STATE_HASH = 'state_hash'
    DEPENDENCY_TREE_HASH = 'dependency_tree_hash'
    CO_OUTPUTS = 'co_outputs'
    NODES = 'nodes'
    TREE = 'tree'
    CALL_DEPENDENCIES = 'call_dependencies'
    INHERITANCE_DEPENDENCIES = 'inheritance_dependencies'
    DEPENDENCIES = 'dependencies'
    REGISTERED_AT = 'registered_at'
