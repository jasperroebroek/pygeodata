from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pygeodata.config import get_config
from pygeodata.extraction import flatten_parameter_dict_for_path
from pygeodata.hash import calculate_dict_hash
from pygeodata.types import Shape, SpatialSpec

ARTIFACT_SUFFIXES = ('.hash.json', '.params.json', '.graph.png')


def generate_path(
    base_dir: str | Path,
    spec: SpatialSpec | None = None,
    name: str | None = None,
    max_path_param_depth: int | None = None,
    **kwargs,
) -> Path:
    """Function that converts a path of the data to the processed data.

    kwargs are expected to be parsed for the filesystem.
    """
    config = get_config()
    es = config.es
    format_fn = config.format_path_fn
    max_depth = max_path_param_depth if max_path_param_depth is not None else config.max_path_param_depth

    base_dir = Path(base_dir)

    flat_kwargs = flatten_parameter_dict_for_path(kwargs)

    params = []
    if kwargs:
        params = (
            [calculate_dict_hash(flat_kwargs)]
            if len(flat_kwargs) > max_depth
            else [f'{k}{es}{v}' for k, v in flat_kwargs.items()]
        )

    if spec is None:
        geo_str = '*'
    elif not spec.is_fully_defined:
        geo_str = 'vector'
    else:
        geo_str = f'{format_fn(Shape(spec.shape))}_{format_fn(spec.transform)}'

    crs_str = '*' if spec is None else format_fn(spec.crs)

    parts = [crs_str, geo_str]
    if name is not None:
        parts.append(name)

    parts.extend(params)

    return Path(
        base_dir,
        *parts,
    )


@dataclass
class CachePathResolver:
    directory: Path
    stem: str
    ext: str

    @classmethod
    def from_path(cls, path: Path) -> 'CachePathResolver':
        stem = path.name.removeprefix('.').split('.')[0]
        ext = ''.join(path.suffixes)
        directory = path.parent

        return cls(
            directory=directory,
            stem=stem,
            ext=ext,
        )

    def get_processed_path(self, ext: str | None = None) -> Path:
        return self.directory / f'{self.stem}.{ext or self.ext}'

    @property
    def processed_path(self) -> Path:
        return self.directory / f'{self.stem}{self.ext}'

    @property
    def params_path(self) -> Path:
        return self.directory / f'.{self.stem}.params.json'

    @property
    def state_hash_path(self) -> Path:
        return self.directory / f'.{self.stem}.hash.json'

    @property
    def execution_graph_path(self) -> Path:
        return self.directory / f'.{self.stem}.graph.pdf'

    @property
    def spec_path(self) -> Path:
        return self.directory / f'.{self.stem}.spec.json'

    def mkdir(self) -> None:
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)

    def glob_cache(self) -> Iterable[Path]:
        return self.directory.glob(f'{self.stem}.*')


@dataclass
class RegistryPathResolver:
    registry_path: Path
    code_path: Path
    lock_path: Path
    graph_path: Path

    @classmethod
    def from_directory(cls, directory: Path) -> 'RegistryPathResolver':
        return cls(
            registry_path=directory / 'source.json',
            code_path=directory / 'code.py',
            lock_path=directory / 'lock.json',
            graph_path=directory / 'graph.pdf',
        )

    def mkdir(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
