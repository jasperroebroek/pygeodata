from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pygeodata.config import JSONKeys, get_config
from pygeodata.extraction import flatten_parameter_dict_for_path
from pygeodata.hash import calculate_dict_hash
from pygeodata.types import Shape, SpatialSpec, SpecKeys

ARTIFACT_SUFFIXES = ('.hash.json', '.params.json', '.graph.png')


def generate_path(
    base_dir: str | Path,
    spec: SpatialSpec,
    name: str | None = None,
    max_path_param_depth: int | None = None,
    **kwargs,
) -> Path:
    """Return the cache directory for an artifact given its spec, name, and params.

    When ``human_readable_paths`` is False (the default), the layout is:
        base_dir / name / hash(spec + params)

    When ``human_readable_paths`` is True, the layout is:
        base_dir / crs / shape_transform / name / param=val ...
    """
    config = get_config()
    base_dir = Path(base_dir)

    flat_kwargs = flatten_parameter_dict_for_path(kwargs)

    if not config.human_readable_paths:
        parts: list[str] = []
        if name is not None:
            parts.append(name)
        hash_input = {
            SpecKeys.SPEC: spec.to_dict(),
            JSONKeys.PARAMS: flat_kwargs,
        }
        parts.append(calculate_dict_hash(hash_input))
        return Path(base_dir, *parts)

    # Human-readable layout
    es = config.es
    format_fn = config.format_path_fn
    max_depth = max_path_param_depth if max_path_param_depth is not None else config.max_path_param_depth

    params: list[str] = []
    if kwargs:
        params = (
            [calculate_dict_hash(flat_kwargs)]
            if len(flat_kwargs) > max_depth
            else [f'{k}{es}{v}' for k, v in flat_kwargs.items()]
        )

    geo_str = 'vector' if not spec.is_fully_defined else f'{format_fn(Shape(spec.shape))}_{format_fn(spec.transform)}'

    crs_str = format_fn(spec.crs)

    path_parts = [crs_str, geo_str]
    if name is not None:
        path_parts.append(name)
    path_parts.extend(params)

    return Path(base_dir, *path_parts)


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
            code_path=directory / 'source.py',
            lock_path=directory / 'source.lock',
            graph_path=directory / 'source.pdf',
        )

    def mkdir(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
