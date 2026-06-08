from dataclasses import dataclass
from pathlib import Path

from pygeodata.config import get_config

CACHE_META_SUFFIXES = frozenset({'.params.json', '.hash.json', '.spec.json', '.graph.pdf', '.graph.png'})
CACHE_DIR_SUFFIXES = frozenset({'.zarr'})


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


@dataclass
class CodeRegistryResolver:
    directory: Path

    @classmethod
    def from_source_hash(cls, source_hash: str) -> 'CodeRegistryResolver':
        return cls(Path(get_config().path_registry) / 'code' / source_hash)

    @property
    def source_path(self) -> Path:
        return self.directory / 'source.py'

    @property
    def meta_path(self) -> Path:
        return self.directory / 'source.json'

    def exists(self) -> bool:
        return self.directory.exists()


@dataclass
class TreeRegistryResolver:
    directory: Path

    @classmethod
    def from_dep_tree_hash(cls, dep_tree_hash: str) -> 'TreeRegistryResolver':
        return cls(Path(get_config().path_registry) / 'snapshots' / dep_tree_hash)

    @property
    def tree_path(self) -> Path:
        return self.directory / 'tree.json'

    @property
    def graph_path(self) -> Path:
        return self.directory / 'graph.pdf'

    def exists(self) -> bool:
        return self.directory.exists()
