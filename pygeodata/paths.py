from dataclasses import dataclass
from pathlib import Path

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
