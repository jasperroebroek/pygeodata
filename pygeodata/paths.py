from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

from pygeodata.config import get_config

CACHE_META_FILES = frozenset({'parameters.json', 'meta.json', 'spec.json', 'graph.pdf', 'graph.png', 'process.lock'})
CACHE_DIR_SUFFIXES = frozenset({'.zarr'})


@dataclass
class CachePathConstructor:
    directory: Path

    @classmethod
    def from_state_hash(cls, state_hash: str, root: Path) -> CachePathConstructor:
        return cls(root / state_hash)

    @classmethod
    def from_path(cls, path: Path) -> CachePathConstructor:
        return cls(path.parent)

    @property
    def params_path(self) -> Path:
        return self.directory / 'parameters.json'

    @property
    def state_hash_path(self) -> Path:
        return self.directory / 'meta.json'

    @property
    def execution_graph_path(self) -> Path:
        return self.directory / 'graph.pdf'

    @property
    def spec_path(self) -> Path:
        return self.directory / 'spec.json'

    def mkdir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def iterdir(self) -> Generator[Path, None, None]:
        for path in self.directory.iterdir():
            if path.name not in CACHE_META_FILES:
                yield path


@dataclass
class CachePathResolver:
    roots: tuple[Path] | None = None

    def glob_meta_paths(self) -> Generator[Path, None, None]:
        roots = (get_config().path_registry, get_config().path_figures) if self.roots is None else self.roots
        for root in roots:
            if root.exists():
                yield from root.rglob('meta.json')


@dataclass
class CodeRegistryPathConstructor:
    directory: Path

    @classmethod
    def from_source_hash(cls, source_hash: str, registry_root: Path | None = None) -> CodeRegistryPathConstructor:
        base = registry_root if registry_root is not None else get_config().path_registry
        return cls(base / 'code' / source_hash)

    @property
    def source_path(self) -> Path:
        return self.directory / 'source.py'

    @property
    def meta_path(self) -> Path:
        return self.directory / 'source.json'

    def exists(self) -> bool:
        return self.directory.exists()


@dataclass
class TreeRegistryPathConstructor:
    directory: Path

    @classmethod
    def from_dep_tree_hash(cls, dep_tree_hash: str, registry_root: Path | None = None) -> TreeRegistryPathConstructor:
        base = registry_root if registry_root is not None else get_config().path_registry
        return cls(base / 'snapshots' / dep_tree_hash)

    @property
    def tree_path(self) -> Path:
        return self.directory / 'tree.json'

    @property
    def graph_path(self) -> Path:
        return self.directory / 'graph.pdf'

    def exists(self) -> bool:
        return self.directory.exists()


@dataclass
class RegistryResolver:
    root: Path | None = None

    def _root(self) -> Path:
        return self.root if self.root is not None else get_config().path_registry

    def glob_source_paths(self) -> Generator[Path, None, None]:
        directory = self._root() / 'code'
        if not directory.exists():
            return
        yield from directory.rglob('source.json')

    def glob_tree_paths(self) -> Generator[Path, None, None]:
        directory = self._root() / 'snapshots'
        if not directory.exists():
            return
        yield from directory.rglob('tree.json')


def classify_file(path: Path) -> str:
    """Return a broad kind string for the given file path based on its extension."""
    suffix = path.suffix.lower()

    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'}:
        return 'image'
    if suffix == '.pdf':
        return 'pdf'
    if suffix in {'.tif', '.tiff', '.nc', '.vrt', '.npy', '.zarr'}:
        return 'raster'
    if suffix == '.json':
        return 'json'
    if suffix in {'.py', '.pyx', '.ipynb'}:
        return 'code'
    return 'file'
