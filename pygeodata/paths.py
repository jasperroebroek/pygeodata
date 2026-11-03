from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pygeodata.config import get_config
from pygeodata.enums import CacheFile, CodeFile, RegistryDir, SnapshotFile

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


CACHE_META_FILES = frozenset(CacheFile)
ZARR_MARKERS = ('zarr.json', '.zgroup', '.zarray', '.zattrs', '.zmetadata')


def is_zarr_store(path: Path) -> bool:
    return path.suffix.lower() == '.zarr' or any((path / marker).exists() for marker in ZARR_MARKERS)


def is_output_path(path: Path) -> bool:
    """Whether ``path`` inside an entry dir is an output: not metadata, hidden or system junk."""
    config = get_config()
    name = path.name
    if (
        name in CACHE_META_FILES
        or name.startswith('.')
        or name in config.removable_system_files
        or name.endswith(config.removable_system_suffixes)
    ):
        return False
    return is_zarr_store(path) if path.is_dir() else path.is_file()


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
        return self.directory / CacheFile.PARAMS

    @property
    def state_hash_path(self) -> Path:
        return self.directory / CacheFile.META

    @property
    def execution_graph_path(self) -> Path:
        return self.directory / CacheFile.GRAPH

    @property
    def spec_path(self) -> Path:
        return self.directory / CacheFile.SPEC

    @property
    def lock_path(self) -> Path:
        return self.directory / CacheFile.LOCK

    def mkdir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def output_paths(self) -> list[Path]:
        return sorted(p for p in self.directory.iterdir() if is_output_path(p))


@dataclass
class CachePathResolver:
    roots: tuple[Path] | None = None

    def iter_entry_dirs(self) -> Generator[Path, None, None]:
        """Entry dirs are the direct child dirs of each root; a child that is itself a root is skipped."""
        roots = get_config().cache_roots if self.roots is None else self.roots
        resolved_roots = {root.resolve() for root in roots}
        for root in roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if child.is_dir() and child.resolve() not in resolved_roots:
                    yield child

    def glob_meta_paths(self) -> Generator[Path, None, None]:
        for entry_dir in self.iter_entry_dirs():
            meta_path = CachePathConstructor(entry_dir).state_hash_path
            if meta_path.is_file():
                yield meta_path


@dataclass
class RegistryResolver:
    root: Path | None = None

    @property
    def code_dir(self) -> Path:
        return self._root() / RegistryDir.CODE

    @property
    def snapshots_dir(self) -> Path:
        return self._root() / RegistryDir.SNAPSHOTS

    def _root(self) -> Path:
        return self.root if self.root is not None else get_config().path_registry

    def glob_source_paths(self) -> Generator[Path, None, None]:
        if self.code_dir.exists():
            yield from self.code_dir.rglob(CodeFile.META)

    def glob_tree_paths(self) -> Generator[Path, None, None]:
        if self.snapshots_dir.exists():
            yield from self.snapshots_dir.rglob(SnapshotFile.TREE)


@dataclass
class CodeRegistryPathConstructor:
    directory: Path

    @classmethod
    def from_source_hash(cls, source_hash: str, registry_root: Path | None = None) -> CodeRegistryPathConstructor:
        return cls(RegistryResolver(registry_root).code_dir / source_hash)

    @property
    def source_path(self) -> Path:
        return self.directory / CodeFile.SOURCE

    @property
    def meta_path(self) -> Path:
        return self.directory / CodeFile.META

    def exists(self) -> bool:
        return self.directory.exists()


@dataclass
class TreeRegistryPathConstructor:
    directory: Path

    @classmethod
    def from_dep_tree_hash(cls, dep_tree_hash: str, registry_root: Path | None = None) -> TreeRegistryPathConstructor:
        return cls(RegistryResolver(registry_root).snapshots_dir / dep_tree_hash)

    @property
    def tree_path(self) -> Path:
        return self.directory / SnapshotFile.TREE

    @property
    def graph_path(self) -> Path:
        return self.directory / SnapshotFile.GRAPH

    def exists(self) -> bool:
        return self.directory.exists()


def classify_file(path: Path) -> str:
    """Return a broad kind string for the given file path based on its extension."""
    suffix = path.suffix.lower()

    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'}:
        return 'image'
    if suffix == '.pdf':
        return 'pdf'
    if suffix in {'.tif', '.tiff', '.nc', '.vrt', '.npy', '.zarr'}:
        return 'cube'
    if suffix == '.html':
        return 'table'
    if suffix == '.json':
        return 'json'
    if suffix in {'.py', '.pyx', '.ipynb'}:
        return 'code'
    return 'file'
