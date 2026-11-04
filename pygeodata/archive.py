"""Export cache entries, with the code snapshots that reproduce them, to a tar archive, and import them.

Archive layout::

    cache/<state_hash>/...          the entry's metadata and output files
    snapshots/<dep_hash>/tree.json  its dependency tree
    code/<source_hash>/...          the source of every class in that tree
"""

from __future__ import annotations

import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import msgspec

from pygeodata.config import get_config
from pygeodata.enums import CacheFile, ObjectType, RegistryDir, SnapshotFile
from pygeodata.paths import CachePathConstructor, CodeRegistryPathConstructor
from pygeodata.records import EntryRecord
from pygeodata.registries.registry import TreeRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from pygeodata.registries.registry import EntryRegistry

CACHE_DIR = 'cache'


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _entry_files(
    state_hash: str,
    entry_dir: Path,
    path_allowed: Callable[[str], bool],
    warnings: list[str],
) -> list[tuple[Path, str]]:
    """The entry's cache files, or [] with a warning appended if it's disallowed/unreadable.

    A caller whose ``path_allowed`` itself raises on violation (e.g. the
    browser's HTTP-boundary guard) still propagates that as a hard failure;
    a plain ``False`` return, or a missing/unreadable directory, is soft-
    failed via ``warnings`` instead so one bad entry doesn't sink the export.
    """
    if not path_allowed(str(entry_dir)):
        warnings.append(f'{state_hash}: path not allowed ({entry_dir})')
        return []
    try:
        paths = CachePathConstructor(entry_dir)
        meta_files = [
            p
            for p in (paths.state_hash_path, paths.params_path, paths.spec_path, paths.execution_graph_path)
            if p.is_file()
        ]
        return [(f, f'{CACHE_DIR}/{entry_dir.name}/{f.name}') for f in [*meta_files, *paths.output_paths()]]
    except OSError as exc:
        warnings.append(f'{state_hash}: could not read {entry_dir} ({exc})')
        return []


def _snapshot_and_code_files(
    dep_hash: str,
    trees: TreeRegistry,
    seen_src_hashes: set[str],
) -> list[tuple[Path, str]]:
    tree = trees.get_snapshot_from_hash(dep_hash)
    if tree is None:
        return []

    files = [(trees.get_tree_path(dep_hash), f'{RegistryDir.SNAPSHOTS}/{dep_hash}/{SnapshotFile.TREE}')]

    src_hashes = {node.hash for node in tree.nodes.values()}
    for src_hash in src_hashes - seen_src_hashes:
        seen_src_hashes.add(src_hash)
        code_dir = CodeRegistryPathConstructor.from_source_hash(src_hash).directory
        if code_dir.exists():
            files.extend((f, f'{RegistryDir.CODE}/{src_hash}/{f.name}') for f in code_dir.iterdir() if f.is_file())

    return files


def collect_export_files(
    state_hashes: list[str],
    registry: EntryRegistry,
    include_snapshots: bool,
    path_allowed: Callable[[str], bool] = lambda _: True,
    tree_registry: TreeRegistry | None = None,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return ``(files, warnings)``: ``(absolute_path, arcname)`` for every file to export, and one
    message per skipped entry. ``path_allowed`` is called on every entry dir before reading it.
    """
    trees = tree_registry if tree_registry is not None else TreeRegistry()
    seen_src_hashes: set[str] = set()
    seen_dep_hashes: set[str] = set()
    files: list[tuple[Path, str]] = []
    warnings: list[str] = []

    for state_hash in state_hashes:
        entry = registry.get_entry(state_hash)
        entry_dir = registry.entry_dir(state_hash)
        if entry is None or entry_dir is None:
            continue

        files.extend(_entry_files(state_hash, entry_dir, path_allowed, warnings))

        dep_hash = entry.record.dependency_tree_hash
        if include_snapshots and dep_hash not in seen_dep_hashes:
            seen_dep_hashes.add(dep_hash)
            files.extend(_snapshot_and_code_files(dep_hash, trees, seen_src_hashes))

    return files, warnings


def write_export_tar(files: list[tuple[Path, str]], dest: Path) -> None:
    """Write files (as returned by collect_export_files) into an uncompressed tar at dest."""
    with tarfile.open(dest, mode='w') as tar:
        for path, arcname in files:
            tar.add(path, arcname=arcname)


@dataclass
class EntryDownload:
    path: Path
    name: str
    needs_tar: bool  # a directory output, e.g. a zarr store


def entry_download(
    state_hash: str,
    registry: EntryRegistry,
    path_allowed: Callable[[str], bool] = lambda _: True,
) -> EntryDownload | None:
    """The entry's primary output, or None when the entry, its output or access is missing.

    ``path_allowed`` is called on the entry dir; callers without an HTTP boundary (the CLI) can
    omit it.
    """
    entry = registry.get_entry(state_hash)
    entry_dir = registry.entry_dir(state_hash)
    if entry is None or entry_dir is None or entry.primary_file is None or not path_allowed(str(entry_dir)):
        return None
    path = entry_dir / entry.primary_file.relpath
    if not path.exists():
        return None
    if path.is_dir():
        return EntryDownload(path=path, name=f'{path.name}.tar', needs_tar=True)
    return EntryDownload(path=path, name=path.name, needs_tar=False)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _entry_cache_root(tar: tarfile.TarFile, member: tarfile.TarInfo) -> Path | None:
    """The local cache root for the entry whose ``meta.json`` is ``member``."""
    f = tar.extractfile(member)
    if f is None:
        return None
    try:
        record = msgspec.json.decode(f.read(), type=EntryRecord)
        return get_config().get_cache_root(ObjectType(record.object_type))
    except (msgspec.DecodeError, OSError, ValueError):
        return None


def _destination(
    parts: tuple[str, ...],
    cache_roots: dict[str, Path | None],
    registry_root: Path,
) -> tuple[Path, Path] | None:
    """``(destination, the root it must stay inside)`` for an archive member, or None to skip it."""
    if len(parts) < 3 or any(p in ('..', '') or Path(p).is_absolute() for p in parts):
        return None
    if parts[0] == CACHE_DIR:
        cache_root = cache_roots.get(parts[1])
        return (cache_root / Path(*parts[1:]), cache_root) if cache_root is not None else None
    if parts[0] in (RegistryDir.CODE, RegistryDir.SNAPSHOTS):
        return registry_root / Path(*parts), registry_root
    return None


def _extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path, root: Path) -> bool:
    resolved = dest.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return False
    src = tar.extractfile(member)
    if src is None:
        return False
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with src, resolved.open('wb') as out:
        shutil.copyfileobj(src, out)
    return True


def import_archive(archive: Path) -> dict[str, int]:
    """Extract an export archive into this project; files that already exist are kept.

    Returns the number of files extracted per top-level archive dir.
    """
    registry_root = Path(get_config().path_registry)
    counts = dict.fromkeys((CACHE_DIR, RegistryDir.CODE, RegistryDir.SNAPSHOTS), 0)

    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        cache_roots: dict[str, Path | None] = {}
        for member in members:
            parts = PurePosixPath(member.name).parts
            if len(parts) == 3 and parts[0] == CACHE_DIR and parts[2] == CacheFile.META:
                cache_roots[parts[1]] = _entry_cache_root(tar, member)

        for member in members:
            if member.isdir():
                continue
            parts = PurePosixPath(member.name).parts
            target = _destination(parts, cache_roots, registry_root)
            if target is None or target[0].exists():
                continue
            if _extract_member(tar, member, *target):
                counts[parts[0]] += 1

    return counts
