"""Shared export logic used by both the CLI and the registry browser."""

from __future__ import annotations

import tarfile
from collections.abc import Callable
from pathlib import Path

from pygeodata.catalog.types import EntryInfo
from pygeodata.paths import CodeRegistryPathConstructor
from pygeodata.registries.registry import TreeRegistry


def _cache_files_for_entry(
    record_id: str,
    entry: EntryInfo,
    path_allowed: Callable[[str], bool],
    warnings: list[str],
) -> list[tuple[Path, str]]:
    """Return this entry's cache files, or [] with a warning appended if it's disallowed/unreadable.

    A caller whose ``path_allowed`` itself raises on violation (e.g. the
    browser's HTTP-boundary guard) still propagates that as a hard failure;
    a plain ``False`` return, or a missing/unreadable directory, is soft-
    failed via ``warnings`` instead so one bad entry doesn't sink the export.
    """
    cache_dir = Path(entry.params_path).parent
    if not path_allowed(str(cache_dir)):
        warnings.append(f'{record_id}: path not allowed ({cache_dir})')
        return []
    try:
        return [(f, f'cache/{cache_dir.name}/{f.name}') for f in cache_dir.iterdir() if f.is_file()]
    except OSError as exc:
        warnings.append(f'{record_id}: could not read {cache_dir} ({exc})')
        return []


def _snapshot_and_code_files(
    dep_hash: str,
    trees: TreeRegistry,
    seen_src_hashes: set[str],
) -> list[tuple[Path, str]]:
    tree = trees.get_snapshot_from_hash(dep_hash)
    if tree is None:
        return []

    files = [(trees.get_tree_path(dep_hash), f'snapshots/{dep_hash}/tree.json')]

    src_hashes = {node.get('hash') for node in tree.nodes.values() if isinstance(node, dict) and node.get('hash')}
    for src_hash in src_hashes - seen_src_hashes:
        seen_src_hashes.add(src_hash)
        code_dir = CodeRegistryPathConstructor.from_source_hash(src_hash).directory
        if code_dir.exists():
            files.extend((f, f'code/{src_hash}/{f.name}') for f in code_dir.iterdir() if f.is_file())

    return files


def collect_export_files(
    record_ids: list[str],
    entries: dict[str, EntryInfo],
    include_snapshots: bool,
    path_allowed: Callable[[str], bool] = lambda _: True,
    tree_registry: TreeRegistry | None = None,
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Return (files, warnings) for all files to be exported.

    ``files`` is a list of (absolute_path, arcname). An entry whose cache
    directory is disallowed. ``warnings`` carries one message per
    skipped entry.

    ``path_allowed`` is called on every cache directory path before
    iterating its contents; a ``False`` return skips that entry.
    """
    trees = tree_registry if tree_registry is not None else TreeRegistry()
    seen_src_hashes: set[str] = set()
    seen_dep_hashes: set[str] = set()
    files: list[tuple[Path, str]] = []
    warnings: list[str] = []

    for record_id in record_ids:
        entry = entries.get(record_id)
        if entry is None:
            continue

        files.extend(_cache_files_for_entry(record_id, entry, path_allowed, warnings))

        if include_snapshots and entry.dep_hash and entry.dep_hash not in seen_dep_hashes:
            seen_dep_hashes.add(entry.dep_hash)
            files.extend(_snapshot_and_code_files(entry.dep_hash, trees, seen_src_hashes))

    return files, warnings


def write_export_tar(files: list[tuple[Path, str]], dest: Path) -> None:
    """Write files (as returned by collect_export_files) into an uncompressed tar at dest."""
    with tarfile.open(dest, mode='w') as tar:
        for path, arcname in files:
            tar.add(path, arcname=arcname)


def single_entry_tar_path(
    record_id: str,
    entries: dict[str, EntryInfo],
    path_allowed: Callable[[str], bool] = lambda _: True,
) -> tuple[Path | None, str | None, bool]:
    """Locate the primary data file/dir for a single export.

    Returns (data_path, download_name, needs_tar) where:
    - data_path is the file/dir to send (None if entry missing or disallowed)
    - download_name is the suggested filename
    - needs_tar is True when data_path is a directory

    ``path_allowed`` is called on the cache directory — the guard is
    provably still in effect for single-entry downloads. Callers without an
    HTTP boundary (the CLI) can omit it and get the permissive default.
    """
    entry = entries.get(record_id)
    if entry is None:
        return None, None, False

    cache_dir = Path(entry.params_path).parent
    if not path_allowed(str(cache_dir)):
        return None, None, False

    data_path = next(
        (f for f in cache_dir.iterdir() if not f.name.startswith('.')),
        None,
    )
    if data_path is None:
        return None, None, False

    if data_path.is_dir():
        return data_path, f'{data_path.name}.tar', True
    return data_path, data_path.name, False
