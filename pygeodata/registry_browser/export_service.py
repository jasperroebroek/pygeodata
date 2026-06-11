"""Export job management, moved out of web.py.

The HTTP-boundary path guard (_assert_allowed_path) is passed in as a callable
so the security check provably stays at the Flask layer and is never skipped.
"""

from __future__ import annotations

import os
import tarfile
import tempfile
import threading
from pathlib import Path
from typing import Callable

from pygeodata.config import get_config
from pygeodata.paths import CodeRegistryResolver
from pygeodata.registry import TreeRegistry

# ---------------------------------------------------------------------------
# Job registry  {job_id: {status, done, total, tmp_path, error}}
# ---------------------------------------------------------------------------

_export_jobs: dict[str, dict] = {}
_export_jobs_lock = threading.Lock()


def collect_export_files(
    record_ids: list[str],
    entries: dict,
    include_snapshots: bool,
    assert_allowed_path: Callable[[str], object],
) -> list[tuple[Path, str]]:
    """Return list of (absolute_path, arcname) for all files to be exported.

    ``assert_allowed_path`` is called on every cache directory path before
    iterating its contents, preserving the HTTP-boundary path guard.
    """
    files: list[tuple[Path, str]] = []
    seen_src_hashes: set[str] = set()
    seen_dep_hashes: set[str] = set()

    for record_id in record_ids:
        entry = entries.get(record_id)
        if entry is None:
            continue

        cache_dir = Path(entry.params_path).parent
        assert_allowed_path(str(cache_dir))
        for f in cache_dir.iterdir():
            if f.is_file():
                files.append((f, f'cache/{cache_dir.name}/{f.name}'))

        if include_snapshots and entry.dep_hash and entry.dep_hash not in seen_dep_hashes:
            seen_dep_hashes.add(entry.dep_hash)
            trees = TreeRegistry(get_config().path_registry)
            tree = trees.get_snapshot(entry.dep_hash)
            if tree is not None:
                files.append((trees.get_tree_path(entry.dep_hash), f'snapshots/{entry.dep_hash}/tree.json'))
                for node in tree.nodes.values():
                    src_hash = node.get('hash') if isinstance(node, dict) else None
                    if src_hash and src_hash not in seen_src_hashes:
                        seen_src_hashes.add(src_hash)
                        code_dir = CodeRegistryResolver.from_source_hash(src_hash).directory
                        if code_dir.exists():
                            for f in code_dir.iterdir():
                                if f.is_file():
                                    files.append((f, f'code/{src_hash}/{f.name}'))

    return files


def run_export_job(job_id: str, files: list[tuple[Path, str]]) -> None:
    """Write all files into a temp tar and update the job registry."""
    job = _export_jobs[job_id]
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.tar')
        os.close(fd)
        with tarfile.open(tmp_path, mode='w') as tar:
            for i, (path, arcname) in enumerate(files):
                tar.add(path, arcname=arcname)
                with _export_jobs_lock:
                    job['done'] = i + 1
        with _export_jobs_lock:
            job['tmp_path'] = tmp_path
            job['status'] = 'complete'
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        with _export_jobs_lock:
            job['status'] = 'error'
            job['error'] = str(exc)


def create_job(job_id: str, total: int) -> None:
    """Register a new job in the jobs dict (call before spawning the thread)."""
    with _export_jobs_lock:
        _export_jobs[job_id] = {
            'status': 'running',
            'done': 0,
            'total': total,
            'tmp_path': None,
            'error': None,
        }


def get_job(job_id: str) -> dict | None:
    """Return a snapshot of the job dict, or None if not found."""
    with _export_jobs_lock:
        return _export_jobs.get(job_id)


def pop_job(job_id: str) -> dict | None:
    """Remove and return the job, or None if already gone."""
    with _export_jobs_lock:
        return _export_jobs.pop(job_id, None)


def single_entry_tar_path(
    record_id: str,
    entries: dict,
    assert_allowed_path: Callable[[str], object],
) -> tuple[Path | None, str | None, bool]:
    """Locate the primary data file/dir for a single export.

    Returns (data_path, download_name, needs_tar) where:
    - data_path is the file/dir to send (None if entry missing)
    - download_name is the suggested filename
    - needs_tar is True when data_path is a directory

    ``assert_allowed_path`` is called on the cache directory — the guard is
    provably still in effect for single-entry downloads.
    """
    entry = entries.get(record_id)
    if entry is None:
        return None, None, False

    cache_dir = Path(entry.params_path).parent
    assert_allowed_path(str(cache_dir))

    data_path = next(
        (f for f in cache_dir.iterdir() if not f.name.startswith('.')),
        None,
    )
    if data_path is None:
        return None, None, False

    if data_path.is_dir():
        return data_path, f'{data_path.name}.tar', True
    return data_path, data_path.name, False
