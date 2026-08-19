from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from pygeodata.catalog.export import collect_export_files, single_entry_tar_path, write_export_tar

__all__ = [
    'collect_export_files',
    'create_job',
    'get_job',
    'pop_job',
    'run_export_job',
    'single_entry_tar_path',
]

# ---------------------------------------------------------------------------
# Job registry  {job_id: {status, done, total, tmp_path, error}}
# ---------------------------------------------------------------------------

_export_jobs: dict[str, dict] = {}
_export_jobs_lock = threading.Lock()


def run_export_job(job_id: str, files: list[tuple[Path, str]]) -> None:
    """Write all files into a temp tar and update the job registry."""
    job = _export_jobs[job_id]
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.tar')
        os.close(fd)
        write_export_tar(files, Path(tmp_path))
        with _export_jobs_lock:
            job['done'] = len(files)
            job['tmp_path'] = tmp_path
            job['status'] = 'complete'
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        with _export_jobs_lock:
            job['status'] = 'error'
            job['error'] = str(exc)


def create_job(job_id: str, total: int, warnings: list[str] | None = None) -> None:
    """Register a new job in the jobs dict (call before spawning the thread).

    ``warnings`` carries any entries collect_export_files skipped (disallowed,
    missing, or unreadable) so the browser can surface them alongside the
    completed job instead of the export silently dropping them.
    """
    with _export_jobs_lock:
        _export_jobs[job_id] = {
            'status': 'running',
            'done': 0,
            'total': total,
            'tmp_path': None,
            'error': None,
            'warnings': warnings or [],
        }


def get_job(job_id: str) -> dict | None:
    """Return a snapshot of the job dict, or None if not found."""
    with _export_jobs_lock:
        return _export_jobs.get(job_id)


def pop_job(job_id: str) -> dict | None:
    """Remove and return the job, or None if already gone."""
    with _export_jobs_lock:
        return _export_jobs.pop(job_id, None)
