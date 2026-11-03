from __future__ import annotations

import contextlib
import json
import os
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from filelock import BaseFileLock


def write_bytes_atomic(path: Path, data: bytes) -> None:
    # a symlinked file stays a symlink: write its target
    path = path.resolve()
    # unique: concurrent writers don't collide
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('xb') as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, data: Any, indent: int | None = None) -> None:
    write_bytes_atomic(path, json.dumps(data, indent=indent).encode('utf-8'))


def file_lock(path: Path, timeout: float = -1) -> BaseFileLock:
    from filelock import FileLock  # noqa: PLC0415 — filelock imports asyncio (~70 ms): keep it off the CLI import path

    return FileLock(path, timeout=timeout)


def try_acquire(lock: BaseFileLock) -> bool:
    from filelock import Timeout  # noqa: PLC0415 — see file_lock

    try:
        lock.acquire(timeout=0)
    except Timeout:
        return False
    return True
