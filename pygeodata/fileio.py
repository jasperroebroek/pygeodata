from __future__ import annotations

import contextlib
import json
import os
import uuid
from typing import TYPE_CHECKING, Any, TypeVar

import msgspec

from pygeodata.enums import JSONKeys

if TYPE_CHECKING:
    from pathlib import Path

    from filelock import BaseFileLock

T = TypeVar('T')


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


def write_typed_json_atomic(path: Path, obj: Any, indent: int | None = None) -> None:
    data = msgspec.json.encode(obj)
    write_bytes_atomic(path, data if indent is None else msgspec.json.format(data, indent=indent))


def read_typed_json(path: Path, type_: type[T]) -> T:
    """Decode ``path`` into ``type_``; raises ``OSError`` or ``msgspec.DecodeError``."""
    return msgspec.json.decode(path.read_bytes(), type=type_)


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


def read_format_version(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    version = data.get(JSONKeys.FORMAT_VERSION) if isinstance(data, dict) else None
    return version if isinstance(version, int) else None
