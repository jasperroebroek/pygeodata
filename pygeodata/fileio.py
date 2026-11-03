from __future__ import annotations

import contextlib
import json
import os
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


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
