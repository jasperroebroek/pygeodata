from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def existing_path_str(path: Path | str | None) -> str | None:
    if path is None:
        return None

    p = Path(path).expanduser().resolve()
    if not p.exists():
        logger.debug('Path does not exist: %s', p)
        return None

    return str(p)


def read_json_dict(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}

    p = Path(path)

    if not p.exists():
        logger.debug('JSON file does not exist: %s', p)
        return {}

    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        logger.warning('Invalid JSON in %s: %s', p, exc)
        return {}
    except OSError as exc:
        logger.warning('Failed to read JSON file %s: %s', p, exc)
        return {}

    if not isinstance(data, dict):
        logger.warning('JSON file does not contain an object: %s', p)
        return {}

    return data


def read_text(path: Path | str | None) -> str | None:
    if path is None:
        return None

    p = Path(path)

    if not p.exists():
        logger.debug('Text file does not exist: %s', p)
        return None

    try:
        return p.read_text(encoding='utf-8')
    except OSError as exc:
        logger.warning('Failed to read text file %s: %s', p, exc)


from pygeodata.file_utils import classify_file

__all__ = ['classify_file', 'existing_path_str', 'read_json_dict', 'read_text']
