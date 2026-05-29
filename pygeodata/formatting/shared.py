import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import CRS

from pygeodata.hash import calculate_array_hash

JSON_SAFE_TYPES = (float, int, str, bool, type(None))


def format_array(value: np.ndarray) -> Any:
    if value.size == 1:
        return value.item()
    h = calculate_array_hash(value)
    return f'array(shape={value.shape}, dtype={value.dtype.name}, hash={h})'


def format_path_object(value: Path) -> str:
    return '-'.join(value.parts)


def format_float(value: float, precision: int = 15) -> str:
    if not np.isfinite(value):
        return str(value).lower()

    s = format(value, f'.{precision}g')
    s = s.replace('.', 'p')
    s = s.replace('+', '')
    return s


def format_crs(value: CRS, sanitize_fn: Callable[[str], str]) -> str:
    s = value.to_string()
    if len(s) > 25:
        return hashlib.sha256(s.encode('utf-8')).hexdigest()
    return sanitize_fn(s)
