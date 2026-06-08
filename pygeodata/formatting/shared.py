from typing import Any

import numpy as np

from pygeodata.hash import calculate_array_hash

JSON_SAFE_TYPES = (float, int, str, bool, type(None))


def format_array(value: np.ndarray) -> Any:
    if value.size == 1:
        return value.item()
    h = calculate_array_hash(value)
    return f'array(shape={value.shape}, dtype={value.dtype.name}, hash={h})'
