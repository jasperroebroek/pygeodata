import hashlib
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np


def format_enum(value: Enum) -> str:
    return f'{value.__class__.__name__}.{value.name}'


def format_path(value: Path) -> str:
    return '/'.join(value.parts)


def format_array(value: np.ndarray) -> Any:
    if value.size == 1:
        return value.item()
    digest = hashlib.sha256(value.tobytes(order='C')).hexdigest()
    return f'array(shape={value.shape}, dtype={value.dtype}, hash={digest})'


@singledispatch
def format_value_as_json(value: Any) -> float | int | str | bool | None:
    JSON_SAFE_TYPES = (float, int, str, bool, type(None))
    if isinstance(value, JSON_SAFE_TYPES):
        return value
    return repr(value)


@singledispatch
def format_value_as_string(value: Any) -> str:
    return str(value)


format_value_as_string.register(Enum, format_enum)
format_value_as_string.register(Path, format_path)
format_value_as_string.register(np.ndarray, lambda x: str(format_array(x)))

format_value_as_json.register(Enum, format_enum)
format_value_as_json.register(Path, format_path)
format_value_as_json.register(np.ndarray, lambda x: format_value_as_json(format_array(x)))
