from collections.abc import Mapping, Sequence
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np

from pygeodata.formatting.shared import JSON_SAFE_TYPES, format_array
from pygeodata.types import AllowsFormatting


@singledispatch
def format_json(value: Any) -> float | int | str | bool | None | dict[str, Any] | list[Any]:
    if isinstance(value, JSON_SAFE_TYPES):
        return value
    if isinstance(value, AllowsFormatting):
        return value.format_as_json()
    return repr(value)


@format_json.register
def _(value: Enum) -> str:
    return f'{value.__class__.__name__}.{value.name}'


@format_json.register
def _(value: str) -> Any:
    if isinstance(value, Enum):
        return format_json.dispatch(Enum)(value)
    return repr(value)


@format_json.register
def _(value: int) -> Any:
    if isinstance(value, Enum):
        return format_json.dispatch(Enum)(value)
    return value


@format_json.register
def _(value: bytes) -> Any:
    return value.hex()


@format_json.register
def _(value: bytearray) -> Any:
    return value.hex()


@format_json.register
def _(value: Path) -> str:
    return str(value)


format_json.register(np.ndarray, lambda x: format_json(format_array(x)))


@format_json.register(Mapping)
def _(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(k): format_json(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}


@format_json.register(Sequence)
def _(value: Sequence[Any]) -> list[Any] | str:
    return [format_json(v) for v in value]


@format_json.register(set)
def _(value: set[Any]) -> list[Any]:
    return [format_json(v) for v in sorted(value, key=repr)]
