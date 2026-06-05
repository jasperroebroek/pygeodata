from collections.abc import Mapping, Sequence
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np

from pygeodata.formatting.shared import JSON_SAFE_TYPES, format_array
from pygeodata.types import AllowsFormatting, SpatialSpec


@singledispatch
def format_json(value: Any, spec: SpatialSpec | None = None) -> float | int | str | bool | None | dict[str, Any] | list[Any]:
    if isinstance(value, JSON_SAFE_TYPES):
        return value
    if isinstance(value, AllowsFormatting):
        return value.format_as_json(spec=spec)
    return repr(value)


@format_json.register
def _(value: Enum, spec: SpatialSpec | None = None) -> str:
    return f'{value.__class__.__name__}.{value.name}'


@format_json.register
def _(value: str, spec: SpatialSpec | None = None) -> Any:
    if isinstance(value, Enum):
        return format_json.dispatch(Enum)(value)
    return value


@format_json.register
def _(value: int, spec: SpatialSpec | None = None) -> Any:
    if isinstance(value, Enum):
        return format_json.dispatch(Enum)(value)
    return value


@format_json.register
def _(value: bytes, spec: SpatialSpec | None = None) -> Any:
    return value.hex()


@format_json.register
def _(value: bytearray, spec: SpatialSpec | None = None) -> Any:
    return value.hex()


@format_json.register
def _(value: Path, spec: SpatialSpec | None = None) -> str:
    return str(value)


format_json.register(np.ndarray, lambda x, spec=None: format_json(format_array(x), spec=spec))


@format_json.register(Mapping)
def _(value: Mapping[Any, Any], spec: SpatialSpec | None = None) -> dict[str, Any]:
    return {str(k): format_json(v, spec=spec) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}


@format_json.register(Sequence)
def _(value: Sequence[Any], spec: SpatialSpec | None = None) -> list[Any] | str:
    return [format_json(v, spec=spec) for v in value]


@format_json.register(set)
def _(value: set[Any], spec: SpatialSpec | None = None) -> list[Any]:
    return [format_json(v, spec=spec) for v in sorted(value, key=repr)]
