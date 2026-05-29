import re
from collections.abc import Mapping, Sequence
from enum import Enum
from functools import singledispatch
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS
from rasterio import CRS as rioCRS

from pygeodata.formatting.shared import (
    JSON_SAFE_TYPES,
    format_array,
    format_crs,
    format_float,
    format_path_object,
)
from pygeodata.hash import calculate_string_hash
from pygeodata.types import AllowsFormatting, Shape

STRUCTURAL_PATTERN = re.compile(r'[\[\]\{\}\(\):,]')


def sanitize(val: Any) -> str:
    return re.sub(r'[^\w\-.,\[\]\(\)\{\} ]', '_', str(val))


def encode_string(value: str) -> str:
    sanitized = sanitize(value)
    if sanitized != value or STRUCTURAL_PATTERN.search(value):
        return calculate_string_hash(value)
    return value


@singledispatch
def format_path(value: Any) -> str:
    if isinstance(value, AllowsFormatting):
        return encode_string(value.format_for_display())

    if isinstance(value, JSON_SAFE_TYPES):
        return repr(value)

    rep = repr(value)
    sanitized = sanitize(rep)
    if sanitized != rep or STRUCTURAL_PATTERN.search(rep):
        return f'{value.__class__.__name__}[{calculate_string_hash(rep)}]'
    return f'{value.__class__.__name__}[{sanitized}]'


format_path.register(float, format_float)
format_path.register(Path, lambda x: encode_string(format_path_object(x)))
format_path.register(np.ndarray, lambda x: encode_string(str(format_array(x))))


@format_path.register
def _(value: CRS | rioCRS) -> str:
    return encode_string(format_crs(value, sanitize))


@format_path.register
def _(value: Affine) -> str:
    parts = [format_float(x) for x in (value.a, value.b, value.c, value.d, value.e, value.f)]
    return f'A[{"_".join(parts)}]'


@format_path.register
def _(value: Shape) -> str:
    inner = 'x'.join(format_path(v) for v in value)
    return f'S[{inner}]'


@format_path.register
def _(value: Enum) -> str:
    return f'{value.__class__.__name__}[{format_path(value.name)}]'


@format_path.register
def _(value: str) -> str:
    if isinstance(value, Enum):
        return format_path.dispatch(Enum)(value)
    return encode_string(value)


@format_path.register
def _(value: int) -> str:
    if isinstance(value, Enum):
        return format_path.dispatch(Enum)(value)
    return str(value)


@format_path.register(Mapping)
def _(value: Mapping[Any, Any]) -> str:
    inner = ', '.join(
        f'{format_path(k)}:{format_path(v)}' for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
    )
    return f'{{{inner}}}'


@format_path.register(Sequence)
def _(value: Sequence[Any]) -> str:
    if isinstance(value, (str, bytes, bytearray)):
        return encode_string(repr(value))
    inner = ', '.join(format_path(v) for v in value)
    return f'[{inner}]'


@format_path.register
def _(value: bytes) -> str:
    return f'bytes[{value.hex()}]'


@format_path.register
def _(value: bytearray) -> str:
    return f'bytearray[{bytes(value).hex()}]'


@format_path.register(set)
def _(value: set[Any]) -> str:
    inner = ', '.join(format_path(v) for v in sorted(value, key=repr))
    return f'{{{inner}}}'
