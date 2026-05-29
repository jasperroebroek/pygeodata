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

from pygeodata.formatting.shared import JSON_SAFE_TYPES, format_crs, format_float, format_path_object
from pygeodata.hash import calculate_array_hash, calculate_string_hash
from pygeodata.types import AllowsFormatting, Shape

UNSAFE_STRING_PATTERN = re.compile(r'--|__|\s_\s')


def sanitize(value: str) -> str:
    value = re.compile(r'[^A-Za-z0-9 _-]+').sub(' ', value)
    return re.compile(r'\s+').sub(' ', value).strip()


def encode_string(value: str) -> str:
    sanitized = sanitize(value)
    if value != sanitized or UNSAFE_STRING_PATTERN.search(value):
        return calculate_string_hash(value)
    return value


@singledispatch
def format_path_simplified(value: Any) -> str:
    if isinstance(value, AllowsFormatting):
        return encode_string(value.format_for_display())
    if isinstance(value, JSON_SAFE_TYPES):
        return repr(value)
    rep = repr(value)
    san = sanitize(rep)
    if rep != san or UNSAFE_STRING_PATTERN.search(san):
        return f'{value.__class__.__name__.lower()}--{calculate_string_hash(rep)}--'
    return f'{value.__class__.__name__.lower()}--{san}--'


format_path_simplified.register(float, format_float)
format_path_simplified.register(Path, lambda x: encode_string(format_path_object(x)))


@format_path_simplified.register
def _(value: CRS | rioCRS) -> str:
    def sanitize(x: str) -> str:
        return re.sub(r'[^\w\-.,\[\]\(\)\{\} ]', '_', str(x))

    return encode_string(format_crs(value, sanitize))


@format_path_simplified.register
def _(value: Affine) -> str:
    parts = [format_float(x) for x in (value.a, value.b, value.c, value.d, value.e, value.f)]
    return f'A--{"_".join(parts)}--'


@format_path_simplified.register
def _(value: Shape) -> str:
    inner = 'x'.join(format_path_simplified(v) for v in value)
    return f'S--{inner}--'


@format_path_simplified.register
def _(value: Enum) -> str:
    return f'{value.__class__.__name__}--{format_path_simplified(value.name)}--'


@format_path_simplified.register
def _(value: str) -> str:
    if isinstance(value, Enum):
        return format_path_simplified.dispatch(Enum)(value)
    return encode_string(value)


@format_path_simplified.register
def _(value: int) -> str:
    if isinstance(value, Enum):
        return format_path_simplified.dispatch(Enum)(value)
    return str(value)


@format_path_simplified.register
def _(value: np.ndarray) -> str:
    if value.size == 1:
        return format_path_simplified(value.item())
    h = calculate_array_hash(value)
    return f'array--{"x".join(map(str, value.shape))}_{value.dtype.name}_{h}--'


@format_path_simplified.register(Mapping)
def _(value: Mapping[Any, Any]) -> str:
    parts = []
    for k, v in sorted(value.items(), key=lambda kv: repr(kv[0])):
        key = format_path_simplified(k)
        val = format_path_simplified(v)
        parts.append(f'{key}--{val}')
    return f'map--{" _ ".join(parts)}--'


@format_path_simplified.register(Sequence)
def _(value: Sequence[Any]) -> str:
    parts = [format_path_simplified(v) for v in value]
    return f'seq--{" _ ".join(parts)}--'


@format_path_simplified.register
def _(value: bytes) -> str:
    return value.hex()


@format_path_simplified.register
def _(value: bytearray) -> str:
    return value.hex()


@format_path_simplified.register(set)
def _(value: set[Any]) -> str:
    parts = [format_path_simplified(v) for v in sorted(value, key=repr)]
    return f'set--{" _ ".join(parts)}--'
