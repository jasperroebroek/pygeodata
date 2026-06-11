from collections.abc import Mapping, Sequence
from enum import Enum
from functools import singledispatch
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from pygeodata.formatting.shared import format_array
from pygeodata.protocols import AllowsFormatting


def _is_container(value: Any) -> bool:
    return isinstance(value, (Mapping, Sequence, set)) and not isinstance(value, (str, bytes, bytearray))


def _indent(level: int) -> str:
    return '    ' * level


def _html(s: str) -> str:
    return escape(s, quote=True)


@singledispatch
def format_html_inline(value: Any) -> str:
    if isinstance(value, AllowsFormatting):
        return _html(value.format_for_display())
    return _html(repr(value))


@format_html_inline.register
def _(value: Enum) -> str:
    return _html(f'{value.__class__.__name__}.{value.name}')


@format_html_inline.register
def _(value: int) -> str:
    if isinstance(value, Enum):
        return format_html_inline.dispatch(Enum)(value)
    return _html(str(value))


@format_html_inline.register
def _(value: str) -> str:
    if isinstance(value, Enum):
        return format_html_inline.dispatch(Enum)(value)
    return _html(value)


@format_html_inline.register
def _(value: bytes) -> str:
    return _html(value.hex())


@format_html_inline.register
def _(value: bytearray) -> str:
    return _html(value.hex())


@format_html_inline.register
def _(value: Path) -> str:
    return _html(str(value))


@format_html_inline.register
def _(value: np.ndarray) -> str:
    return _html(str(format_array(value)))


@format_html_inline.register(Mapping)
def _(value: Mapping[Any, Any]) -> str:
    inner = ', '.join(
        f'{format_html_inline(k)}: {format_html_inline(v)}'
        for k, v in sorted(value.items(), key=lambda kv: repr(kv[0]))
    )
    return _html('{') + inner + _html('}')


@format_html_inline.register(Sequence)
def _(value: Sequence[Any]) -> str:
    inner = ', '.join(format_html_inline(v) for v in value)
    return _html('[') + inner + _html(']')


@format_html_inline.register(set)
def _(value: set[Any]) -> str:
    inner = ', '.join(format_html_inline(v) for v in sorted(value, key=repr))
    return _html('{') + inner + _html('}')


def format_html_block(value: Any, indent: int = 0, nested: bool = True) -> str:
    if isinstance(value, AllowsFormatting):
        prefix = _indent(indent) if nested else ''
        return f'{prefix}{_html(value.format_for_display())}'

    if isinstance(value, Mapping):
        lines: list[str] = []
        for k, v in sorted(value.items(), key=lambda kv: repr(kv[0])):
            if _is_container(v):
                lines.append(f'{_indent(indent)}{format_html_inline(k)} =')
                lines.append(format_html_block(v, indent + 1, nested=True))
            else:
                lines.append(f'{_indent(indent)}{format_html_inline(k)} = {format_html_inline(v)}')
        return '\n'.join(lines)

    if isinstance(value, set):
        items = sorted(value, key=repr)
        lines: list[str] = []
        for item in items:
            if _is_container(item):
                lines.append(f'{_indent(indent)}-')
                lines.append(format_html_block(item, indent + 1, nested=True))
            else:
                lines.append(f'{_indent(indent)}- {format_html_inline(item)}')
        return '\n'.join(lines)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        lines: list[str] = []
        for item in value:
            if _is_container(item):
                lines.append(f'{_indent(indent)}-')
                lines.append(format_html_block(item, indent + 1, nested=True))
            else:
                lines.append(f'{_indent(indent)}- {format_html_inline(item)}')
        return '\n'.join(lines)

    prefix = _indent(indent) if nested else ''
    return f'{prefix}{format_html_inline(value)}'
