from collections.abc import Generator, Iterable, Mapping
from typing import Any

from pygeodata.protocols import T


def extract_instances(value: Any, target_type: type[T]) -> Generator[T]:
    """
    Recursively yield all instances of `target_type` found in a nested structure.

    Traverses arbitrary iterables and mappings recursively, while treating strings
    and bytes as atomic (non-iterable) values.

    Parameters
    ----------
    value : Any
        A value that may contain `target_type` instances at any level of nesting.
    target_type : Type[T]
        The type to search for.

    Yields
    ------
    T
        Each instance of `target_type` found in the structure.
    """
    if isinstance(value, target_type):
        yield value
        return

    if isinstance(value, (str, bytes)):
        return

    if isinstance(value, Mapping):
        for item in value.values():
            yield from extract_instances(item, target_type)
        return

    if isinstance(value, Iterable):
        for item in value:
            yield from extract_instances(item, target_type)
        return
