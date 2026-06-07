from collections.abc import Generator, Iterable, Mapping, Sequence
from typing import Any

from pygeodata.config import get_config
from pygeodata.types import HasParameters, T


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


def flatten_parameter_value_for_path(prefix: str, value: Any) -> dict[str, Any]:
    """
    Flatten a potentially nested parameter value into a dict of path-safe key-value pairs.

    Nested :class:`Artifact` instances are expanded with ``__``-separated keys.
    Lists and dicts containing artifacts get additional entries for each.

    Parameters
    ----------
    prefix : str
        The key prefix for the flattened entry.
    value : Any
        The value to flatten.

    Returns
    -------
    dict[str, Any]
        A flat dict of path-safe string values, keyed by their dotted parameter names.
    """
    flat: dict[str, Any] = {}
    format_fn = get_config().format_path_fn

    if isinstance(value, set):
        value = tuple(sorted(value, key=repr))

    if isinstance(value, HasParameters):
        flat[prefix] = value.__class__.__name__
        for nk, nv in value.get_params().items():
            flat.update(flatten_parameter_value_for_path(f'{prefix}__{nk}', nv))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        flat[prefix] = format_fn(value)
        for idx, item in enumerate(value):
            if any(extract_instances(item, HasParameters)):
                flat.update(flatten_parameter_value_for_path(f'{prefix}_{idx}', item))
    elif isinstance(value, dict):
        flat[prefix] = format_fn(value)
        for k in sorted(value):
            if any(extract_instances(value[k], HasParameters)):
                flat.update(flatten_parameter_value_for_path(f'{prefix}_{k}', value[k]))
    else:
        flat[prefix] = format_fn(value)

    return flat


def flatten_parameter_dict_for_path(params: Mapping[str, Any]) -> dict[str, Any]:
    """
    Flatten a potentially nested parameter dict into a dict of path-safe key-value pairs.

    Nested :class:`Artifact` instances are expanded with ``__``-separated keys.
    Lists and dicts containing artifacts get additional entries for each.

    Parameters
    ----------
    params: Mapping[str, Any]
        A dict of parameter name-value pairs.

    Returns
    -------
    dict[str, Any]
        A flat dict of path-safe string values, keyed by their extended parameter names.
    """
    flat_params = {}
    for k, v in sorted(params.items(), key=lambda kv: str(kv[0])):
        flat_params.update(flatten_parameter_value_for_path(k, v))
    return flat_params
