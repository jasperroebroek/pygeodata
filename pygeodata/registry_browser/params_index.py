from typing import Any

from pygeodata.config import JSONKeys
from pygeodata.formatting.json import format_json
from pygeodata.catalog.types import LinkedEntry, ParamRow

HIDDEN_KEYS = frozenset(key.value for key in JSONKeys)


def _is_data_ref(value: Any) -> bool:
    return isinstance(value, dict) and JSONKeys.CLASS_NAME in value and JSONKeys.PARAMS in value


def _format_scalar(value: Any) -> str:
    """Format a leaf value to a display string."""
    formatted = format_json(value)
    if isinstance(formatted, str):
        return formatted
    if isinstance(formatted, list):
        return '[' + ', '.join(str(v) for v in formatted) + ']'
    return str(formatted)


def _flatten(
    value: Any,
    key: str,
    scope: str,
    rows: list[ParamRow],
    linked_entries: list[LinkedEntry] | None,
    depth: int,
) -> None:
    """Recursively walk a value and emit ParamRow objects."""
    if _is_data_ref(value):
        class_name = value[JSONKeys.CLASS_NAME]
        inner = value.get(JSONKeys.PARAMS, {})
        # Emit the ref row itself
        search_blob = f'{scope} {key} {class_name} data_ref'.lower()
        rows.append(
            ParamRow(
                path=f'{scope}.{key}' if scope else key,
                key_group=scope,
                final_key=key,
                value_text=class_name,
                value_type='data_ref',
                search_blob=search_blob,
                depth=depth,
            ),
        )
        if linked_entries is not None:
            linked_entries.append(
                LinkedEntry(
                    param_name=key,
                    class_name=class_name,
                    state_hash=value.get(JSONKeys.STATE_HASH),
                    params_summary={k: _format_scalar(v) for k, v in inner.items()},
                ),
            )
        child_scope = f'{scope} › {key}' if scope else key
        for k, v in inner.items():
            if k in HIDDEN_KEYS:
                continue
            _flatten(v, k, child_scope, rows, linked_entries, depth + 1)
        return

    if isinstance(value, dict):
        child_scope = f'{scope} › {key}' if scope else key
        for k, v in value.items():
            if k in HIDDEN_KEYS:
                continue
            _flatten(v, k, child_scope, rows, linked_entries, depth + 1)
        return

    if isinstance(value, (list, tuple)):
        formatted = format_json(value)
        if not isinstance(formatted, list):
            formatted = list(formatted) if hasattr(formatted, '__iter__') else [formatted]

        # Check if any element expands (is a dict/data_ref)
        any_complex = any(isinstance(item, (dict, list, tuple)) for item in value)

        if any_complex:
            for idx, item in enumerate(value):
                indexed_key = f'{key}[{idx}]'
                if isinstance(item, (dict, list, tuple)):
                    _flatten(item, indexed_key, scope, rows, linked_entries, depth)
                else:
                    val_str = _format_scalar(item)
                    path = f'{scope}.{indexed_key}' if scope else indexed_key
                    search_blob = f'{scope} {indexed_key} {val_str}'.lower()
                    rows.append(
                        ParamRow(
                            path=path,
                            key_group=scope,
                            final_key=indexed_key,
                            value_text=val_str,
                            value_type='scalar',
                            search_blob=search_blob,
                            depth=depth,
                        ),
                    )
        else:
            # All scalars — emit one row per element with key[]
            list_key = f'{key}[]'
            for item in value:
                val_str = _format_scalar(item)
                path = f'{scope}.{list_key}' if scope else list_key
                search_blob = f'{scope} {list_key} {val_str}'.lower()
                rows.append(
                    ParamRow(
                        path=path,
                        key_group=scope,
                        final_key=list_key,
                        value_text=val_str,
                        value_type='list_member',
                        search_blob=search_blob,
                        depth=depth,
                    ),
                )
        return

    # Scalar leaf
    val_str = _format_scalar(value)
    path = f'{scope}.{key}' if scope else key
    search_blob = f'{scope} {key} {val_str}'.lower()
    rows.append(
        ParamRow(
            path=path,
            key_group=scope,
            final_key=key,
            value_text=val_str,
            value_type=type(value).__name__,
            search_blob=search_blob,
            depth=depth,
        ),
    )


def flatten_params(
    params: dict,
    linked_entries: list[LinkedEntry] | None = None,
) -> list[ParamRow]:
    rows: list[ParamRow] = []
    for key, value in params.items():
        if key in HIDDEN_KEYS:
            continue
        _flatten(value, key, '', rows, linked_entries, 0)
    return rows
