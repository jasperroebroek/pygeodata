from __future__ import annotations

import ast
import functools
import hashlib
import json
from typing import TYPE_CHECKING

from pygeodata.ast import get_source_ast_tree

if TYPE_CHECKING:
    import numpy as np


# Required fields that can still hold None; every other None field is optional and omitted.
_NULLABLE_REQUIRED_FIELDS = frozenset({('Constant', 'value'), ('MatchSingleton', 'value'), ('Interpolation', 'str')})

# Fields added to existing nodes after Python 3.14, with the value the parser gives when the new
# feature is not used; omitted at that value so code that does not use it keeps its hash.
_ADDED_FIELDS = {('Import', 'is_lazy'): 0, ('ImportFrom', 'is_lazy'): 0}


def canonical_ast_dump(node: object) -> str:
    """``ast.dump(node)`` as Python 3.13 and 3.14 format it, on every Python version.

    ``ast.dump`` itself differs between versions (3.12 adds ``type_params=[]``, 3.13 omits empty
    fields, 3.15 omits ``ctx=Load()``), and source hashes must not depend on the interpreter.
    """
    if isinstance(node, ast.AST):
        name = type(node).__name__
        fields = []
        for field in node._fields:
            value = getattr(node, field, None)
            if value is None and (name, field) not in _NULLABLE_REQUIRED_FIELDS:
                continue
            if isinstance(value, list) and not value:
                continue
            if (name, field) in _ADDED_FIELDS and value == _ADDED_FIELDS[name, field]:
                continue
            fields.append(f'{field}={canonical_ast_dump(value)}')
        return f'{name}({", ".join(fields)})'
    if isinstance(node, list):
        return f'[{", ".join(canonical_ast_dump(item) for item in node)}]'
    return repr(node)


def calculate_ast_hash(tree: ast.AST) -> str:
    return hashlib.sha256(canonical_ast_dump(tree).encode()).hexdigest()


@functools.cache
def calculate_cls_source_hash(cls: type) -> str:
    return calculate_ast_hash(get_source_ast_tree(cls))


def calculate_dict_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def calculate_array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def calculate_string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
