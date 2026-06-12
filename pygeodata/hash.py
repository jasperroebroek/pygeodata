from __future__ import annotations

import ast
import functools
import hashlib
import json
from typing import TYPE_CHECKING

from pygeodata.ast import get_source_ast_tree

if TYPE_CHECKING:
    import numpy as np


@functools.cache
def calculate_cls_source_hash(cls: type) -> str:
    tree = get_source_ast_tree(cls)
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def calculate_dict_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def calculate_array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def calculate_string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()
