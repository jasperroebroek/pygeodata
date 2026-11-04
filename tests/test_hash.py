import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

import pygeodata
from pygeodata.hash import _ADDED_FIELDS, _NULLABLE_REQUIRED_FIELDS, calculate_ast_hash, canonical_ast_dump

# The fields of every AST node on Python 3.14, the reference version of the canonical dump.
AST_FIELDS_3_14 = json.loads((Path(__file__).parent / 'fixtures' / 'ast_fields_3_14.json').read_text())
ON_REFERENCE_VERSION = sys.version_info[:2] in {(3, 13), (3, 14)}

# Parses on every supported Python version; covers optional, nullable-required and empty-list fields.
SAMPLE = """
@decorator(key=None)
class Sample(Base, metaclass=Meta):
    x: int = 1
    y = None
    z = ...

    def method(self, a, /, b=2, *args, c, d=None, **kwargs) -> str:
        import os.path as osp
        from . import sibling
        match a:
            case None:
                return f"{b!r:>{c}} {'q'}"
            case [1, *rest] if rest:
                pass
            case {'k': v, **others}:
                pass
        try:
            yield from (i for i in range(3) if i)
        except* ValueError as err:
            raise RuntimeError from err
        lambda: (yield)
        return b'' if b else 0.5j
"""

SAMPLE_HASH = '5ea2b4a3dc8bb6d25cba1ecf67c3c040991eef8e861963a7416bd2852f232910'


def test_sample_hash_is_pinned() -> None:
    """Identical on every Python version; a change here invalidates every cache entry."""
    assert calculate_ast_hash(ast.parse(SAMPLE)) == SAMPLE_HASH


def test_calculate_ast_hash_is_sha256_of_canonical_dump() -> None:
    tree = ast.parse(SAMPLE)
    assert calculate_ast_hash(tree) == hashlib.sha256(canonical_ast_dump(tree).encode()).hexdigest()


@pytest.mark.skipif(not ON_REFERENCE_VERSION, reason='the canonical dump is ast.dump of Python 3.13 and 3.14')
@pytest.mark.parametrize(
    'path',
    [*sorted(Path(pygeodata.__file__).parent.rglob('*.py')), Path(__file__)],
    ids=lambda p: p.name,
)
def test_canonical_dump_matches_ast_dump(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    assert canonical_ast_dump(tree) == ast.dump(tree)


@pytest.mark.skipif(not ON_REFERENCE_VERSION, reason='the canonical dump is ast.dump of Python 3.13 and 3.14')
def test_canonical_dump_matches_ast_dump_on_sample() -> None:
    tree = ast.parse(SAMPLE)
    assert canonical_ast_dump(tree) == ast.dump(tree)


@pytest.mark.skipif(sys.version_info < (3, 13), reason='AST field types are introspectable from 3.13')
def test_nullable_required_fields_match_cpython() -> None:
    nullable_required = {
        (cls.__name__, field)
        for cls in vars(ast).values()
        # Constant's subclasses (Ellipsis, Num, ...) are deprecated aliases, gone in 3.14
        if isinstance(cls, type) and issubclass(cls, ast.AST) and ast.Constant not in cls.__mro__[1:]
        for field, field_type in getattr(cls, '_field_types', {}).items()
        if field_type is object
    }
    if sys.version_info >= (3, 14):
        assert nullable_required == _NULLABLE_REQUIRED_FIELDS
    else:  # Interpolation arrived in 3.14
        assert nullable_required < _NULLABLE_REQUIRED_FIELDS


def test_fields_added_after_3_14_are_listed() -> None:
    """A new Python adding a field to an existing node fails here: add it to ``_ADDED_FIELDS``."""
    added = {
        (cls.__name__, field)
        for cls in vars(ast).values()
        if isinstance(cls, type) and issubclass(cls, ast.AST) and cls.__name__ in AST_FIELDS_3_14
        for field in cls._fields
        if field not in AST_FIELDS_3_14[cls.__name__]
    }
    assert added <= _ADDED_FIELDS.keys()


@pytest.mark.skipif(sys.version_info < (3, 15), reason='lazy imports are new in Python 3.15')
def test_added_field_changes_hash_when_used() -> None:
    assert calculate_ast_hash(ast.parse('lazy import os')) != calculate_ast_hash(ast.parse('import os'))
