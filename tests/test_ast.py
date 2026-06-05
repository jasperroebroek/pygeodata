import ast
import textwrap
from unittest.mock import patch

import pytest

from pygeodata.ast import (
    build_symbol_tables,
    find_names_in_ast_tree,
    get_module_ast_tree,
    get_source_ast_tree,
    resolve_reference_name,
)


def parse(code: str) -> ast.AST:
    return ast.parse(textwrap.dedent(code))


def test_build_symbol_tables_collects_imports_and_defs():
    tree = parse(
        """
        from a import Foo as Bar
        import pkg.mod as m

        class LocalOne:
            pass

        class LocalTwo:
            pass
        """,
    )

    tables = build_symbol_tables(tree)

    assert tables.imported_objects == {'Bar': 'Foo'}
    assert tables.module_aliases == {'m': 'pkg.mod'}
    assert tables.local_defs == {'LocalOne', 'LocalTwo'}


def test_resolve_reference_name_bare_local_class():
    tree = parse(
        """
        class Loader:
            pass

        x = Loader
        """,
    )
    tables = build_symbol_tables(tree)
    valid_names = {'Loader'}

    name_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == 'Loader' and isinstance(node.ctx, ast.Load)
    )

    assert resolve_reference_name(name_node, tables, valid_names) == 'Loader'


def test_resolve_reference_name_import_alias():
    tree = parse(
        """
        from a import Foo as Bar
        x = Bar
        """,
    )
    tables = build_symbol_tables(tree)
    valid_names = {'Foo'}

    name_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == 'Bar' and isinstance(node.ctx, ast.Load)
    )

    assert resolve_reference_name(name_node, tables, valid_names) == 'Foo'


def test_find_names_in_ast_tree_includes_bare_and_called_names():
    tree = parse(
        """
        from a import Loader as L
        import pkg.mod as m

        class Loader:
            pass

        x = Loader
        y = L()
        z = Loader.get_class_name()
        w = m.Loader.get_class_name()
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Loader'}) == {'Loader'}


def test_find_names_in_ast_tree_includes_bases_by_default():
    tree = parse(
        """
        class Base:
            pass

        class Child(Base):
            pass
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Base'}) == {'Base'}


def test_find_names_in_ast_tree_can_exclude_bases():
    tree = parse(
        """
        class Base:
            pass

        class Child(Base):
            pass
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Base'}, exclude_bases=True) == set()


def test_find_names_in_ast_tree_excluding_bases_keeps_body_references():
    tree = parse(
        """
        class Base:
            pass

        class Child(Base):
            def build(self):
                return Base()
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Base'}) == {'Base'}
    assert find_names_in_ast_tree(tree, tables, {'Base'}, exclude_bases=True) == {'Base'}


def test_find_names_in_ast_tree_excluding_bases_keeps_non_base_references():
    tree = parse(
        """
        class Base:
            pass

        class Helper:
            pass

        class Child(Base):
            x = Helper
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Base', 'Helper'}) == {'Base', 'Helper'}
    assert find_names_in_ast_tree(tree, tables, {'Base', 'Helper'}, exclude_bases=True) == {'Helper'}


def test_find_names_in_ast_tree_with_multiple_bases():
    tree = parse(
        """
        class Base1:
            pass

        class Base2:
            pass

        class Child(Base1, Base2):
            pass
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Base1', 'Base2'}) == {'Base1', 'Base2'}
    assert find_names_in_ast_tree(tree, tables, {'Base1', 'Base2'}, exclude_bases=True) == set()


def test_find_names_in_ast_tree_with_base_alias():
    tree = parse(
        """
        from a import Base as L

        class Child(L):
            pass
        """,
    )
    tables = build_symbol_tables(tree)

    assert find_names_in_ast_tree(tree, tables, {'Base'}) == {'Base'}
    assert find_names_in_ast_tree(tree, tables, {'Base'}, exclude_bases=True) == set()


def test_resolve_reference_name_returns_none_for_unresolved_import():
    tree = parse(
        """
        from a import Foo as Bar
        x = Bar
        """,
    )
    tables = build_symbol_tables(tree)
    name_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == 'Bar' and isinstance(node.ctx, ast.Load)
    )
    assert resolve_reference_name(name_node, tables, {'Other'}) is None


def test_resolve_reference_name_local_def_not_in_valid_names():
    tree = parse(
        """
        class Loader:
            pass

        x = Loader
        """,
    )
    tables = build_symbol_tables(tree)
    name_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == 'Loader' and isinstance(node.ctx, ast.Load)
    )
    assert resolve_reference_name(name_node, tables, {'Other'}) is None


def test_resolve_reference_name_nested_attribute_chain():
    tree = parse(
        """
        import a.b.c
        x = a.b.Loader
        """,
    )
    tables = build_symbol_tables(tree)
    attr_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == 'Loader'
    )
    assert resolve_reference_name(attr_node, tables, {'Loader'}) is None


def test_find_names_call_positional_arg():
    tree = parse(
        """
        class Loader:
            pass

        some_func(Loader)
        """,
    )
    tables = build_symbol_tables(tree)
    assert find_names_in_ast_tree(tree, tables, {'Loader'}) == {'Loader'}


def test_find_names_call_keyword_arg():
    tree = parse(
        """
        class Loader:
            pass

        some_func(spec=Loader)
        """,
    )
    tables = build_symbol_tables(tree)
    assert find_names_in_ast_tree(tree, tables, {'Loader'}) == {'Loader'}


def test_find_names_decorator_under_exclude_bases():
    tree = parse(
        """
        class Decorator:
            pass

        class Base:
            pass

        @Decorator
        class Child(Base):
            pass
        """,
    )
    tables = build_symbol_tables(tree)
    assert find_names_in_ast_tree(tree, tables, {'Decorator', 'Base'}, exclude_bases=True) == {'Decorator'}


def test_get_source_ast_tree_raises_on_missing_source():
    class Dummy:
        pass

    with patch('pygeodata.ast.get_source_code', side_effect=OSError('unavailable')):
        get_source_ast_tree.cache_clear()
        try:
            with pytest.raises(OSError, match='AST Parsing failed'):
                get_source_ast_tree(Dummy)
        finally:
            get_source_ast_tree.cache_clear()


def test_get_module_ast_tree_raises_when_no_module_file():
    class Dummy:
        pass

    with patch('pygeodata.ast.inspect.getmodule', return_value=None):
        get_module_ast_tree.cache_clear()
        try:
            with pytest.raises(OSError, match='Module file not available'):
                get_module_ast_tree(Dummy)
        finally:
            get_module_ast_tree.cache_clear()
