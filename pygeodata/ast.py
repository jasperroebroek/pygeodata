import ast
import functools
import inspect
import textwrap
from pathlib import Path

from pygeodata.types import SymbolTables


@functools.cache
def get_source_code(cls: type) -> str:
    return textwrap.dedent(inspect.getsource(cls))


@functools.cache
def get_module_ast_tree(cls: type) -> ast.AST:
    module = inspect.getmodule(cls)
    if module is None or not hasattr(module, '__file__') or module.__file__ is None:
        raise OSError('Module file not available')

    source = Path(module.__file__).read_text(encoding='utf-8')
    return ast.parse(source, filename=module.__file__)


@functools.cache
def get_source_ast_tree(cls: type) -> ast.AST:
    try:
        source = get_source_code(cls)
        return ast.parse(source)
    except (TypeError, OSError) as err:
        raise OSError(
            'AST Parsing failed. Caching is disabled. You are likely in a REPL/Notebook environment. Use standard .py files.',
        ) from err


@functools.cache
def get_parsed_source_code(cls: type) -> str:
    return ast.unparse(get_source_ast_tree(cls))


def build_symbol_tables(tree: ast.AST) -> SymbolTables:
    imported_objects: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    local_defs: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == '*':
                    continue
                local_name = alias.asname or alias.name
                imported_objects[local_name] = alias.name

        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split('.', 1)[0]
                module_aliases[local_name] = alias.name

        elif isinstance(node, ast.ClassDef):
            local_defs.add(node.name)

    return SymbolTables(
        imported_objects=imported_objects,
        module_aliases=module_aliases,
        local_defs=local_defs,
    )


def resolve_reference_name(
    node: ast.AST,
    tables: SymbolTables,
    valid_names: set[str],
) -> str | None:
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        name = node.id

        if name in tables.imported_objects:
            resolved = tables.imported_objects[name]
            return resolved if resolved in valid_names else None

        if name in tables.local_defs and name in valid_names:
            return name

        if name in valid_names:
            return name

        return None

    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        if isinstance(node.value, ast.Name):
            base = node.value.id

            if base in tables.imported_objects:
                resolved = tables.imported_objects[base]
                return resolved if resolved in valid_names else None

            if base in tables.module_aliases and node.attr in valid_names:
                return node.attr

            if base in tables.local_defs and base in valid_names:
                return base

        if isinstance(node.value, ast.Attribute):
            return resolve_reference_name(node.value, tables, valid_names)

    return None


def find_names_in_ast_tree(
    tree: ast.AST,
    tables: SymbolTables,
    valid_names: set[str],
    exclude_bases: bool = False,
    found: set[str] | None = None,
) -> set[str]:
    if found is None:
        found = set()

    if isinstance(tree, ast.Call):
        resolved = resolve_reference_name(tree.func, tables, valid_names)
        if resolved is not None:
            found.add(resolved)

        for arg in tree.args:
            resolved = resolve_reference_name(arg, tables, valid_names)
            if resolved is not None:
                found.add(resolved)

        for kw in tree.keywords:
            resolved = resolve_reference_name(kw.value, tables, valid_names)
            if resolved is not None:
                found.add(resolved)

    elif (isinstance(tree, ast.Name) and isinstance(tree.ctx, ast.Load)) or (
        isinstance(tree, ast.Attribute) and isinstance(tree.ctx, ast.Load)
    ):
        resolved = resolve_reference_name(tree, tables, valid_names)
        if resolved is not None:
            found.add(resolved)

    if isinstance(tree, ast.ClassDef) and exclude_bases:
        for decorator in tree.decorator_list:
            find_names_in_ast_tree(
                decorator,
                tables=tables,
                valid_names=valid_names,
                exclude_bases=exclude_bases,
                found=found,
            )
        for keyword in tree.keywords:
            find_names_in_ast_tree(
                keyword,
                tables=tables,
                valid_names=valid_names,
                exclude_bases=exclude_bases,
                found=found,
            )
        for stmt in tree.body:
            find_names_in_ast_tree(
                stmt,
                tables=tables,
                valid_names=valid_names,
                exclude_bases=exclude_bases,
                found=found,
            )
        return found

    for child in ast.iter_child_nodes(tree):
        find_names_in_ast_tree(
            child,
            tables=tables,
            valid_names=valid_names,
            exclude_bases=exclude_bases,
            found=found,
        )

    return found
