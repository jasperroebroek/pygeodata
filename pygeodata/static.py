"""The project's tracked classes and their dependency tree hashes, read from source without importing it.

A class is tracked when its bases reach one of pygeodata's base classes, directly or through other
project classes; bases are resolved through each file's imports. Its facts are derived by the same
functions as an imported class's (:mod:`pygeodata.dependencies`), so both give the same hashes.

Not seen: classes created at runtime or defined inside functions, and tracked classes from
installed packages. Such classes are unknown here, never reported as current.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pygeodata.analysis import ClassRecord, ProjectAnalysis, tree_hashes_of
from pygeodata.ast import build_symbol_tables, parse_class_source
from pygeodata.dependencies import ClassFacts, call_dependencies
from pygeodata.hash import calculate_ast_hash
from pygeodata.project import iter_project_files
from pygeodata.tracked_object import TrackedObject

if TYPE_CHECKING:
    from pathlib import Path

    from pygeodata.graph_types import SymbolTables

# pygeodata's own TrackedObject classes a project class can derive from; each is its own object type
BASE_CLASSES = frozenset({'TrackedObject', 'Artifact', 'Data', 'Table', 'Figure'})


@dataclass(frozen=True)
class _Ref:
    """What a base class expression names: ``module`` is None for a name defined or unresolved locally."""

    module: str | None
    name: str


@dataclass
class _Candidate:
    node: ast.ClassDef
    path: Path
    lines: list[str]
    tables: SymbolTables
    bases: list[_Ref]


def _module_classes(body: list[ast.stmt]) -> list[ast.ClassDef]:
    """Class definitions run on import: at module level, also inside ``if``/``try``/``with`` blocks."""
    found: list[ast.ClassDef] = []
    for stmt in body:
        if isinstance(stmt, ast.ClassDef):
            found.append(stmt)
        elif isinstance(stmt, (ast.If, ast.With)):
            found.extend(_module_classes(stmt.body))
            found.extend(_module_classes(getattr(stmt, 'orelse', [])))
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, *(h.body for h in stmt.handlers), stmt.orelse, stmt.finalbody):
                found.extend(_module_classes(block))
    return found


def _imports(tree: ast.Module) -> tuple[dict[str, _Ref], dict[str, str]]:
    """Names imported from modules, and module aliases."""
    names: dict[str, _Ref] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = '.' * node.level + (node.module or '')
            for alias in node.names:
                names[alias.asname or alias.name] = _Ref(module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    modules[alias.name.split('.', 1)[0]] = alias.name.split('.', 1)[0]
    return names, modules


def _dotted(node: ast.expr) -> list[str] | None:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        parent = _dotted(node.value)
        return [*parent, node.attr] if parent is not None else None
    return None


def _base_ref(node: ast.expr, names: dict[str, _Ref], modules: dict[str, str]) -> _Ref | None:
    if isinstance(node, ast.Subscript):  # e.g. Data[xr.Dataset]
        node = node.value
    parts = _dotted(node)
    if parts is None:
        return None
    if len(parts) == 1:
        return names.get(parts[0], _Ref(None, parts[0]))
    if parts[0] in modules:
        return _Ref('.'.join([modules[parts[0]], *parts[1:-1]]), parts[-1])
    return None


def _is_pygeodata_base(ref: _Ref) -> bool:
    return ref.module is not None and ref.module.split('.', 1)[0] == 'pygeodata' and ref.name in BASE_CLASSES


def _read_candidates(root: Path, problems: list[str]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for path in iter_project_files(root):
        try:
            text = path.read_text(encoding='utf-8')
            tree = ast.parse(text, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            problems.append(f'{path}: {exc}')
            continue
        names, modules = _imports(tree)
        tables = build_symbol_tables(tree)
        lines = text.splitlines(keepends=True)
        for node in _module_classes(tree.body):
            bases = [ref for base in node.bases if (ref := _base_ref(base, names, modules)) is not None]
            candidates.append(_Candidate(node=node, path=path, lines=lines, tables=tables, bases=bases))
    return candidates


def _object_types(candidates: list[_Candidate]) -> dict[str, str]:
    """Tracked class name → object type, found by following bases until nothing changes.

    The object type comes from the first base that is tracked, as in ``TrackedObject.__init_subclass__``.
    """
    object_types: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for candidate in candidates:
            name = candidate.node.name
            if name in object_types:
                continue
            for ref in candidate.bases:
                if _is_pygeodata_base(ref):
                    object_types[name] = ref.name
                elif ref.module is None or ref.module.split('.', 1)[0] != 'pygeodata':
                    if ref.name not in object_types:
                        continue
                    object_types[name] = object_types[ref.name]
                else:
                    continue
                changed = True
                break
    return object_types


def _first_line(node: ast.ClassDef) -> int:
    return min([node.lineno, *(d.lineno for d in node.decorator_list)])


def _class_source(candidate: _Candidate) -> str:
    """The class's lines from its first decorator, as ``inspect.getsource`` returns them."""
    node = candidate.node
    return ''.join(candidate.lines[_first_line(node) - 1 : node.end_lineno])


def analyse_project(root: Path) -> ProjectAnalysis:
    problems: list[str] = []
    candidates = _read_candidates(root, problems)
    object_types = _object_types(candidates)

    tracked = [c for c in candidates if c.node.name in object_types]
    counts: dict[str, int] = {}
    for candidate in tracked:
        counts[candidate.node.name] = counts.get(candidate.node.name, 0) + 1
    duplicates = {name for name, n in counts.items() if n > 1}
    problems.extend(f'{name}: defined more than once; left out' for name in sorted(duplicates))

    names = set(object_types) - duplicates
    all_facts: dict[str, ClassFacts] = {}
    locations: dict[str, tuple[str, int]] = {}
    for candidate in tracked:
        name = candidate.node.name
        if name in duplicates:
            continue
        class_tree = parse_class_source(_class_source(candidate))
        facts = ClassFacts(
            name=name,
            object_type=object_types[name],
            source_hash=calculate_ast_hash(class_tree),
            call_deps=call_dependencies(class_tree, candidate.tables, names, name),
            bases=frozenset(ref.name for ref in candidate.bases if not _is_pygeodata_base(ref) and ref.name in names),
        )
        all_facts[name] = facts
        locations[name] = (str(candidate.path), _first_line(candidate.node))
    hashes = tree_hashes_of(all_facts)
    classes = {
        name: ClassRecord(facts=facts, path=locations[name][0], lineno=locations[name][1], tree_hash=hashes[name])
        for name, facts in all_facts.items()
    }
    return ProjectAnalysis(mode='static', root=str(root), classes=classes, problems=problems)


def compare_with_imports(analysis: ProjectAnalysis) -> list[str]:
    """Where the imported classes and those read from source disagree, one message per class."""
    imported = {cls.get_class_name(): cls for cls in TrackedObject.get_registered_objects()}
    messages: list[str] = []
    for name in sorted(analysis.classes.keys() | imported.keys()):
        if name not in imported:
            messages.append(f'{name}: found in source but not imported')
        elif name not in analysis.classes:
            messages.append(f'{name}: imported but not found in source')
        elif imported[name].get_dependency_tree_hash() != analysis.tree_hashes[name]:
            messages.append(f'{name}: imported and source dependency trees differ')
    return messages
