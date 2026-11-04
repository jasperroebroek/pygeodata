"""Dependency trees over neutral class descriptions.

A :class:`ClassFacts` describes one tracked class without referring to a live class object, so the
same functions serve an imported class (:meth:`~pygeodata.tracked_object.TrackedObject.get_facts`)
and one read from source without importing it (:mod:`pygeodata.static`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pygeodata.ast import find_names_in_ast_tree
from pygeodata.enums import JSONKeys
from pygeodata.hash import calculate_dict_hash

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable

    from pygeodata.graph_types import SymbolTables

    # class name → its facts; raises KeyError for a name it does not know
    FactsLookup = Callable[[str], 'ClassFacts']


@dataclass(frozen=True)
class ClassFacts:
    name: str
    object_type: str
    source_hash: str
    # tracked classes referenced in the class body, decorators and keywords
    call_deps: frozenset[str]
    # direct tracked bases; inheritance dependencies are their transitive closure
    bases: frozenset[str]


def call_dependencies(class_tree: ast.AST, tables: SymbolTables, tracked: set[str], name: str) -> frozenset[str]:
    """Tracked class names used by the class, resolved through its module's symbol tables."""
    return frozenset(find_names_in_ast_tree(class_tree, tables, tracked, exclude_bases=True) - {name})


def inheritance_dependencies(facts: FactsLookup, name: str) -> set[str]:
    """All tracked ancestors of ``name``."""
    ancestors: set[str] = set()
    pending = list(facts(name).bases)
    while pending:
        base = pending.pop()
        if base not in ancestors:
            ancestors.add(base)
            pending.extend(facts(base).bases)
    return ancestors


@dataclass
class DependencyTree:
    nodes: set[str]
    call_edges: set[tuple[str, str]]
    inheritance_edges: set[tuple[str, str]]


def dependency_tree(facts: FactsLookup, name: str) -> DependencyTree:
    """Everything ``name`` depends on, transitively, through calls and inheritance."""
    tree = DependencyTree(nodes=set(), call_edges=set(), inheritance_edges=set())
    pending = [name]
    while pending:
        current = pending.pop()
        if current in tree.nodes:
            continue
        tree.nodes.add(current)
        for dep in facts(current).call_deps:
            tree.call_edges.add((current, dep))
            pending.append(dep)
        for dep in inheritance_dependencies(facts, current):
            tree.inheritance_edges.add((current, dep))
            pending.append(dep)
    return tree


def dependency_tree_dict(facts: FactsLookup, name: str) -> dict:
    """The hashed form of :func:`dependency_tree`; also the content of a ``tree.json`` snapshot."""
    tree = dependency_tree(facts, name)
    return {
        JSONKeys.NODES: {
            node: {'hash': facts(node).source_hash, JSONKeys.OBJECT_TYPE: facts(node).object_type}
            for node in tree.nodes
        },
        JSONKeys.CALL_EDGES: sorted([s, t] for s, t in tree.call_edges),
        JSONKeys.INHERITANCE_EDGES: sorted([s, t] for s, t in tree.inheritance_edges),
    }


def dependency_tree_hash(facts: FactsLookup, name: str) -> str:
    return calculate_dict_hash(dependency_tree_dict(facts, name))
