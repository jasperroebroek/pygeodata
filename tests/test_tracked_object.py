import json
from collections.abc import Generator

import pytest

from pygeodata.config import JSONKeys
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryConstructor, TreeRegistryConstructor
from pygeodata.tracked_object import TrackedObject
from tests.fixtures.tracked_objects import Bar, C, D, Foo, SimpleTrackedObject, TrackedBase, TrackedChild


@pytest.fixture(autouse=True)
def restore_registry() -> Generator[None, None, None]:
    saved = dict(TrackedObject._registry)
    yield
    TrackedObject._registry = saved
    TrackedObject.clear_function_caches()


def test_tracked_object_get_class_name() -> None:
    assert SimpleTrackedObject.get_class_name() == 'SimpleTrackedObject'


def test_tracked_object_registry_and_find_object_class() -> None:
    assert TrackedObject.find_object_class('Foo') is Foo
    assert TrackedObject.find_object_class('Bar') is Bar
    assert TrackedObject.find_object_class('Baz') is None


def test_dependency_graph_no_self_inheritance_edges() -> None:
    """In the flat dependency graph, there should be no (cls, cls) inheritance edges."""
    graph = TrackedChild.get_dependency_graph()

    for src, dst in graph.inheritance_edges:
        assert src is not dst, 'Class must not inherit from itself in dependency_graph'


def test_initialize_registry_creates_files() -> None:
    """Basic sanity: update_registry writes source.py, source.json, and tree.json."""
    SimpleTrackedObject.update_registry()

    code_resolver = CodeRegistryConstructor.from_source_hash(calculate_cls_source_hash(SimpleTrackedObject))
    tree_resolver = TreeRegistryConstructor.from_dep_tree_hash(SimpleTrackedObject.get_dependency_tree_hash())

    assert code_resolver.meta_path.exists()
    assert code_resolver.source_path.exists()
    assert tree_resolver.tree_path.exists()

    meta = json.loads(code_resolver.meta_path.read_text())
    assert JSONKeys.SOURCE_HASH in meta

    tree_data = json.loads(tree_resolver.tree_path.read_text())
    assert JSONKeys.NODES in tree_data
    assert JSONKeys.CALL_EDGES in tree_data
    assert JSONKeys.INHERITANCE_EDGES in tree_data


def test_initialize_registry_handles_real_call_cycle() -> None:
    """
    Realistic circular dependency: A calls B, B calls A.

    The AST-based get_call_dependencies will see A <-> B as a cycle.
    initialize_registry must terminate thanks to its visited guard.
    """
    # This should not raise RecursionError if init_registry
    # correctly tracks visited classes.
    C.update_registry()

    c_tree = TreeRegistryConstructor.from_dep_tree_hash(C.get_dependency_tree_hash())
    d_tree = TreeRegistryConstructor.from_dep_tree_hash(D.get_dependency_tree_hash())
    assert c_tree.tree_path.exists()
    assert d_tree.tree_path.exists()


def test_dependency_tree_no_self_inheritance() -> None:
    result = TrackedChild.get_dependency_tree()
    child_name = TrackedChild.get_class_name()
    base_name = TrackedBase.get_class_name()
    assert [child_name, base_name] in result.inheritance_edges
    assert [child_name, child_name] not in result.inheritance_edges


def test_duplicated_names() -> None:
    with pytest.raises(ValueError, match='Duplicate TrackedObject class name'):

        class DuplicateTracked(TrackedObject):
            pass
