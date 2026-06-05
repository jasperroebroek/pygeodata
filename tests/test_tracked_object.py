import json
from collections.abc import Generator
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys
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
    """Basic sanity: initialize_registry writes source.json and source.py."""
    SimpleTrackedObject.update_registry()

    json_file = SimpleTrackedObject.resolve_registry_paths().registry_path
    py_file = SimpleTrackedObject.resolve_registry_paths().code_path

    assert json_file.exists()
    assert py_file.exists()

    data = json.loads(json_file.read_text())
    assert JSONKeys.SOURCE_HASH in data
    assert JSONKeys.DEPENDENCY_TREE_HASH in data
    assert JSONKeys.TREE in data


def test_initialize_registry_handles_real_call_cycle() -> None:
    """
    Realistic circular dependency: A calls B, B calls A.

    The AST-based get_call_dependencies will see A <-> B as a cycle.
    initialize_registry must terminate thanks to its visited guard.
    """
    # This should not raise RecursionError if init_registry
    # correctly tracks visited classes.
    C.update_registry()

    assert C.resolve_registry_paths().registry_path.exists()
    assert D.resolve_registry_paths().registry_path.exists()


def test_dependency_tree_no_self_inheritance(tmp_path: Path) -> None:
    tree = TrackedChild.get_dependency_tree()

    inh = tree['inheritance_dependencies']
    assert TrackedBase.get_class_name() in inh
    assert TrackedChild.get_class_name() not in inh


def test_duplicated_names() -> None:
    with pytest.raises(ValueError, match='Duplicate TrackedObject class name'):

        class DuplicateTracked(TrackedObject):
            pass
