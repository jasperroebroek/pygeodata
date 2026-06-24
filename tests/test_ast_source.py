import ast
import json
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from pygeodata.ast import get_parsed_source_code, get_source_ast_tree, get_source_code
from pygeodata.config import JSONKeys
from pygeodata.data import Data
from pygeodata.figure import Figure
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.hash import calculate_cls_source_hash as _hash
from pygeodata.paths import CodeRegistryPathConstructor, TreeRegistryPathConstructor
from tests.fixtures.data import Child, CircularLoader, HardcodedDependencyLoader, LoaderA, LoaderB, LoaderD, Parent


def test_ast_hunter_finds_all_dependencies() -> None:
    HardcodedDependencyLoader.get_all_dependencies.cache_clear()

    deps = HardcodedDependencyLoader.get_all_dependencies()
    assert LoaderA in deps
    assert HardcodedDependencyLoader not in deps


def test_ast_does_not_include_self_as_dependency() -> None:
    LoaderA.get_all_dependencies.cache_clear()
    assert LoaderA not in LoaderA.get_all_dependencies()


def test_ast_includes_parent_class_in_mro() -> None:
    Child.get_all_dependencies.cache_clear()
    assert Parent in Child.get_all_dependencies()


@patch('pygeodata.hash.get_source_ast_tree')
def test_ast_ignores_formatting_and_comments(mock_ast_tree: MagicMock) -> None:
    calculate_cls_source_hash.cache_clear()
    mock_ast_tree.return_value = ast.parse(
        textwrap.dedent("""
        class MyLoader(Data):
            # TODO: fix this
            def process(self):

                x = 5
                return x
    """),
    )
    h1 = calculate_cls_source_hash(LoaderA)

    calculate_cls_source_hash.cache_clear()
    mock_ast_tree.return_value = ast.parse(
        textwrap.dedent("""
        class MyLoader(Data):
            def process(self):
                x = 5
                return x
    """),
    )
    h2 = calculate_cls_source_hash(LoaderA)

    assert h1 == h2


@patch('pygeodata.hash.get_source_ast_tree')
def test_ast_changes_when_logic_changes(mock_ast_tree: MagicMock) -> None:
    calculate_cls_source_hash.cache_clear()
    mock_ast_tree.return_value = ast.parse(
        textwrap.dedent("""
        class MyLoader(Data):
            def process(self):
                x = 5
                return x
    """),
    )
    h1 = calculate_cls_source_hash(LoaderA)

    calculate_cls_source_hash.cache_clear()
    mock_ast_tree.return_value = ast.parse(
        textwrap.dedent("""
        class MyLoader(Data):
            def process(self):
                x = 99
                return x
    """),
    )
    h2 = calculate_cls_source_hash(LoaderA)

    assert h1 != h2


@patch('inspect.getsource')
def test_get_source_metadata_raises_on_repl_source(mock_getsource: MagicMock) -> None:
    """In a REPL/notebook, getsource raises OSError. We expect a clear user error."""
    LoaderA.clear_function_caches()
    get_source_code.cache_clear()
    get_parsed_source_code.cache_clear()
    get_source_ast_tree.cache_clear()
    mock_getsource.side_effect = OSError('source unavailable')
    with pytest.raises(OSError, match='AST Parsing failed'):
        get_parsed_source_code(LoaderA)


def test_save_and_validate_code_state() -> None:
    LoaderA.clear_function_caches()
    LoaderA.write_registry()
    assert LoaderA.is_registry_valid()


def test_code_state_invalid_when_no_file() -> None:
    assert not LoaderA.is_registry_valid()


def test_code_state_invalid_when_tree_file_deleted() -> None:
    LoaderA.write_registry()
    tree_resolver = TreeRegistryPathConstructor.from_dep_tree_hash(LoaderA.get_dependency_tree_hash())
    tree_resolver.tree_path.unlink()
    assert not LoaderA.is_registry_valid()


def test_initialize_class_code_state_heals_missing_tree_file() -> None:
    LoaderA.write_registry()
    tree_resolver = TreeRegistryPathConstructor.from_dep_tree_hash(LoaderA.get_dependency_tree_hash())
    tree_resolver.tree_path.unlink()

    assert not LoaderA.is_registry_valid()

    LoaderA.write_registry()
    assert LoaderA.is_registry_valid()


def test_initialize_class_code_state_cascades_to_all_dependencies() -> None:
    """
    If HardcodedDependencyLoader initializes its state, it must also initialize
    the state of the hidden 'LoaderA'.
    """
    assert not HardcodedDependencyLoader.is_registry_valid()
    assert not LoaderA.is_registry_valid()

    HardcodedDependencyLoader.update_registry()

    assert HardcodedDependencyLoader.is_registry_valid()
    assert LoaderA.is_registry_valid()


def test_state_hash_deterministic() -> None:
    LoaderA.get_dependency_tree_hash.cache_clear()
    h1 = LoaderA(year=2000).get_dependency_tree_hash()
    h2 = LoaderA(year=2000).get_dependency_tree_hash()
    assert h1 == h2


def test_state_hash_differs_by_param() -> None:
    LoaderA.get_dependency_tree_hash.cache_clear()
    h1 = LoaderA(year=2000).get_instance_hash()
    h2 = LoaderA(year=2001).get_instance_hash()
    assert h1 != h2


def test_state_hash_includes_upstream_loader() -> None:
    """Changing an upstream loader's parameter must cascade to change the downstream hash."""
    LoaderB.get_dependency_tree_hash.cache_clear()
    LoaderA.get_dependency_tree_hash.cache_clear()

    h1 = LoaderB(features=[LoaderA(year=2000)]).get_instance_hash()
    h2 = LoaderB(features=[LoaderA(year=2001)]).get_instance_hash()
    assert h1 != h2


def test_state_hash_mixed_list() -> None:
    """A list with a Data and a string should resolve correctly without throwing AttributeError."""
    h1 = LoaderD(items=[LoaderA(year=2000), 'europe']).get_instance_hash()
    h2 = LoaderD(items=[LoaderA(year=2000), 'asia']).get_instance_hash()

    assert isinstance(h1, str)
    assert len(h1) == 64
    assert h1 != h2


def test_dependency_tree_hash_stable() -> None:
    h1 = LoaderA.get_dependency_tree_hash()
    h2 = LoaderA.get_dependency_tree_hash()
    assert h1 == h2


@patch('inspect.getsource')
def test_dependency_tree_hash_changes_with_code(mock_getsource: MagicMock) -> None:
    """Simulate a source code change and verify that the dependency tree hash changes."""
    LoaderA.clear_function_caches()

    mock_getsource.return_value = 'class LoaderA(Data): x=1'

    original_hash = LoaderA.get_dependency_tree_hash()

    LoaderA.clear_function_caches()
    get_source_code.cache_clear()
    calculate_cls_source_hash.cache_clear()
    get_source_ast_tree.cache_clear()

    mock_getsource.return_value = 'class LoaderA(Data): x=99'

    new_hash = LoaderA.get_dependency_tree_hash()
    assert original_hash != new_hash


def test_circular_dependency_in_hierarchy_hash_does_not_infinite_loop() -> None:
    """If a class somehow depends on itself, get_dependency_tree_hash should return 'circular' safely."""
    CircularLoader.clear_function_caches()

    class_hash = CircularLoader.get_dependency_tree_hash()

    assert isinstance(class_hash, str)
    assert len(class_hash) > 0


def test_call_dependencies_distinct_from_inheritance() -> None:
    Child.get_call_dependencies.cache_clear()
    Child.get_inheritance_dependencies.cache_clear()
    assert Parent not in Child.get_call_dependencies()
    assert Parent in Child.get_inheritance_dependencies()


def test_call_dependencies_finds_hardcoded_instantiation() -> None:
    HardcodedDependencyLoader.get_call_dependencies.cache_clear()
    assert LoaderA in HardcodedDependencyLoader.get_call_dependencies()


def test_dependency_tree_hash_same_across_instances_with_different_params() -> None:
    LoaderA.get_dependency_tree_hash.cache_clear()
    h1 = LoaderA(year=2000).get_dependency_tree_hash()
    h2 = LoaderA(year=9999).get_dependency_tree_hash()
    assert h1 == h2


def test_write_registry_content() -> None:
    LoaderA.clear_function_caches()
    LoaderA.write_registry()

    code_resolver = CodeRegistryPathConstructor.from_source_hash(_hash(LoaderA))
    data = json.loads(code_resolver.meta_path.read_text())
    assert data[JSONKeys.CLASS_NAME] == 'LoaderA'
    assert JSONKeys.SOURCE_HASH in data
    tree_resolver = TreeRegistryPathConstructor.from_dep_tree_hash(LoaderA.get_dependency_tree_hash())
    tree_data = json.loads(tree_resolver.tree_path.read_text())
    assert JSONKeys.NODES in tree_data
    assert JSONKeys.CALL_EDGES in tree_data
    assert JSONKeys.INHERITANCE_EDGES in tree_data


def test_write_registry_writes_code_file() -> None:
    LoaderA.clear_function_caches()
    LoaderA.write_registry()

    code_resolver = CodeRegistryPathConstructor.from_source_hash(_hash(LoaderA))
    assert code_resolver.source_path.exists()
    assert 'LoaderA' in code_resolver.source_path.read_text()


def test_get_registered_objects_returns_correct_type_family() -> None:
    data_objects = Data.get_registered_objects()
    figure_objects = Figure.get_registered_objects()

    assert LoaderA in data_objects
    assert not any(issubclass(cls, Figure) for cls in data_objects)
    assert not any(issubclass(cls, Data) for cls in figure_objects)


def test_has_dependencies_true_for_loader_with_deps() -> None:
    assert HardcodedDependencyLoader.has_dependencies()


def test_has_dependencies_false_for_isolated_loader() -> None:
    LoaderA.get_all_dependencies.cache_clear()
    assert not LoaderA.has_dependencies()


def test_find_object_class_returns_registered_class() -> None:
    assert LoaderA.find_object_class('LoaderA') is LoaderA


def test_find_object_class_returns_none_for_unknown() -> None:
    assert LoaderA.find_object_class('NonExistentClass') is None
