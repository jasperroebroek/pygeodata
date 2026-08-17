import ast
import textwrap
from unittest.mock import MagicMock, patch

from pygeodata.config import JSONKeys
from pygeodata.formatting.json import format_json
from pygeodata.hash import calculate_cls_source_hash, calculate_dict_hash
from pygeodata.spec import SpecKeys
from tests.fixtures.data import ClassParamLoader, LoaderA


def _clear_hash_caches(*classes: type) -> None:
    """Drop the memoised source and dependency-tree hashes.

    ``get_dependency_tree_hash`` is ``functools.cache``d per class, so without this a
    test that mutates a class's source would compare a hash against itself and pass
    while asserting nothing.
    """
    calculate_cls_source_hash.cache_clear()
    for cls in classes:
        cls.clear_function_caches()


def test_tracked_class_serialises_with_dependency_tree_hash() -> None:
    result = format_json(LoaderA)

    assert result[JSONKeys.CLASS_NAME] == 'LoaderA'
    assert result[JSONKeys.DEPENDENCY_TREE_HASH] == LoaderA.get_dependency_tree_hash()


def test_untracked_class_still_serialises_to_dotted_name() -> None:
    class Plain:
        pass

    assert format_json(dict) == 'builtins.dict'
    assert format_json(Plain).endswith('.Plain')


def test_tracked_instance_still_serialises_as_instance() -> None:
    """Instances must keep routing to AllowsFormatting, not the class branch."""
    result = format_json(LoaderA(year=2000))

    assert JSONKeys.PARAMS in result
    assert JSONKeys.INSTANCE_HASH in result
    assert JSONKeys.DEPENDENCY_TREE_HASH not in result


@patch('pygeodata.hash.get_source_ast_tree')
def test_state_hash_changes_when_class_valued_param_changes(
    mock_ast_tree: MagicMock,
    sample_spatial_spec,
) -> None:
    """The regression: editing a class passed as a param must invalidate the cache.

    Asserts on ``get_state_hash`` because that is what ``is_cache_valid`` consults;
    ``get_params_hash`` is a grouping key and never gates cache validity.
    """
    artifact = ClassParamLoader(loader_cls=LoaderA)

    _clear_hash_caches(ClassParamLoader, LoaderA)
    mock_ast_tree.return_value = ast.parse(
        textwrap.dedent("""
        class LoaderA(Data):
            def _process(self, spec):
                return 1
    """),
    )
    before = artifact.get_state_hash(sample_spatial_spec)

    _clear_hash_caches(ClassParamLoader, LoaderA)
    mock_ast_tree.return_value = ast.parse(
        textwrap.dedent("""
        class LoaderA(Data):
            def _process(self, spec):
                return 2
    """),
    )
    after = artifact.get_state_hash(sample_spatial_spec)

    assert before != after


def test_state_hash_stable_without_class_valued_params(sample_spatial_spec) -> None:
    artifact = LoaderA(year=2000)

    assert artifact.get_params_as_json() == {'year': 2000}
    assert artifact.get_state_hash(sample_spatial_spec) == calculate_dict_hash(
        {
            JSONKeys.INSTANCE_HASH: calculate_dict_hash(
                {
                    JSONKeys.DEPENDENCY_TREE_HASH: LoaderA.get_dependency_tree_hash(),
                    JSONKeys.PARAMS: {'year': 2000},
                },
            ),
            SpecKeys.SPEC: sample_spatial_spec.to_dict(),
        },
    )
