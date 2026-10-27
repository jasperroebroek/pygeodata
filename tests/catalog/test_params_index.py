from pygeodata.catalog.params_index import flatten_params
from pygeodata.catalog.types import LinkedEntry, ParamRow, ParamRowType
from pygeodata.config import JSONKeys


def rows(params: dict, linked: list | None = None) -> list[ParamRow]:
    return flatten_params(params, linked_entries=linked)


def test_scalar_int() -> None:
    result = rows({'year': 2020})
    assert len(result) == 1
    assert result[0].final_key == 'year'
    assert result[0].value_text == '2020'
    assert result[0].value_type == 'int'
    assert result[0].depth == 0


def test_scalar_string() -> None:
    result = rows({'region': 'europe'})
    assert result[0].value_text == 'europe'
    assert result[0].value_type == 'str'


def test_scalar_none() -> None:
    result = rows({'x': None})
    assert result[0].value_text == 'None'


def test_scalar_bool() -> None:
    result = rows({'flag': True})
    assert result[0].value_type == 'bool'


def test_path_no_scope() -> None:
    result = rows({'year': 2020})
    assert result[0].path == 'year'
    assert result[0].key_group == ''


def test_path_nested_dict() -> None:
    result = rows({'outer': {'inner': 42}})
    assert len(result) == 1
    r = result[0]
    assert r.final_key == 'inner'
    assert r.key_group == 'outer'
    assert r.path == 'outer.inner'
    assert r.depth == 1


def test_path_doubly_nested_dict() -> None:
    result = rows({'a': {'b': {'c': 1}}})
    assert result[0].path == 'a › b.c'
    assert result[0].depth == 2
    assert result[0].key_group == 'a › b'


def test_hidden_keys_skipped_at_root() -> None:
    params = {
        JSONKeys.CLASS_NAME: 'ShouldBeHidden',
        'year': 2020,
    }
    result = rows(params)
    keys = [r.final_key for r in result]
    assert 'year' in keys
    assert JSONKeys.CLASS_NAME not in keys


def test_hidden_keys_skipped_in_nested_dict() -> None:
    params = {'meta': {JSONKeys.STATE_HASH: 'abc', 'label': 'x'}}
    result = rows(params)
    keys = [r.final_key for r in result]
    assert 'label' in keys
    assert JSONKeys.STATE_HASH not in keys


def test_flat_scalar_list() -> None:
    result = rows({'years': [2019, 2020, 2021]})
    assert all(r.final_key == 'years[]' for r in result)
    assert all(r.row_type == ParamRowType.LIST_MEMBER for r in result)
    assert len(result) == 3
    values = [r.value_text for r in result]
    assert '2019' in values and '2020' in values and '2021' in values


def test_complex_list_indexed_keys() -> None:
    result = rows({'items': [{'a': 1}, {'a': 2}]})
    keys = [r.final_key for r in result]
    assert 'items[0]' in keys or any('a' in r.final_key for r in result)


def test_mixed_list_scalar_and_dict() -> None:
    result = rows({'mixed': [1, {'x': 2}]})
    # scalar gets emitted as indexed, dict expands
    assert len(result) >= 1


def test_data_ref_emits_row_and_linked_entry() -> None:
    linked: list[LinkedEntry] = []
    params = {
        'loader': {
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.PARAMS: {'year': 2020},
        },
    }
    result = rows(params, linked=linked)

    # First row is the data_ref itself
    ref_rows = [r for r in result if r.row_type == ParamRowType.DATA_REF]
    assert len(ref_rows) == 1
    assert ref_rows[0].value_text == 'MyLoader'
    assert ref_rows[0].final_key == 'loader'

    # Linked entry captured (no params_summary — derived at display time)
    assert len(linked) == 1
    assert linked[0].class_name == 'MyLoader'
    assert linked[0].param_name == 'loader'


def test_data_ref_child_params_also_flattened() -> None:
    params = {
        'loader': {
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.PARAMS: {'region': 'europe'},
        },
    }
    result = rows(params)
    keys = [r.final_key for r in result]
    assert 'region' in keys


def test_data_ref_hidden_keys_in_inner_params_skipped() -> None:
    params = {
        'loader': {
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.PARAMS: {JSONKeys.STATE_HASH: 'abc', 'label': 'x'},
        },
    }
    result = rows(params)
    keys = [r.final_key for r in result]
    assert JSONKeys.STATE_HASH not in keys
    assert 'label' in keys


def test_linked_entries_none_does_not_crash() -> None:
    params = {
        'loader': {
            JSONKeys.CLASS_NAME: 'MyLoader',
            JSONKeys.PARAMS: {'year': 2020},
        },
    }
    result = rows(params, linked=None)
    assert any(r.row_type == ParamRowType.DATA_REF for r in result)


def test_search_blob_contains_key_and_value() -> None:
    result = rows({'scenario': 'ssp126'})
    assert 'scenario' in result[0].search_blob
    assert 'ssp126' in result[0].search_blob


def test_search_blob_data_ref_contains_class_name() -> None:
    params = {
        'base': {
            JSONKeys.CLASS_NAME: 'BaseLoader',
            JSONKeys.PARAMS: {},
        },
    }
    result = rows(params)
    ref_row = next(r for r in result if r.row_type == ParamRowType.DATA_REF)
    assert 'baseloader' in ref_row.search_blob
    assert 'data_ref' in ref_row.search_blob


def test_class_ref_emits_one_row() -> None:
    """A tracked class passed as a param value must not be silently dropped."""
    params = {
        'feature_cls': {
            JSONKeys.CLASS_NAME: 'X',
            JSONKeys.DEPENDENCY_TREE_HASH: '728939a8',
        },
    }
    result = rows(params)
    assert len(result) == 1
    assert result[0].value_text == 'X'
    assert result[0].row_type == ParamRowType.CLASS_REF
    assert result[0].final_key == 'feature_cls'
    assert result[0].path == 'feature_cls'


def test_class_ref_nested_in_data_ref() -> None:
    params = {
        'spec': {
            JSONKeys.CLASS_NAME: 'FeatureSpec',
            JSONKeys.PARAMS: {
                'feature_cls': {
                    JSONKeys.CLASS_NAME: 'MLFeatureBioclim',
                    JSONKeys.DEPENDENCY_TREE_HASH: 'abc123',
                },
                'variable': 1,
            },
        },
    }
    result = rows(params)
    class_refs = [r for r in result if r.row_type == ParamRowType.CLASS_REF]
    assert len(class_refs) == 1
    assert class_refs[0].value_text == 'MLFeatureBioclim'
    assert class_refs[0].key_group == 'spec'
    # The sibling scalar and the wrapper are still emitted.
    assert {r.value_text for r in result} == {'FeatureSpec', 'MLFeatureBioclim', '1'}


def test_class_ref_search_blob_includes_class_name() -> None:
    params = {
        'loader': {
            JSONKeys.CLASS_NAME: 'BioclimEnsemble',
            JSONKeys.DEPENDENCY_TREE_HASH: 'deadbeef',
        },
    }
    blob = rows(params)[0].search_blob
    assert 'bioclimensemble' in blob
    assert 'class_ref' in blob


def test_class_ref_without_hash() -> None:
    result = rows({'cls': {JSONKeys.CLASS_NAME: 'NoHash'}})
    assert len(result) == 1
    assert result[0].value_text == 'NoHash'
    assert result[0].row_type == ParamRowType.CLASS_REF


def test_value_type_is_python_type_not_structural() -> None:
    """value_type carries the Python type; row_type carries the structural role."""
    result = rows({'years': [2019, 2020]})
    assert all(r.value_type == 'int' for r in result)
    assert all(r.row_type == ParamRowType.LIST_MEMBER for r in result)

    mixed = rows({'items': [{'a': 1}, 'plain']})
    scalar_row = next(r for r in mixed if r.value_text == 'plain')
    assert scalar_row.value_type == 'str'
    assert scalar_row.row_type == ParamRowType.VALUE


def test_empty_params() -> None:
    assert rows({}) == []


def test_tuple_treated_like_list() -> None:
    result = rows({'vals': (1, 2)})
    assert len(result) == 2
