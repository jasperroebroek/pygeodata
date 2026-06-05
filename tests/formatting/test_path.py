from pathlib import Path

import pytest

from pygeodata.formatting.path import format_path
from pygeodata.formatting.path_simplified import format_path_simplified
from tests.formatting.helpers import Color, IntColor, Size, make_artifact

BOTH = [
    pytest.param(format_path, id='format_path'),
    pytest.param(format_path_simplified, id='format_path_simplified'),
]


@pytest.mark.parametrize('fn', BOTH)
@pytest.mark.parametrize('value,expected', [
    (42, '42'),
    (0, '0'),
    (-5, '-5'),
    ('hello', 'hello'),
    ('hello_world', 'hello_world'),
])
def test_scalars_passthrough(fn, value, expected) -> None:
    assert fn(value) == expected


@pytest.mark.parametrize('fn', BOTH)
def test_str_with_slash_hashed(fn) -> None:
    assert '/' not in fn('hello/world')


@pytest.mark.parametrize('fn', BOTH)
def test_path_object_no_slashes(fn) -> None:
    result = fn(Path('data/file.tif'))
    assert '/' not in result
    assert len(result) > 0


@pytest.mark.parametrize('fn', BOTH)
def test_artifact_renders_class_name(fn) -> None:
    assert fn(make_artifact('MyLoader')) == 'MyLoader'


@pytest.mark.parametrize('fn', BOTH)
def test_different_artifact_classes_differ(fn) -> None:
    assert fn(make_artifact('LoaderA')) != fn(make_artifact('LoaderB'))


@pytest.mark.parametrize('fn', BOTH)
def test_artifact_same_class_different_params_collide_by_design(fn) -> None:
    """format_path collapses Artifact to class name — params don't affect the output."""
    assert fn(make_artifact('MyLoader', {'n': 10})) == fn(make_artifact('MyLoader', {'n': 99}))


@pytest.mark.parametrize('fn', BOTH)
def test_list_renders_as_string(fn) -> None:
    assert isinstance(fn([1, 2, 3]), str)


@pytest.mark.parametrize('fn', BOTH)
def test_list_order_matters(fn) -> None:
    assert fn([1, 2]) != fn([2, 1])


@pytest.mark.parametrize('fn', BOTH)
def test_dict_renders_as_string(fn) -> None:
    assert isinstance(fn({'a': 1}), str)


def test_format_path_enum() -> None:
    assert format_path(Color.RED) == 'Color[RED]'
    assert format_path(Color.GREEN) == 'Color[GREEN]'
    assert format_path(IntColor.ONE) == 'IntColor[ONE]'


def test_format_path_enum_in_list() -> None:
    result = format_path([Color.RED, Color.GREEN])
    assert 'Color[RED]' in result
    assert 'Color[GREEN]' in result


def test_format_path_enum_in_dict() -> None:
    assert 'Color[RED]' in format_path({'color': Color.RED})


def test_format_path_enum_in_nested_dict() -> None:
    assert 'Color[GREEN]' in format_path({'outer': {'color': Color.GREEN}})


def test_format_path_str_with_colon_hashed() -> None:
    assert ':' not in format_path('EPSG:4326')


def test_format_path_bytes() -> None:
    b = b'\xff'
    assert format_path(b) != format_path('ff')
    assert format_path(b) != format_path(b.hex())


def test_format_path_simplified_enum() -> None:
    assert format_path_simplified(Color.RED) == 'Color--RED--'
    assert format_path_simplified(IntColor.ONE) == 'IntColor--ONE--'


def test_format_path_simplified_artifact() -> None:
    assert format_path_simplified(make_artifact('DataMaskLoader')) == 'DataMaskLoader'


def test_format_path_simplified_list_hashed() -> None:
    result = format_path_simplified([1, 2, 3])
    assert '[1, 2, 3]' not in result
    assert len(result) > 0


def test_format_path_simplified_dict_hashed() -> None:
    result = format_path_simplified({'a': 1, 'b': 2})
    assert '{' not in result


def test_format_path_simplified_str_with_slash_hashed() -> None:
    assert '/' not in format_path_simplified('hello/world')
