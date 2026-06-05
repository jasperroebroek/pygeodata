from enum import Enum
from pathlib import Path

import pytest

from pygeodata.formatting.path import format_path
from pygeodata.formatting.path_simplified import format_path_simplified
from tests.formatting.helpers import Color, IntColor, Size, make_artifact

BOTH = [
    pytest.param(format_path, id='format_path'),
    pytest.param(format_path_simplified, id='format_path_simplified'),
]


def assert_no_collision(*values, fn):
    results = [fn(v) for v in values]
    assert len(results) == len(set(results)), f'Collision detected among: {list(zip(values, results))}'


@pytest.mark.parametrize('fn', BOTH)
@pytest.mark.parametrize(
    'string,value',
    [
        ('[1, 2]', [1, 2]),
        ('{a:1}', {'a': 1}),
        ('[]', []),
        ('{}', {}),
        ('Color[RED]', Color.RED),
        ('tmp/data', Path('tmp/data')),
    ],
)
def test_string_does_not_collide_with_type(fn, string, value) -> None:
    assert fn(string) != fn(value)


@pytest.mark.parametrize(
    'string,value',
    [
        ('{}', set()),
        ('[[1], 2]', [[1], 2]),
        ('{a:{b:1}}', {'a': {'b': 1}}),
        ('a:1', {'a': 1}),
        ('1, 2', [1, 2]),
        ('{1, 2}', {1, 2}),
        ('data/file.tif', Path('data/file.tif')),
    ],
)
def test_string_does_not_collide_with_type_full_only(string, value) -> None:
    assert format_path(string) != format_path(value)


def test_empty_containers_no_collision_simplified() -> None:
    assert_no_collision([], {}, set(), fn=format_path_simplified)


@pytest.mark.parametrize('fn', BOTH)
def test_empty_list_does_not_collide_with_empty_dict(fn) -> None:
    assert fn([]) != fn({})


@pytest.mark.parametrize(
    'a,b',
    [
        ([1, 2], {1, 2}),
        ([42], 42),
        (['hello'], 'hello'),
        ([[1, 2]], [1, 2]),
        ([{'a': 1}], {'a': 1}),
    ],
)
def test_list_does_not_collide_with_similar(a, b) -> None:
    assert format_path(a) != format_path(b)


def test_nested_list_ambiguity() -> None:
    assert_no_collision([1, [2, 3]], [[1, 2], 3], [1, 2, 3], fn=format_path)


@pytest.mark.parametrize('fn', BOTH)
def test_ordered_containers_no_collision(fn) -> None:
    assert_no_collision([1, 2], [2, 1], [1, 2, 3], fn=fn)


@pytest.mark.parametrize('fn', BOTH)
@pytest.mark.parametrize(
    'a,b',
    [
        (Color.RED, 'RED'),
        (Color.RED, Size.RED),
    ],
)
def test_enum_does_not_collide(fn, a, b) -> None:
    assert fn(a) != fn(b)


@pytest.mark.parametrize(
    'a,b',
    [
        (Color.RED, 1),
        (Color.RED, {'Color': 'RED'}),
        ([Color.RED], ['RED']),
        ([Color.RED], ['Color[RED]']),
    ],
)
def test_enum_does_not_collide_full_only(a, b) -> None:
    assert format_path(a) != format_path(b)


def test_enum_list_no_collision() -> None:
    assert_no_collision([Color.RED], ['RED'], ['Color[RED]'], fn=format_path)


def test_int_enum_vs_plain_int() -> None:
    assert format_path(IntColor.ONE) != format_path(1)


def test_enum_same_value_different_name() -> None:
    class Other(Enum):
        BLUE = 1

    assert format_path(Color.RED) != format_path(Other.BLUE)


@pytest.mark.parametrize(
    'a,b',
    [
        ({'a': 1, 'b': 2}, {'a': 2, 'b': 1}),
        ({'a': {'b': 'c'}}, {'a': 'b', 'c': ''}),
    ],
)
def test_dict_does_not_collide(a, b) -> None:
    assert format_path(a) != format_path(b)


@pytest.mark.parametrize(
    'a,b',
    [
        (1, True),
        ('', None),
        (None, []),
        (None, ''),
        (None, 0),
    ],
)
def test_scalar_does_not_collide(a, b) -> None:
    assert format_path(a) != format_path(b)


@pytest.mark.parametrize('string', ['MyLoader[abc]', 'MyLoader[someHash]'])
def test_artifact_does_not_collide_with_string(string) -> None:
    assert format_path(make_artifact('MyLoader')) != format_path(string)


def test_artifact_does_not_collide_with_dict() -> None:
    assert format_path(make_artifact('MyLoader')) != format_path({'class_name': 'MyLoader'})


@pytest.mark.parametrize('fn', BOTH)
def test_path_does_not_collide_with_string(fn) -> None:
    assert fn(Path('foo/bar')) != fn('foo/bar')


def test_bytes_does_not_collide_with_hex_string() -> None:
    b = b'\xff'
    assert format_path(b) != format_path('ff')
    assert format_path(b) != format_path(b.hex())
