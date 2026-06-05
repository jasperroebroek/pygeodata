from pygeodata.formatting.html import format_html_block, format_html_inline
from tests.formatting.helpers import Color, make_artifact


def test_html_inline_int() -> None:
    assert format_html_inline(42) == '42'


def test_html_inline_str_plain() -> None:
    assert format_html_inline('hello') == 'hello'


def test_html_inline_str_html_special_chars_escaped() -> None:
    result = format_html_inline('<b>bold</b>')
    assert '&lt;' in result
    assert '&gt;' in result
    assert '<b>' not in result


def test_html_inline_str_ampersand_escaped() -> None:
    assert '&amp;' in format_html_inline('a & b')


def test_html_inline_bytes_rendered_as_hex() -> None:
    assert format_html_inline(b'\xff\x00') == 'ff00'


def test_html_inline_enum_renders_classname_and_member() -> None:
    assert format_html_inline(Color.RED) == 'Color.RED'
    assert format_html_inline(Color.GREEN) == 'Color.GREEN'


def test_html_inline_list_of_enums() -> None:
    result = format_html_inline([Color.RED, Color.GREEN])
    assert 'Color.RED' in result
    assert 'Color.GREEN' in result


def test_html_inline_dict_with_enum_value() -> None:
    assert format_html_inline({'color': Color.RED}) == r'{color: Color.RED}'


def test_html_inline_artifact_shows_class_name_only() -> None:
    a = make_artifact('DataMaskLoader', {'variables': ['CLIMATE', 'LAI'], 'filter': False})
    result = format_html_inline(a)
    assert 'DataMaskLoader' in result
    assert 'variables' not in result
    assert 'filter' not in result


def test_html_inline_nested_artifact_in_dict_not_expanded() -> None:
    a = make_artifact('MyLoader', {'n': 10})
    result = format_html_inline({'mask': a})
    assert 'MyLoader' in result
    assert 'n' not in result or result.count('n') <= 1


def test_html_inline_nested_artifact_in_list_not_expanded() -> None:
    assert 'SomeLoader' in format_html_inline([make_artifact('SomeLoader')])


def test_html_block_artifact_shows_class_name_only() -> None:
    a = make_artifact('DataMaskingLoader', {'variables': ['x'], 'filter': False})
    result = format_html_block(a)
    assert 'DataMaskingLoader' in result
    assert 'variables' not in result
    assert 'filter' not in result


def test_html_block_artifact_nested_in_dict_not_expanded() -> None:
    a = make_artifact('NestedLoader', {'x': 1})
    result = format_html_block({'mask': a})
    assert 'mask' in result
    assert 'NestedLoader' in result
    assert '&#39;x&#39;' not in result


def test_html_block_dict_with_enum_value() -> None:
    result = format_html_block({'color': Color.RED})
    assert 'color' in result
    assert 'Color.RED' in result


def test_html_block_list_of_enums() -> None:
    result = format_html_block([Color.RED, Color.GREEN])
    assert 'Color.RED' in result
    assert 'Color.GREEN' in result


def test_html_block_nested_list_of_enums_in_dict() -> None:
    result = format_html_block({'values': [Color.RED, Color.GREEN]})
    assert 'values' in result
    assert 'Color.RED' in result
    assert 'Color.GREEN' in result


def test_html_block_deeply_nested_dict() -> None:
    result = format_html_block({'outer': {'inner': 42}})
    assert 'outer' in result
    assert 'inner' in result
    assert '42' in result


def test_html_block_primitive_int() -> None:
    assert '99' in format_html_block(99)


def test_html_block_primitive_str() -> None:
    assert 'hello' in format_html_block('hello')
