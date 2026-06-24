import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.registry_browser.popups import (
    _inject_graph_links,
    _linkify_class_names,
    build_json_popup,
    build_source_popup,
)
from pygeodata.tracked_object import TrackedObject


# --- _linkify_class_names ---


def test_linkify_replaces_known_class() -> None:
    result = _linkify_class_names('class MyLoader:', frozenset({'MyLoader'}), 'Other')
    assert 'data-cls="MyLoader"' in result
    assert 'src-cls-link' in result


def test_linkify_excludes_current_class() -> None:
    result = _linkify_class_names('class MyLoader:', frozenset({'MyLoader'}), 'MyLoader')
    assert 'data-cls' not in result


def test_linkify_no_known_classes_passthrough() -> None:
    source = 'class MyLoader: pass'
    result = _linkify_class_names(source, frozenset(), 'Other')
    assert result == source


def test_linkify_longest_first() -> None:
    known = frozenset({'Loader', 'BioLoader'})
    result = _linkify_class_names('BioLoader', known, 'Other')
    assert result.count('data-cls') == 1
    assert 'data-cls="BioLoader"' in result


def test_linkify_whole_word_only() -> None:
    known = frozenset({'Load'})
    result = _linkify_class_names('Loader', known, 'Other')
    assert 'data-cls' not in result


def test_linkify_escapes_html_in_class_name() -> None:
    known = frozenset({'My<Loader>'})
    result = _linkify_class_names('My&lt;Loader&gt;', known, 'Other')
    # name with html chars — regex won't match word boundary, result unchanged
    assert 'data-cls' not in result or '&lt;' not in result


# --- _inject_graph_links ---


def test_inject_graph_links_adds_data_cls() -> None:
    svg = '<g id="node1" class="node">\n<title>MyLoader</title><text>x</text></g>'
    result = _inject_graph_links(svg, frozenset({'MyLoader'}))
    assert 'data-cls="MyLoader"' in result
    assert 'graph-node-link' in result


def test_inject_graph_links_unknown_class_unchanged() -> None:
    svg = '<g id="node1" class="node">\n<title>UnknownLoader</title><text>x</text></g>'
    result = _inject_graph_links(svg, frozenset({'KnownLoader'}))
    assert 'data-cls' not in result


def test_inject_graph_links_empty_svg() -> None:
    assert _inject_graph_links('', frozenset({'MyLoader'})) == ''


# --- build_json_popup ---


def test_build_json_popup_valid(tmp_path: Path) -> None:
    f = tmp_path / 'data.json'
    f.write_text(json.dumps({'key': 'value'}))
    result = build_json_popup(str(f))
    assert result['title'] == 'data.json'
    assert result['json'] == {'key': 'value'}


def test_build_json_popup_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_json_popup(str(tmp_path / 'nope.json'))


def test_build_json_popup_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / 'bad.json'
    f.write_text('not json')
    with pytest.raises(ValueError, match='Invalid JSON'):
        build_json_popup(str(f))


# --- build_source_popup ---


def test_build_source_popup_from_registry(tmp_path: Path) -> None:
    source_file = tmp_path / 'source.py'
    source_file.write_text('class DummyForPopup(TrackedObject): pass')
    saved = dict(TrackedObject._registry)
    try:
        cls = type('DummyForPopup', (TrackedObject,), {})
        with patch('pygeodata.registry_browser.popups.get_source_code', return_value='class DummyForPopup: pass'):
            result = build_source_popup('DummyForPopup')
        assert 'DummyForPopup' in result['title']
        assert 'diff-table' in result['html']
    finally:
        TrackedObject._registry = saved
        TrackedObject.clear_function_caches()


def test_build_source_popup_from_file(tmp_path: Path) -> None:
    f = tmp_path / 'source.py'
    f.write_text('class OfflineLoader: pass')
    result = build_source_popup('OfflineLoader', source_path=str(f))
    assert 'OfflineLoader' in result['title']
    assert 'OfflineLoader' in result['html']


def test_build_source_popup_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_source_popup('Ghost', source_path=str(tmp_path / 'nope.py'))


def test_build_source_popup_no_class_no_path_raises() -> None:
    with pytest.raises(KeyError):
        build_source_popup('AbsolutelyUnknown')
