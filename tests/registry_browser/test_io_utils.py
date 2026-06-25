import json
from pathlib import Path

import pytest

from pygeodata.catalog.io_utils import existing_path_str, read_json_dict, read_text


def test_existing_path_str_none() -> None:
    assert existing_path_str(None) is None


def test_existing_path_str_exists(tmp_path: Path) -> None:
    f = tmp_path / 'file.txt'
    f.touch()
    result = existing_path_str(f)
    assert result is not None
    assert Path(result).exists()


def test_existing_path_str_missing(tmp_path: Path) -> None:
    assert existing_path_str(tmp_path / 'nonexistent.txt') is None


def test_existing_path_str_returns_string(tmp_path: Path) -> None:
    f = tmp_path / 'file.txt'
    f.touch()
    assert isinstance(existing_path_str(f), str)


def test_read_json_dict_none() -> None:
    assert read_json_dict(None) == {}


def test_read_json_dict_missing_file(tmp_path: Path) -> None:
    assert read_json_dict(tmp_path / 'missing.json') == {}


def test_read_json_dict_valid(tmp_path: Path) -> None:
    f = tmp_path / 'data.json'
    f.write_text(json.dumps({'key': 'value'}))
    assert read_json_dict(f) == {'key': 'value'}


def test_read_json_dict_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / 'bad.json'
    f.write_text('not json {{{')
    assert read_json_dict(f) == {}


def test_read_json_dict_non_object_json(tmp_path: Path) -> None:
    f = tmp_path / 'list.json'
    f.write_text('[1, 2, 3]')
    assert read_json_dict(f) == {}


def test_read_text_none() -> None:
    assert read_text(None) is None


def test_read_text_missing(tmp_path: Path) -> None:
    assert read_text(tmp_path / 'nope.txt') is None


def test_read_text_valid(tmp_path: Path) -> None:
    f = tmp_path / 'hello.txt'
    f.write_text('hello world')
    assert read_text(f) == 'hello world'
