from pathlib import Path

import pytest

from pygeodata.config import Config, get_config, set_config
from pygeodata.formatting.path import format_path
from pygeodata.formatting.path_simplified import format_path_simplified


def test_temporary_override(tmp_path: Path) -> None:
    original = get_config().path_cache
    with set_config(path_cache=tmp_path) as cfg:
        assert cfg.path_cache == tmp_path
    assert get_config().path_cache == original


def test_multiple_fields_overridden_simultaneously(tmp_path: Path) -> None:
    with set_config(path_cache=tmp_path, num_threads=8) as cfg:
        assert cfg.path_cache == tmp_path
        assert cfg.num_threads == 8


def test_set_config_restores_all_fields_after_exit(tmp_path: Path) -> None:
    original_cache = get_config().path_cache
    original_threads = get_config().num_threads
    with set_config(path_cache=tmp_path, num_threads=8):
        pass
    assert get_config().path_cache == original_cache
    assert get_config().num_threads == original_threads


def test_set_config_restores_on_exception(tmp_path: Path) -> None:
    original = get_config().path_cache
    with pytest.raises(RuntimeError), set_config(path_cache=tmp_path):
        raise RuntimeError('boom')
    assert get_config().path_cache == original


def test_set_config_nesting() -> None:
    with set_config(num_threads=4):
        assert get_config().num_threads == 4
        with set_config(num_threads=8):
            assert get_config().num_threads == 8
        assert get_config().num_threads == 4
    assert get_config().num_threads == 1


def test_set_config_yields_config_instance() -> None:
    with set_config() as cfg:
        assert isinstance(cfg, Config)


def test_config_invalid_key() -> None:
    with pytest.raises(ValueError), set_config(invalid_key='value'):
        pass


def test_config_update_raises_on_invalid_key() -> None:
    with pytest.raises(ValueError, match='Invalid config key'):
        get_config().update(nonexistent='x')


def test_get_config_returns_singleton() -> None:
    assert get_config() is get_config()


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.path_cache == Path('data_processed')
    assert cfg.path_figures == Path('figures')
    assert cfg.path_registry == Path('.source')
    assert cfg.num_threads == 1
    assert cfg.filesystem_allows_punctuation is True
    assert cfg.human_readable_paths is False
    assert cfg.flatten_figures is False


def test_es_punctuation_on() -> None:
    with set_config(filesystem_allows_punctuation=True):
        assert get_config().es == '='


def test_es_punctuation_off() -> None:
    with set_config(filesystem_allows_punctuation=False):
        assert get_config().es != '='


def test_format_path_fn_punctuation_on() -> None:
    with set_config(filesystem_allows_punctuation=True):
        assert get_config().format_path_fn is format_path


def test_format_path_fn_punctuation_off() -> None:
    with set_config(filesystem_allows_punctuation=False):
        assert get_config().format_path_fn is format_path_simplified
