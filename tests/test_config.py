from pathlib import Path

import pytest

from pygeodata.config import Config, get_config, set_config


def test_default_config():
    cfg = get_config()
    assert cfg.path_data_processed == Path('data_processed')


def test_temporary_override():
    original = get_config().path_data_processed
    with set_config(path_data_processed=Path('/tmp')) as cfg:
        assert cfg.path_data_processed == Path('/tmp')
    assert get_config().path_data_processed == original


def test_multiple_overrides():
    with set_config(path_data_processed=Path('/tmp')) as cfg:
        assert isinstance(cfg, Config)
        assert cfg.path_data_processed == Path('/tmp')


def test_config_invalid_key():
    with pytest.raises(ValueError):
        with set_config(invalid_key='value'):
            pass
