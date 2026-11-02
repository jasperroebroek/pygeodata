import sys
from dataclasses import fields
from pathlib import Path

import pytest

from pygeodata.config import Config, get_config, load_project_settings, set_config


def _pop_settings_modules() -> None:
    for name in [n for n, m in list(sys.modules.items()) if hasattr(m, '_config_keys')]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _restore_config() -> None:
    """Snapshot the global config and the settings modules around each test."""
    original = {f.name: getattr(get_config(), f.name) for f in fields(get_config())}
    _pop_settings_modules()
    yield
    get_config().update(**original)
    _pop_settings_modules()


def test_temporary_override(tmp_path: Path) -> None:
    original = get_config().path_cache
    with set_config(path_cache=tmp_path) as cfg:
        assert cfg.path_cache == tmp_path
    assert get_config().path_cache == original


def test_multiple_fields_overridden_simultaneously(tmp_path: Path) -> None:
    with set_config(path_cache=tmp_path, num_threads=8) as cfg:
        assert cfg.path_cache == tmp_path
        assert cfg.num_threads == 8


def test_cache_roots_contains_all_three_roots() -> None:
    cfg = Config()
    assert set(cfg.cache_roots) == {cfg.path_cache, cfg.path_tables, cfg.path_figures}


def test_cache_roots_deduplicates_coincident_paths(tmp_path: Path) -> None:
    with set_config(path_tables=tmp_path, path_figures=tmp_path) as cfg:
        assert cfg.cache_roots.count(tmp_path) == 1


def test_load_project_settings_absent_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_project_settings() == []


def test_load_project_settings_applies_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\n'
        f'get_config().update(path_cache={str(tmp_path / "cache")!r}, num_threads=4)\n',
        encoding='utf-8',
    )

    assert load_project_settings() == [tmp_path / 'pygeodata_settings.py']
    assert get_config().path_cache == tmp_path / 'cache'
    assert get_config().num_threads == 4


def test_load_project_settings_raises_on_broken_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'pygeodata_settings.py').write_text('raise RuntimeError("broken settings")\n', encoding='utf-8')

    with pytest.raises(RuntimeError, match='broken settings'):
        load_project_settings()


def test_load_project_settings_from_src_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=5)\n',
        encoding='utf-8',
    )

    monkeypatch.chdir(tmp_path)
    assert load_project_settings() == [tmp_path / 'src' / 'pygeodata_settings.py']
    assert get_config().num_threads == 5


def test_load_project_settings_loads_root_and_src(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=5)\n',
        encoding='utf-8',
    )
    (tmp_path / 'pygeodata_settings.py').write_text(
        f'from pygeodata.config import get_config\nget_config().update(path_cache={str(tmp_path / "cache")!r})\n',
        encoding='utf-8',
    )

    monkeypatch.chdir(tmp_path)
    assert load_project_settings() == [
        tmp_path / 'pygeodata_settings.py',
        tmp_path / 'src' / 'pygeodata_settings.py',
    ]
    assert get_config().path_cache == tmp_path / 'cache'
    assert get_config().num_threads == 5


def test_load_project_settings_does_not_search_parents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'pygeodata_settings.py').write_text('VALUE = 1\n', encoding='utf-8')
    nested = tmp_path / 'a' / 'b'
    nested.mkdir(parents=True)

    monkeypatch.chdir(nested)
    assert load_project_settings() == []


def test_load_project_settings_explicit_path_loads_alongside_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / 'confs').mkdir()
    explicit = tmp_path / 'confs' / 'my_settings.py'
    explicit.write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=5)\n',
        encoding='utf-8',
    )
    (tmp_path / 'pygeodata_settings.py').write_text(
        f'from pygeodata.config import get_config\nget_config().update(path_cache={str(tmp_path / "cache")!r})\n',
        encoding='utf-8',
    )

    monkeypatch.chdir(tmp_path)
    assert load_project_settings(settings_path=explicit) == [
        explicit,
        tmp_path / 'pygeodata_settings.py',
    ]
    assert get_config().num_threads == 5
    assert get_config().path_cache == tmp_path / 'cache'


def test_load_project_settings_errors_on_overlapping_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=4)\n',
        encoding='utf-8',
    )
    (tmp_path / 'src' / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=5)\n',
        encoding='utf-8',
    )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match='overlapping config keys: num_threads'):
        load_project_settings()


def test_load_project_settings_imported_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'pygeodata_settings.py').write_text('VALUE = 1\n', encoding='utf-8')

    assert load_project_settings() == [tmp_path / 'pygeodata_settings.py']
    sys.modules['pygeodata_settings'].VALUE = 2
    assert load_project_settings() == [tmp_path / 'pygeodata_settings.py']
    assert sys.modules['pygeodata_settings'].VALUE == 2


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
    assert cfg.path_tables == Path('tables')
    assert cfg.path_figures == Path('figures')
    assert cfg.path_registry == Path('.source')
    assert cfg.num_threads == 1
