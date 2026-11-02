import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from pygeodata.cli import cli
from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config

SETTINGS_MODULE = 'pygeodata_settings'


@pytest.fixture(autouse=True)
def _pop_settings_module() -> None:
    sys.modules.pop(SETTINGS_MODULE, None)
    yield
    sys.modules.pop(SETTINGS_MODULE, None)


def _write_entry(entry_dir: Path, state_hash: str) -> None:
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / 'meta.json').write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: 'SettingsLoader',
                JSONKeys.OBJECT_TYPE: 'Table',
                JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
                JSONKeys.STATE_HASH: state_hash,
            },
        ),
        encoding='utf-8',
    )
    (entry_dir / 'spec.json').write_text(json.dumps({'crs': 'EPSG:4326'}), encoding='utf-8')


def test_cli_applies_pygeodata_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_root = tmp_path / 'custom_cache'
    catalog_db = tmp_path / 'custom_catalog.db'
    (tmp_path / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\n'
        f'get_config().update(path_cache={str(cache_root)!r}, path_catalog={str(catalog_db)!r})\n',
        encoding='utf-8',
    )
    _write_entry(cache_root / 'e1', 'e1')

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0, result.output
    assert f'Loaded settings: {tmp_path / "pygeodata_settings.py"}' in result.output
    assert 'Indexed 1 entry' in result.output
    assert catalog_db.exists()


def test_cli_settings_flag_imports_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'confs').mkdir()
    settings = tmp_path / 'confs' / 'my_settings.py'
    settings.write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=5)\n',
        encoding='utf-8',
    )

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['--settings', str(settings), 'clean-locks', '--dry-run'])
    assert result.exit_code == 0, result.output
    assert f'Loaded settings: {settings}' in result.output
    assert get_config().num_threads == 5


def test_cli_applies_settings_from_src_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'pygeodata_settings.py').write_text(
        'from pygeodata.config import get_config\nget_config().update(num_threads=5)\n',
        encoding='utf-8',
    )

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['clean-locks', '--dry-run'])
    assert result.exit_code == 0, result.output
    assert f'Loaded settings: {tmp_path / "src" / "pygeodata_settings.py"}' in result.output
    assert get_config().num_threads == 5


def test_cli_fails_on_overlapping_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    result = CliRunner().invoke(cli, ['clean-locks', '--dry-run'])
    assert result.exit_code != 0
    assert 'overlapping config keys: num_threads' in result.output


def test_cli_fails_cleanly_on_broken_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'pygeodata_settings.py').write_text('raise RuntimeError("broken settings")\n', encoding='utf-8')

    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code != 0
    assert 'settings module failed: broken settings' in result.output
