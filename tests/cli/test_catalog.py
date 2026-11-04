import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner

from pygeodata._version import CATALOG_VERSION
from pygeodata.catalog import store
from pygeodata.catalog.types import CatalogData
from pygeodata.cli import cli
from pygeodata.config import get_config, set_config
from pygeodata.spec import SpatialSpec
from tests.fixtures.records import make_record, write_meta


def write_entry(state_hash: str, class_name: str, spec_dict: dict) -> None:
    entry_dir = get_config().path_cache / state_hash
    write_meta(entry_dir, make_record(state_hash, class_name))
    (entry_dir / 'spec.json').write_text(json.dumps(spec_dict), encoding='utf-8')


def test_catalog_build(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry('s1', 'SimpleTable', spec_dict)
    write_entry('s2', 'OtherTable', spec_dict)

    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0
    assert 'Indexed 2 entries' in result.output

    catalog = CatalogData.from_file(store.catalog_path())
    assert len(catalog.entries) == 2
    assert len(catalog.specs) == 1


def test_catalog_build_prunes_missing_rows(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry('s1', 'SimpleTable', spec_dict)
    write_entry('s2', 'OtherTable', spec_dict)

    CliRunner().invoke(cli, ['catalog', 'build'])
    shutil.rmtree(get_config().path_cache / 's2')

    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0
    assert 'pruned 1 entry' in result.output

    assert len(CatalogData.from_file(store.catalog_path()).entries) == 1


def test_browse_flags_available() -> None:
    result = CliRunner().invoke(cli, ['browse', '--help'])
    assert result.exit_code == 0
    for flag in ('--fast', '--reconcile-interval', '--import-all', '--read-only', '--writable'):
        assert flag in result.output
    assert '--reimport-on-reconcile' not in result.output
    assert '--verbose-import' not in result.output


def _fake_browser(monkeypatch: pytest.MonkeyPatch, open_registry_browser: object) -> None:
    """pygeodata-browser is optional and not installed in CI."""
    serve = ModuleType('pygeodata_browser.serve')
    serve.open_registry_browser = open_registry_browser
    monkeypatch.setitem(sys.modules, 'pygeodata_browser', ModuleType('pygeodata_browser'))
    monkeypatch.setitem(sys.modules, 'pygeodata_browser.serve', serve)


@pytest.mark.parametrize(('flag', 'expected'), [('--read-only', True), ('--writable', False), (None, None)])
def test_browse_passes_the_mode_override(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, expected: bool | None
) -> None:
    opened = {}
    _fake_browser(monkeypatch, lambda *args, **kwargs: opened.update(kwargs))
    result = CliRunner().invoke(cli, ['browse', *([flag] if flag else [])])
    assert result.exit_code == 0, result.output
    assert opened['read_only'] is expected


def test_browse_writable_needs_the_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_browser(monkeypatch, lambda *args, **kwargs: None)
    with set_config(catalog_enabled=False):
        result = CliRunner().invoke(cli, ['browse', '--writable'])
    assert result.exit_code != 0
    assert 'catalog_enabled' in result.output


def test_catalog_build_import_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'project_module.py').write_text('x = 1', encoding='utf-8')

    result = CliRunner().invoke(cli, ['catalog', 'build', '--import-all'])
    assert result.exit_code == 0
    assert 'Imported 1 module(s)' in result.output


def test_catalog_build_warns_on_missing_root(tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0
    assert f'Warning: {tmp_path / "data_processed"} does not exist yet' in result.output


@pytest.mark.skipif(os.name != 'posix' or os.geteuid() == 0, reason='needs non-root POSIX permissions')
def test_catalog_build_errors_on_unreadable_root() -> None:
    root = get_config().path_cache
    root.mkdir(parents=True)
    root.chmod(0)
    try:
        result = CliRunner().invoke(cli, ['catalog', 'build'])
        assert result.exit_code != 0
        assert 'not readable' in result.output
    finally:
        root.chmod(0o755)


def test_entry_list_on_an_outdated_mirror_catalog_is_a_clean_error() -> None:
    catalog = store._new_catalog()
    catalog.header.catalog_version = CATALOG_VERSION - 1
    catalog.header.writer.root = '/eos/elsewhere'
    catalog.dump(get_config().path_catalog)

    result = CliRunner().invoke(cli, ['entry', 'list'])

    assert result.exit_code == 1
    assert 'pygeodata catalog build' in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
