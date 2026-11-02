import json
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from pygeodata.catalog import store
from pygeodata.cli import cli
from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config
from pygeodata.spec import SpatialSpec


def write_entry(state_hash: str, class_name: str, spec_dict: dict) -> None:
    entry_dir = get_config().path_cache / state_hash
    entry_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        JSONKeys.CLASS_NAME: class_name,
        JSONKeys.OBJECT_TYPE: 'Table',
        JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
        JSONKeys.STATE_HASH: state_hash,
        JSONKeys.SPEC_HASH: 'spec1',
    }
    (entry_dir / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')
    (entry_dir / 'spec.json').write_text(json.dumps(spec_dict), encoding='utf-8')


def test_catalog_build(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry('s1', 'SimpleTable', spec_dict)
    write_entry('s2', 'OtherTable', spec_dict)

    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0
    assert 'Indexed 2 entries' in result.output

    catalog = store.load_catalog()
    assert len(catalog['entries']) == 2
    assert len(catalog['dimensions']['crs']) == 1


def test_catalog_build_prunes_missing_rows(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry('s1', 'SimpleTable', spec_dict)
    write_entry('s2', 'OtherTable', spec_dict)

    CliRunner().invoke(cli, ['catalog', 'build'])
    shutil.rmtree(get_config().path_cache / 's2')

    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0
    assert 'pruned 1 entry' in result.output

    assert len(store.load_catalog()['entries']) == 1


def test_browse_flags_available() -> None:
    result = CliRunner().invoke(cli, ['browse', '--help'])
    assert result.exit_code == 0
    for flag in ('--fast', '--reconcile-interval', '--reimport-on-reconcile'):
        assert flag in result.output


def test_catalog_build_import_all(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'project_module.py').write_text('x = 1', encoding='utf-8')

    result = CliRunner().invoke(cli, ['catalog', 'build', '--import-all'])
    assert result.exit_code == 0
    assert 'Imported 1 module(s)' in result.output


def test_catalog_build_log_interval_flag(sample_spatial_spec: SpatialSpec) -> None:
    write_entry('s1', 'SimpleTable', sample_spatial_spec.to_dict())
    write_entry('s2', 'OtherTable', sample_spatial_spec.to_dict())

    result = CliRunner().invoke(cli, ['catalog', 'build', '--log-interval', '1'])
    assert result.exit_code == 0
    assert 'Indexed 2 entries' in result.output

    assert len(store.load_catalog()['entries']) == 2


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
