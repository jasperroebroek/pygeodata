import json
import shutil
from pathlib import Path

from click.testing import CliRunner
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pygeodata.catalog import models, store
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
    assert 'Indexed 2 entry(ies)' in result.output

    engine = store.get_engine(str((tmp_path / '.source').resolve()), format_version=FORMAT_VERSION)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(models.Entry)) == 2
        assert session.scalar(select(func.count()).select_from(models.Crs)) == 1


def test_catalog_build_prunes_missing_rows(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    spec_dict = sample_spatial_spec.to_dict()
    write_entry('s1', 'SimpleTable', spec_dict)
    write_entry('s2', 'OtherTable', spec_dict)

    CliRunner().invoke(cli, ['catalog', 'build'])
    shutil.rmtree(get_config().path_cache / 's2')

    result = CliRunner().invoke(cli, ['catalog', 'build'])
    assert result.exit_code == 0
    assert 'pruned 1 row(s)' in result.output

    engine = store.get_engine(str((tmp_path / '.source').resolve()), format_version=FORMAT_VERSION)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(models.Entry)) == 1


def test_browse_flags_available() -> None:
    result = CliRunner().invoke(cli, ['browse', '--help'])
    assert result.exit_code == 0
    for flag in ('--fast', '--reconcile-interval', '--reimport-on-reconcile'):
        assert flag in result.output
