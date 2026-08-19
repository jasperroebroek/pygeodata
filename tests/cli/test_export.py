import json
import tarfile
from pathlib import Path

from click.testing import CliRunner

from pygeodata.cli import cli
from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config, set_config
from pygeodata.paths import CachePathConstructor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_full_entry(state_hash: str, class_name: str, object_type: str = 'Data', ext: str = 'tif') -> Path:
    """Write a complete cache entry (meta/params/spec + output file) that discover_entries() picks up.

    Export goes through ``discover_entries()``, which reads parameters.json
    and spec.json for each record in addition to meta.json. Returns the
    cache directory.
    """
    cache_dir = get_config().path_cache / state_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolver = CachePathConstructor(cache_dir)
    meta = {
        JSONKeys.CLASS_NAME: class_name,
        JSONKeys.OBJECT_TYPE: object_type,
        JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
        JSONKeys.STATE_HASH: state_hash,
    }
    resolver.state_hash_path.write_text(json.dumps(meta), encoding='utf-8')
    resolver.params_path.write_text('{}', encoding='utf-8')
    resolver.spec_path.write_text('{}', encoding='utf-8')
    (cache_dir / f'output.{ext}').write_bytes(b'data')
    return cache_dir


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_all_writes_tar(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive), '--all'])

    assert result.exit_code == 0, result.output
    assert archive.exists()
    with tarfile.open(archive, 'r:*') as tar:
        names = tar.getnames()
    assert 'cache/abc123/output.tif' in names
    assert 'cache/abc123/meta.json' in names


def test_export_by_class(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader')
    write_full_entry('def456', 'OtherLoader')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive), '--class', 'SimpleLoader'])

    assert result.exit_code == 0, result.output
    with tarfile.open(archive, 'r:*') as tar:
        names = tar.getnames()
    assert 'cache/abc123/output.tif' in names
    assert 'cache/def456/output.tif' not in names


def test_export_by_type(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader', object_type='Data')
    write_full_entry('fig111', 'MyFigure', object_type='Figure', ext='png')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive), '--type', 'figure'])

    assert result.exit_code == 0, result.output
    with tarfile.open(archive, 'r:*') as tar:
        names = tar.getnames()
    assert 'cache/fig111/output.png' in names
    assert 'cache/abc123/output.tif' not in names


def test_export_by_hash_prefix(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive), 'abc12'])

    assert result.exit_code == 0, result.output
    with tarfile.open(archive, 'r:*') as tar:
        names = tar.getnames()
    assert 'cache/abc123/output.tif' in names


def test_export_no_selector_errors(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive)])

    assert result.exit_code != 0
    assert 'Provide HASH_PREFIXES' in result.output


def test_export_conflicting_selectors_errors(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive), '--all', '--class', 'SimpleLoader'])

    assert result.exit_code != 0
    assert 'only one of' in result.output


def test_export_unknown_class_errors(tmp_path: Path) -> None:
    write_full_entry('abc123', 'SimpleLoader')
    archive = tmp_path / 'out.tar'

    runner = CliRunner()
    result = runner.invoke(cli, ['export', str(archive), '--class', 'NoSuchClass'])

    assert result.exit_code != 0
    assert 'No entries found' in result.output


# ---------------------------------------------------------------------------
# export -> import round trip
# ---------------------------------------------------------------------------


def test_export_import_round_trip(tmp_path: Path) -> None:
    """The archive export writes must be exactly what import can read back.

    Regression guard for the format mismatch where export wrote a plain
    (uncompressed) tar but import only accepted 'r:gz' — an export could
    never be re-imported. Exercises both commands for real, no mocking.
    """
    write_full_entry('abc123', 'SimpleLoader')
    archive = tmp_path / 'roundtrip.tar'

    runner = CliRunner()
    export_result = runner.invoke(cli, ['export', str(archive), '--all'])
    assert export_result.exit_code == 0, export_result.output
    assert archive.exists()

    new_root = tmp_path / 'new_project'
    with set_config(
        path_cache=new_root / 'data_processed',
        path_tables=new_root / 'tables',
        path_figures=new_root / 'figures',
        path_registry=new_root / '.source',
    ):
        import_result = runner.invoke(cli, ['import', str(archive)])
        assert import_result.exit_code == 0, import_result.output
        assert '4 cache files' in import_result.output

        imported_file = new_root / 'data_processed' / 'abc123' / 'output.tif'
        assert imported_file.exists()
        assert imported_file.read_bytes() == b'data'
