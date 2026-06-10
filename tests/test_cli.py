import io
import json
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from pygeodata.cli import cli
from pygeodata.config import JSONKeys, get_config
from tests.fixtures.data import SimpleLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tar(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """Write an in-memory tar.gz to tmp_path and return its path."""
    archive = tmp_path / 'export.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return archive


def _hash_json(class_name: str = 'SimpleLoader', object_type: str = 'Data') -> bytes:
    return json.dumps({JSONKeys.CLASS_NAME: class_name, JSONKeys.OBJECT_TYPE: object_type}).encode()


def _invoke(archive: Path) -> str:
    runner = CliRunner()
    result = runner.invoke(cli, ['import', str(archive)])
    return result.output


# ---------------------------------------------------------------------------
# Cache import
# ---------------------------------------------------------------------------


def test_import_cache_file_lands_in_data_cache(tmp_path: Path) -> None:
    hash_dir = 'abc123'
    archive = _make_tar(tmp_path, {
        f'cache/{hash_dir}/.simple_loader.hash.json': _hash_json(),
        f'cache/{hash_dir}/simple_loader.tif': b'raster',
    })

    _invoke(archive)

    dest_dir = get_config().path_cache / hash_dir
    assert (dest_dir / 'simple_loader.tif').exists()


def test_import_cache_hash_json_also_extracted(tmp_path: Path) -> None:
    hash_dir = 'abc123'
    archive = _make_tar(tmp_path, {
        f'cache/{hash_dir}/.simple_loader.hash.json': _hash_json(),
        f'cache/{hash_dir}/simple_loader.tif': b'raster',
    })

    _invoke(archive)

    dest_dir = get_config().path_cache / hash_dir
    assert (dest_dir / '.simple_loader.hash.json').exists()


def test_import_cache_unknown_object_type_skipped(tmp_path: Path) -> None:
    hash_dir = 'abc123'
    archive = _make_tar(tmp_path, {
        f'cache/{hash_dir}/.unknown.hash.json': _hash_json('SimpleLoader', object_type='UnknownType'),
        f'cache/{hash_dir}/data.tif': b'raster',
    })

    _invoke(archive)

    assert not (get_config().path_cache / hash_dir).exists()


def test_import_cache_existing_entry_not_overwritten(tmp_path: Path) -> None:
    hash_dir = 'abc123'
    archive = _make_tar(tmp_path, {
        f'cache/{hash_dir}/.simple_loader.hash.json': _hash_json(),
        f'cache/{hash_dir}/simple_loader.tif': b'new content',
    })

    existing = get_config().path_cache / hash_dir / 'simple_loader.tif'
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b'original')

    _invoke(archive)

    assert existing.read_bytes() == b'original'


# ---------------------------------------------------------------------------
# Registry import (code / snapshots)
# ---------------------------------------------------------------------------


def test_import_code_entry_lands_in_registry(tmp_path: Path) -> None:
    src_hash = 'deadbeef'
    archive = _make_tar(tmp_path, {
        f'code/{src_hash}/source.py': b'class Foo: pass',
        f'code/{src_hash}/source.json': b'{}',
    })

    _invoke(archive)

    code_dir = get_config().path_registry / 'code' / src_hash
    assert (code_dir / 'source.py').exists()
    assert (code_dir / 'source.json').exists()


def test_import_snapshot_entry_lands_in_registry(tmp_path: Path) -> None:
    dep_hash = 'cafebabe'
    archive = _make_tar(tmp_path, {
        f'snapshots/{dep_hash}/tree.json': b'{}',
    })

    _invoke(archive)

    snap_dir = get_config().path_registry / 'snapshots' / dep_hash
    assert (snap_dir / 'tree.json').exists()


def test_import_registry_existing_entry_not_overwritten(tmp_path: Path) -> None:
    src_hash = 'deadbeef'
    archive = _make_tar(tmp_path, {
        f'code/{src_hash}/source.py': b'class Bar: pass',
    })

    existing = get_config().path_registry / 'code' / src_hash / 'source.py'
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b'original')

    _invoke(archive)

    assert existing.read_bytes() == b'original'


# ---------------------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------------------


def test_import_prints_counts(tmp_path: Path) -> None:
    hash_dir = 'abc123'
    archive = _make_tar(tmp_path, {
        f'cache/{hash_dir}/.simple_loader.hash.json': _hash_json(),
        f'cache/{hash_dir}/simple_loader.tif': b'raster',
        f'code/deadbeef/source.py': b'class Foo: pass',
        f'snapshots/cafebabe/tree.json': b'{}',
    })

    output = _invoke(archive)

    assert 'cache files' in output
    assert 'code snapshot files' in output
    assert 'tree snapshot files' in output


def test_import_ignores_malformed_paths(tmp_path: Path) -> None:
    archive = _make_tar(tmp_path, {
        'cache/onlyone': b'data',
        'other/stuff/file.txt': b'ignored',
    })

    output = _invoke(archive)
    assert 'Imported' in output
