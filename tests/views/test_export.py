from pathlib import Path
from types import SimpleNamespace

from pygeodata.catalog.types import FileRef, FileRole
from pygeodata.views.export import _cache_files_for_entry, single_entry_tar_path


def _entry_dir(tmp_path: Path) -> Path:
    entry_dir = tmp_path / 'data_processed' / 'abc'
    entry_dir.mkdir(parents=True)
    for name in ('graph.pdf', 'meta.json', 'parameters.json', 'spec.json', 'process.lock'):
        (entry_dir / name).write_bytes(b'x')
    return entry_dir


def _entry(entry_dir: Path, primary: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        params_path=str(entry_dir / 'parameters.json'),
        class_name='Foo',
        object_type='Data',
        primary_file=FileRef(relpath=primary, kind='raster', role=FileRole.PRIMARY) if primary else None,
        file_path=lambda ref: entry_dir / ref.relpath,
    )


def test_export_includes_zarr_output_and_skips_lock(tmp_path: Path) -> None:
    entry_dir = _entry_dir(tmp_path)
    (entry_dir / 'out.zarr').mkdir()
    (entry_dir / 'out.zarr' / 'zarr.json').write_text('{}')

    files = _cache_files_for_entry('abc', _entry(entry_dir), lambda _: True, [])

    arcnames = {arcname for _, arcname in files}
    assert 'cache/abc/out.zarr' in arcnames
    assert 'cache/abc/meta.json' in arcnames
    assert 'cache/abc/process.lock' not in arcnames


def test_single_entry_download_returns_primary_output(tmp_path: Path) -> None:
    entry_dir = _entry_dir(tmp_path)
    (entry_dir / 'result.tif').write_bytes(b'x')

    data_path, name, needs_tar = single_entry_tar_path('abc', {'abc': _entry(entry_dir, 'result.tif')})

    assert data_path == entry_dir / 'result.tif'
    assert name == 'result.tif'
    assert needs_tar is False


def test_single_entry_download_of_a_missing_output_is_none(tmp_path: Path) -> None:
    entry_dir = _entry_dir(tmp_path)

    assert single_entry_tar_path('abc', {'abc': _entry(entry_dir, 'result.tif')}) == (None, None, False)
