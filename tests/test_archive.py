from pathlib import Path

from pygeodata.archive import collect_export_files, entry_download
from pygeodata.config import get_config
from pygeodata.registries.registry import EntryRegistry
from tests.fixtures.records import make_record, write_meta


def write_entry(*outputs: str) -> Path:
    entry_dir = get_config().path_cache / 'abc'
    write_meta(entry_dir, make_record('abc', 'Foo', 'Data', ext='tif'))
    for name in ('graph.pdf', 'parameters.json', 'spec.json', 'process.lock'):
        (entry_dir / name).write_bytes(b'x')
    for name in outputs:
        (entry_dir / name).write_bytes(b'x')
    return entry_dir


def test_export_includes_zarr_output_and_skips_lock() -> None:
    entry_dir = write_entry()
    (entry_dir / 'out.zarr').mkdir()
    (entry_dir / 'out.zarr' / 'zarr.json').write_text('{}')

    files, warnings = collect_export_files(['abc'], EntryRegistry(), include_snapshots=False)

    arcnames = {arcname for _, arcname in files}
    assert 'cache/abc/out.zarr' in arcnames
    assert 'cache/abc/meta.json' in arcnames
    assert 'cache/abc/process.lock' not in arcnames
    assert warnings == []


def test_export_skips_disallowed_entry_with_a_warning() -> None:
    write_entry('result.tif')

    files, warnings = collect_export_files(
        ['abc'],
        EntryRegistry(),
        include_snapshots=False,
        path_allowed=lambda _: False,
    )

    assert files == []
    assert len(warnings) == 1


def test_entry_download_is_the_primary_output() -> None:
    entry_dir = write_entry('result.csv', 'result.tif')

    download = entry_download('abc', EntryRegistry())

    assert download.path == entry_dir.absolute() / 'result.tif'
    assert download.name == 'result.tif'
    assert download.needs_tar is False


def test_entry_download_of_a_missing_output_is_none() -> None:
    entry_dir = write_entry('result.tif')
    registry = EntryRegistry()
    (entry_dir / 'result.tif').unlink()

    assert entry_download('abc', registry) is None
