from pathlib import Path

import pytest

from pygeodata.config import set_config
from pygeodata.paths import CachePathConstructor, CachePathResolver, classify_file, is_output_path, is_zarr_store
from pygeodata.spec import SpatialSpec
from tests.fixtures.data import EmptyLoader, LoaderA, LoaderB, LoaderC, LoaderD


def test_exclude_and_private_params() -> None:
    loader = LoaderC(target='deforestation', n_jobs=8, _private_state=True)
    params = loader.get_params()
    assert 'target' in params
    assert 'n_jobs' in params
    assert '_private_state' not in params


def test_deterministic_sorting() -> None:
    loader1 = LoaderB(features=[LoaderA(year=1990), LoaderA(year=2020)])
    loader2 = LoaderB(features=[LoaderA(year=2020), LoaderA(year=1990)])
    assert loader1.get_params() == loader2.get_params()


def test_processed_dir_is_cache_root_slash_state_hash(sample_spatial_spec: SpatialSpec) -> None:
    loader = LoaderA(year=2000)
    expected = loader.get_cache_root() / loader.get_state_hash(sample_spatial_spec)
    assert loader.resolve_cache_paths(sample_spatial_spec).directory == expected


def test_mixed_list_state_hash_no_crash(sample_spatial_spec: SpatialSpec) -> None:
    h = LoaderD(items=[LoaderA(year=2000), 'europe']).get_state_hash(sample_spatial_spec)
    assert isinstance(h, str)
    assert len(h) == 64


def test_parameterless_loader_hash() -> None:
    EmptyLoader.get_dependency_tree_hash.cache_clear()
    h = EmptyLoader().get_instance_hash()
    assert isinstance(h, str)
    assert len(h) == 64


def test_classify_file_html_is_table() -> None:
    assert classify_file(Path('x/preview.html')) == 'table'


def test_classify_file_png_is_image() -> None:
    assert classify_file(Path('x/plot.png')) == 'image'


# --- is_zarr_store ---


def test_is_zarr_store_by_suffix(tmp_path: Path) -> None:
    zarr_dir = tmp_path / 'archive.zarr'
    zarr_dir.mkdir()
    assert is_zarr_store(zarr_dir)


@pytest.mark.parametrize('marker', ['zarr.json', '.zgroup', '.zarray', '.zattrs', '.zmetadata'])
def test_is_zarr_store_by_marker(tmp_path: Path, marker: str) -> None:
    zarr_dir = tmp_path / 'archive'
    zarr_dir.mkdir()
    (zarr_dir / marker).touch()
    assert is_zarr_store(zarr_dir)


def test_is_zarr_store_false(tmp_path: Path) -> None:
    regular_dir = tmp_path / 'regular'
    regular_dir.mkdir()
    (regular_dir / 'data.tif').touch()
    assert not is_zarr_store(regular_dir)


# --- is_output_path / output_paths ---


@pytest.mark.parametrize(
    ('name', 'is_dir', 'expected'),
    [
        ('result.tif', False, True),
        ('data.zarr', True, True),
        ('subdir', True, False),
        ('parameters.json', False, False),
        ('meta.json', False, False),
        ('spec.json', False, False),
        ('graph.pdf', False, False),
        ('process.lock', False, False),
        ('.DS_Store', False, False),
        ('.meta.json.abc.tmp', False, False),
        ('result.tif~', False, False),
        ('Thumbs.db', False, False),
    ],
)
def test_is_output_path(tmp_path: Path, name: str, is_dir: bool, expected: bool) -> None:
    path = tmp_path / name
    if is_dir:
        path.mkdir()
    else:
        path.write_bytes(b'x')
    assert is_output_path(path) is expected


def test_output_paths_sorted(tmp_path: Path) -> None:
    for name in ('b.csv', 'meta.json', 'a.tif'):
        (tmp_path / name).write_bytes(b'x')
    assert [p.name for p in CachePathConstructor(tmp_path).output_paths()] == ['a.tif', 'b.csv']


# --- iter_entry_dirs ---


def test_iter_entry_dirs_direct_children_only(tmp_path: Path) -> None:
    root = tmp_path / 'data_processed'
    (root / 'h1' / 'out.zarr' / 'c').mkdir(parents=True)
    (root / 'h2').mkdir()
    (root / 'stray.txt').write_bytes(b'x')
    assert [p.name for p in CachePathResolver((root,)).iter_entry_dirs()] == ['h1', 'h2']


def test_iter_entry_dirs_skips_nested_root(tmp_path: Path) -> None:
    data = tmp_path / 'data_processed'
    figures = data / 'figures'
    (data / 'h1').mkdir(parents=True)
    (figures / 'h2').mkdir(parents=True)
    with set_config(path_cache=data, path_figures=figures):
        assert sorted(p.name for p in CachePathResolver().iter_entry_dirs()) == ['h1', 'h2']
