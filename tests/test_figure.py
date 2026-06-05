from collections.abc import Generator
from pathlib import Path

import pytest

from pygeodata.config import get_config, set_config
from pygeodata.extraction import flatten_parameter_dict_for_path
from pygeodata.figure import Figure
from pygeodata.paths import generate_path
from pygeodata.types import SpatialSpec
from tests.fixtures.figures import DummyFigure, NoParamsFigure, SimpleFigure, TwoParamFigure


@pytest.fixture(
    params=[
        (True, False, False),
        (False, False, False),
        (True, True, False),
        (False, True, False),
        (True, False, True),
        (False, False, True),
        (True, True, True),
        (False, True, True),
    ],
    autouse=True,
    ids=[
        'punct-on,human-off,flat-off',
        'punct-off,human-off,flat-off',
        'punct-on,human-on,flat-off',
        'punct-off,human-on,flat-off',
        'punct-on,human-off,flat-on',
        'punct-off,human-off,flat-on',
        'punct-on,human-on,flat-on',
        'punct-off,human-on,flat-on',
    ],
)
def all_figure_layouts(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    punct, human, flat = request.param
    with set_config(filesystem_allows_punctuation=punct, human_readable_paths=human, flatten_figures=flat):
        yield


# --- get_processed_dir ---

def test_figure_get_processed_dir_flatten(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    with set_config(flatten_figures=True):
        assert Figure().get_processed_dir(sample_spatial_spec) == tmp_path / 'figures'


def test_figure_get_processed_dir_human_readable(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    with set_config(flatten_figures=False, human_readable_paths=True):
        inner_shape = '1800x3600'
        inner_transform = '0p1_0_-180_0_-0p1_90'
        geo_str = (
            f'S[{inner_shape}]_A[{inner_transform}]'
            if get_config().filesystem_allows_punctuation
            else f'S--{inner_shape}--_A--{inner_transform}--'
        )
        expected = tmp_path / 'figures' / 'EPSG_4326' / geo_str / 'Figure'
        assert Figure().get_processed_dir(sample_spatial_spec) == expected


def test_figure_get_processed_dir_hash_based(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    with set_config(flatten_figures=False, human_readable_paths=False):
        expected = generate_path(spec=sample_spatial_spec, name='Figure', base_dir=tmp_path / 'figures')
        assert Figure().get_processed_dir(sample_spatial_spec) == expected


# --- get_filename ---

def test_figure_get_filename_no_params() -> None:
    fig = NoParamsFigure()
    with set_config(flatten_figures=False):
        assert fig.get_filename(ext='png') == f'{fig.get_file_stem()}.png'


def test_figure_get_filename_no_flatten_uses_plain_stem() -> None:
    fig = SimpleFigure(a=5)
    with set_config(flatten_figures=False):
        assert fig.get_filename(ext='png') == f'{fig.get_file_stem()}.png'


def test_figure_get_filename_flatten_human_readable_single_param() -> None:
    fig = SimpleFigure(a=5)
    with set_config(flatten_figures=True, human_readable_paths=True):
        stem = fig.get_file_stem()
        sep = '_' if get_config().filesystem_allows_punctuation else ' '
        es = '=' if get_config().filesystem_allows_punctuation else '-'
        assert fig.get_filename(ext='png') == f'{stem}{sep}a{es}5.png'


def test_figure_get_filename_flatten_human_readable_multiple_params() -> None:
    fig = TwoParamFigure(a=3, b='x')
    with set_config(flatten_figures=True, human_readable_paths=True):
        stem = fig.get_file_stem()
        sep = '_' if get_config().filesystem_allows_punctuation else ' '
        es = '=' if get_config().filesystem_allows_punctuation else '-'
        params = flatten_parameter_dict_for_path(fig.get_params(exclude=True))
        joined = sep.join(f'{k}{es}{v}' for k, v in params.items())
        assert fig.get_filename(ext='png') == f'{stem}{sep}{joined}.png'


def test_figure_get_filename_flatten_hash_based(sample_spatial_spec: SpatialSpec) -> None:
    fig = SimpleFigure(a=5)
    with set_config(flatten_figures=True, human_readable_paths=False):
        assert fig.get_filename(ext='png', spec=sample_spatial_spec) == f'{fig.get_state_hash(sample_spatial_spec)}.png'


# --- process / roundtrip ---

def test_figure_process_creates_file_and_hash_and_params(sample_spatial_spec: SpatialSpec) -> None:
    fig = DummyFigure(a=1)
    fig.process(sample_spatial_spec)

    assert fig.get_processed_path(sample_spatial_spec).exists()
    assert fig.get_state_hash_path(sample_spatial_spec).exists()
    assert fig.resolve_cache_paths(sample_spatial_spec).params_path.exists()


def test_figure_is_processed_roundtrip(sample_spatial_spec: SpatialSpec) -> None:
    fig = DummyFigure(a=2)
    fig.process(sample_spatial_spec)
    assert fig.is_processed(sample_spatial_spec)

    fig.get_state_hash_path(sample_spatial_spec).unlink()
    assert not fig.is_processed(sample_spatial_spec)
