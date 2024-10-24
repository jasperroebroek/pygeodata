import os
import shutil
from pathlib import Path

import pytest
from affine import Affine
from rasterio import CRS

from data_framework.data_entry import RasterDataEntry
from data_framework.paths import pc
from data_framework.utils import transform_to_str


@pytest.fixture
def wtd_entry():
    wtd_entry = RasterDataEntry('wtd', path=Path('data/wtd.tif'))
    return wtd_entry


@pytest.fixture
def wtd_entry_fake():
    wtd_entry = RasterDataEntry('wtd', path=Path('data/wtd_fake.tif'))
    return wtd_entry


@pytest.fixture(autouse=True)
def run_around_tests(tmp_path):
    pc.path_data_processed = tmp_path


@pytest.fixture
def crs():
    return CRS.from_epsg(3857)


@pytest.fixture
def transform():
    return Affine.scale(0.1, 0.1)


@pytest.fixture
def shape():
    return (1000, 1000)


def test_raster_data_entry(wtd_entry):
    assert wtd_entry.path == Path('data/wtd.tif')


def test_raster_data_exists(wtd_entry):
    print(os.getcwd())
    assert wtd_entry.path_exists()


def test_data_entry_processed_path(wtd_entry):
    crs, transform, shape = wtd_entry.get_crs_transform_shape_from_file()

    assert (
            wtd_entry.get_processed_path(*wtd_entry.get_crs_transform_shape_from_file()) ==
            Path(
                pc.path_data_processed /
                crs.to_string().replace(":", "_"),
                transform_to_str(transform),
                f"{shape[0]}-{shape[1]}",
                "RasterDataEntry",
                "wtd.tif"
            )
    )


def test_get_reprojected_paths(wtd_entry, crs, transform, shape):
    path = wtd_entry.get_processed_path(crs, transform, shape)
    assert path == Path(
        pc.path_data_processed,
        "EPSG_3857",
        "affine_0.1000_0.0000_0.0000_0.0000_0.1000_0.0000",
        "1000-1000",
        "RasterDataEntry",
        "wtd.tif"
    )
    assert not path.exists()
    assert not path.is_symlink()

    da = wtd_entry.load()
    path = wtd_entry.get_processed_path(*wtd_entry.get_crs_transform_shape_from_file())
    assert da.encoding['source'] == str(path)


def test_generate(wtd_entry_fake, crs, transform, shape):
    with pytest.raises(NotImplementedError):
        wtd_entry_fake.process(crs, transform, shape)


def test_reproject(wtd_entry):
    crs, transform, shape = wtd_entry.parse_crs_transform_shape(CRS.from_authority('ESRI', '54012'))
    wtd_entry.reproject(crs, transform, shape)

    path = wtd_entry.get_processed_path(crs, transform, shape)
    assert not path.is_symlink()
    assert path.exists()

    assert wtd_entry.get_fp().crs.to_string() == "EPSG:4326"
