import pytest
import rasterio as rio
from affine import Affine
from pygeodata.options import RasterCreationOptions
from pygeodata.processors.reprojection import Reprojector
from pygeodata.types import SpatialSpec
from pyproj import CRS
from rasterio import RasterioIOError
from rasterio.enums import Compression
from rasterio.errors import CRSError
from rasterio.warp import calculate_default_transform
from tests.conftest import LUH2_NC, WTD_TIF


def test_reprojection_creates_file(sample_geotiff, sample_spatial_spec, tmp_path):
    output_path = tmp_path / 'output.tif'

    processor = Reprojector(sample_geotiff)
    processor(output_path, sample_spatial_spec)

    # Check that the file was created
    assert output_path.exists()

    # Check that the output has the expected CRS and shape
    with rio.open(output_path) as src:
        assert src.crs.to_epsg() == 4326
        assert src.shape == (sample_spatial_spec.shape[0], sample_spatial_spec.shape[1])


def test_reprojection_with_invalid_crs(sample_geotiff, tmp_path):
    output_path = tmp_path / 'output.tif'

    spec = SpatialSpec(crs='INVALID CRS', transform=Affine.identity(), shape=(100, 100))

    processor = Reprojector(sample_geotiff)
    with pytest.raises(CRSError):
        processor(output_path, spec)


def test_reprojection_with_compression(sample_geotiff, sample_spatial_spec, tmp_path):
    """Test processing with compression options."""
    output_path = tmp_path / 'compressed.tif'

    processor = Reprojector(
        sample_geotiff,
        raster_creation_options=RasterCreationOptions(
            compress='lzw',
            compress_level=6,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        ),
    )

    processor(output_path, sample_spatial_spec)

    with rio.open(output_path) as src:
        assert src.compression == Compression.lzw


def test_reprojection_with_tiff_same_spec(tmp_path):
    output_path = tmp_path / 'output.tif'

    with rio.open(WTD_TIF) as src:
        spec = SpatialSpec(
            crs=src.crs,
            transform=src.transform,
            shape=src.shape,
        )

    processor = Reprojector(WTD_TIF)
    processor(output_path, spec)

    with rio.open(output_path) as src:
        assert src.crs.to_epsg() == 4326
        assert src.shape == (spec.shape[0], spec.shape[1])


def test_reprojection_with_tiff_new_crs(tmp_path):
    output_path = tmp_path / 'output.tif'

    crs = CRS.from_authority('ESRI', '54012')

    with rio.open(WTD_TIF) as src:
        src_crs = src.crs
        width = src.width
        height = src.height

    transform, dst_width, dst_height = calculate_default_transform(
        src_crs,
        crs,
        width,
        height,
        src.bounds.left,
        src.bounds.bottom,
        src.bounds.right,
        src.bounds.top,
    )

    spec = SpatialSpec(
        crs=crs,
        transform=transform,
        shape=(dst_width, dst_height),
    )

    processor = Reprojector(WTD_TIF)
    processor(output_path, spec)

    with rio.open(output_path) as dst:
        assert dst.crs.to_string() == 'ESRI:54012'
        assert dst.shape == (spec.shape[0], spec.shape[1])


def test_reprojection_nc_mutiple_variables(tmp_path):
    output_path = tmp_path / 'output.tif'

    crs = CRS.from_epsg(4326)

    with rio.open(LUH2_NC) as src:
        spec = SpatialSpec(
            crs=crs,
            transform=src.transform,
            shape=src.shape,
        )

    processor = Reprojector(LUH2_NC, src_crs=crs)

    with pytest.raises(RasterioIOError):
        processor(output_path, spec)


def test_reprojection_nc_single_variable(tmp_path):
    output_path = tmp_path / 'output.tif'

    crs = CRS.from_epsg(4326)

    with rio.open(LUH2_NC) as src:
        spec = SpatialSpec(
            crs=crs,
            transform=src.transform,
            shape=src.shape,
        )

    processor = Reprojector(f'netcdf:{LUH2_NC}:primf', src_crs=crs)
    processor(output_path, spec)

    with rio.open(output_path) as dst:
        assert dst.crs.to_epsg() == 4326
        assert dst.shape == (spec.shape[0], spec.shape[1])
        assert dst.count == 86
