import numpy as np
import rasterio as rio
from numpy import dtype

from pygeodata.processors.rasterizer import Rasterizer
from tests.conftest import COUNTRIES_SHP


def test_rasterizer_float(tmp_path, sample_spatial_spec):
    output_path = tmp_path / 'output.tif'

    Rasterizer(COUNTRIES_SHP, dtype=np.float64)(output_path, sample_spatial_spec)

    with rio.open(output_path) as src:
        assert src.crs.to_epsg() == 4326
        assert src.shape == (sample_spatial_spec.shape[0], sample_spatial_spec.shape[1])
        assert src.count == 1
        assert dtype(src.dtypes[0]) == np.float64
        assert np.isnan(src.read(1)[0, 0])


def test_rasterizer_int(tmp_path, sample_spatial_spec):
    output_path = tmp_path / 'output.tif'

    Rasterizer(COUNTRIES_SHP, dtype=np.int32, fill_value=-1)(output_path, sample_spatial_spec)

    with rio.open(output_path) as src:
        assert src.crs.to_epsg() == 4326
        assert src.shape == (sample_spatial_spec.shape[0], sample_spatial_spec.shape[1])
        assert src.count == 1
        assert dtype(src.dtypes[0]) == np.int32
        assert src.read(1)[0, 0] == -1
