import pytest

from pygeodata.registry_browser.models import SpecInfo
from pygeodata.spec import SpatialSpec, format_resolution


def _spec(crs=None, bounds=None, resolution=None, shape=None) -> SpecInfo:
    """Build a SpecInfo via the full SpatialSpec → from_spec path."""
    from affine import Affine
    from pyproj import CRS

    if crs is None and bounds is None and resolution is None:
        return SpecInfo()

    crs_obj = CRS.from_user_input(crs) if crs else CRS.from_epsg(4326)

    if resolution is not None and bounds is not None:
        # Build a transform from bounds + resolution
        res_x, res_y = (resolution, resolution) if isinstance(resolution, (int, float)) else resolution
        xmin, ymin, xmax, ymax = bounds
        transform = Affine(res_x, 0, xmin, 0, -res_y, ymax)
        rows = int((ymax - ymin) / res_y)
        cols = int((xmax - xmin) / res_x)
        return SpecInfo.from_spec(SpatialSpec(crs=crs_obj, transform=transform, shape=(rows, cols)))

    # Just CRS, no transform
    return SpecInfo.from_spec(SpatialSpec(crs=crs_obj))


def test_spec_info_from_spec_minimal() -> None:
    info = SpecInfo()
    assert info.crs is None
    assert info.resolution is None
    assert info.shape is None
    assert info.bounds is None
    assert info.bounds_latlon is None


def test_spec_info_from_spec_crs_set() -> None:
    info = _spec(crs='EPSG:4326')
    assert info.crs == 'EPSG:4326'


def test_spec_info_from_spec_shape() -> None:
    info = _spec(crs='EPSG:4326', bounds=[-180, -90, 180, 90], resolution=0.1)
    assert info.shape is not None


def test_spec_info_from_spec_bounds_set() -> None:
    info = _spec(crs='EPSG:4326', bounds=[-180, -90, 180, 90], resolution=0.1)
    assert info.bounds is not None


def test_spec_info_bounds_latlon_epsg4326() -> None:
    info = _spec(crs='EPSG:4326', bounds=[-180, -90, 180, 90], resolution=1.0)
    assert info.bounds_latlon is not None
    lat_min, lon_min, lat_max, lon_max = info.bounds_latlon
    assert lat_min == pytest.approx(-90.0, abs=1)
    assert lon_min == pytest.approx(-180.0, abs=1)
    assert lat_max == pytest.approx(90.0, abs=1)
    assert lon_max == pytest.approx(180.0, abs=1)


def test_spec_info_bounds_latlon_none_when_no_transform_no_area() -> None:
    # A CRS with no area_of_use and no transform → bounds raises ValueError → bounds_latlon is None
    from pyproj import CRS
    from pygeodata.spec import SpatialSpec
    # Build a custom CRS with no area_of_use (e.g. a local engineering CRS)
    # Fall back: just test the ValueError path in from_spec directly
    class _BadSpec:
        crs = CRS.from_epsg(4326)
        @property
        def bounds(self):
            raise ValueError('no bounds')
        @property
        def resolution(self):
            raise ValueError('no resolution')
        shape = None
    info = SpecInfo.from_spec(_BadSpec())  # type: ignore[arg-type]
    assert info.bounds_latlon is None


def test_spec_info_bounds_latlon_invalid_crs_does_not_raise() -> None:
    # CRS.from_user_input('NOT_A_CRS') raises at SpatialSpec.from_dict level;
    # here we verify compute_bounds_latlon handles ProjError gracefully.
    from pygeodata.spec import compute_bounds_latlon
    result = compute_bounds_latlon([0, 0, 1, 1], 'NOT_A_CRS')
    assert result is None


# --- format_resolution ---

def test_format_resolution_none() -> None:
    assert format_resolution(None, None) is None


def test_format_resolution_empty_list() -> None:
    assert format_resolution([], None) is None


def test_format_resolution_scalar_non_list() -> None:
    result = format_resolution(0.1, None)
    assert result == '0.1'


def test_format_resolution_equal_values_degrees() -> None:
    result = format_resolution([0.1, 0.1], 'EPSG:4326')
    assert '°' in result
    assert '0.1' in result


def test_format_resolution_different_values() -> None:
    result = format_resolution([100.0, 200.0], 'EPSG:32632')
    assert '×' in result


def test_format_resolution_integer_display() -> None:
    result = format_resolution([1000.0, 1000.0], 'EPSG:32632')
    assert '1000' in result
    assert '.' not in result.split('m')[0].split('°')[0]


def test_format_resolution_single_value() -> None:
    result = format_resolution([0.5], 'EPSG:4326')
    assert '0.5' in result
    assert '°' in result


def test_format_resolution_unknown_crs_defaults_to_metres() -> None:
    result = format_resolution([100.0, 100.0], 'INVALID')
    assert 'm' in result


def test_format_resolution_accepts_crs_object() -> None:
    from pyproj import CRS
    crs = CRS.from_epsg(4326)
    result = format_resolution([0.1, 0.1], crs)
    assert '°' in result
