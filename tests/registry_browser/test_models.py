import pytest

from pygeodata.registry_browser.models import SpecInfo


def test_spec_info_from_spec_json_minimal() -> None:
    info = SpecInfo.from_spec_json({})
    assert info.crs is None
    assert info.resolution is None
    assert info.shape is None
    assert info.bounds is None
    assert info.bounds_latlon is None


def test_spec_info_from_spec_json_crs_empty_string() -> None:
    info = SpecInfo.from_spec_json({'crs': ''})
    assert info.crs is None


def test_spec_info_from_spec_json_crs_set() -> None:
    info = SpecInfo.from_spec_json({'crs': 'EPSG:4326'})
    assert info.crs == 'EPSG:4326'


def test_spec_info_from_spec_json_shape() -> None:
    info = SpecInfo.from_spec_json({'shape': [1800, 3600]})
    assert info.shape == str([1800, 3600])


def test_spec_info_from_spec_json_shape_empty() -> None:
    info = SpecInfo.from_spec_json({'shape': ''})
    assert info.shape is None


def test_spec_info_from_spec_json_bounds_set() -> None:
    info = SpecInfo.from_spec_json({'bounds': [0, 10, 90, 180]})
    assert info.bounds == str([0, 10, 90, 180])


def test_spec_info_from_spec_json_bounds_latlon_epsg4326() -> None:
    info = SpecInfo.from_spec_json({
        'crs': 'EPSG:4326',
        'bounds': [-180.0, -90.0, 180.0, 90.0],
    })
    assert info.bounds_latlon is not None
    lat_min, lon_min, lat_max, lon_max = info.bounds_latlon
    assert lat_min == pytest.approx(-90.0, abs=1)
    assert lon_min == pytest.approx(-180.0, abs=1)
    assert lat_max == pytest.approx(90.0, abs=1)
    assert lon_max == pytest.approx(180.0, abs=1)


def test_spec_info_bounds_latlon_none_when_no_crs() -> None:
    info = SpecInfo.from_spec_json({'bounds': [0, 10, 90, 180]})
    assert info.bounds_latlon is None


def test_spec_info_bounds_latlon_none_when_wrong_length() -> None:
    info = SpecInfo.from_spec_json({'crs': 'EPSG:4326', 'bounds': [0, 10]})
    assert info.bounds_latlon is None


def test_spec_info_bounds_latlon_invalid_crs_does_not_raise() -> None:
    info = SpecInfo.from_spec_json({'crs': 'NOT_A_CRS', 'bounds': [0, 0, 1, 1]})
    assert info.bounds_latlon is None


# --- _format_resolution ---

def test_format_resolution_none() -> None:
    assert SpecInfo._format_resolution(None, None) is None


def test_format_resolution_empty_list() -> None:
    assert SpecInfo._format_resolution([], None) is None


def test_format_resolution_scalar_non_list() -> None:
    result = SpecInfo._format_resolution(0.1, None)
    assert result == '0.1'


def test_format_resolution_equal_values_degrees() -> None:
    result = SpecInfo._format_resolution([0.1, 0.1], 'EPSG:4326')
    assert '°' in result
    assert '0.1' in result


def test_format_resolution_different_values() -> None:
    result = SpecInfo._format_resolution([100.0, 200.0], 'EPSG:32632')
    assert '×' in result


def test_format_resolution_integer_display() -> None:
    result = SpecInfo._format_resolution([1000.0, 1000.0], 'EPSG:32632')
    assert '1000' in result
    assert '.' not in result.split('m')[0].split('°')[0]


def test_format_resolution_single_value() -> None:
    result = SpecInfo._format_resolution([0.5], 'EPSG:4326')
    assert '0.5' in result
    assert '°' in result


def test_format_resolution_unknown_crs_defaults_to_metres() -> None:
    result = SpecInfo._format_resolution([100.0, 100.0], 'INVALID')
    assert 'm' in result
