import json
from dataclasses import asdict

import pytest

from pygeodata import load
from pygeodata.config import set_config
from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec


def test_loader_initialization(sample_loader_class):
    loader = sample_loader_class()
    assert loader.get_name() == 'sample_loader'
    assert hasattr(loader, 'processor')
    assert hasattr(loader, 'driver')
    assert loader.get_class_name() == 'SampleLoader'
    assert loader.get_params() == {}


def test_loader_no_processor_raises(mock_spec, tmp_path):
    class Naked(DataLoader):
        pass

    with set_config(path_data_processed=tmp_path), pytest.raises(NotImplementedError):
        Naked().process(mock_spec)


def test_name_conversion_simple(sample_loader_class):
    assert sample_loader_class().get_name() == 'sample_loader'


def test_name_conversion_camel_case():
    class MyDataLoader(DataLoader):
        processor = lambda self: None

    assert MyDataLoader().get_name() == 'my_data_loader'


def test_name_conversion_acronyms():
    class XMLHTTPLoader(DataLoader):
        processor = lambda self: None

    assert XMLHTTPLoader.get_name() == 'xmlhttp_loader'


def test_name_conversion_mixed():
    class USGSElevationLoader(DataLoader):
        processor = lambda self: None

    assert USGSElevationLoader.get_name() == 'usgs_elevation_loader'


def test_get_params_excludes_private(sample_loader_class_complex):
    loader = sample_loader_class_complex(time=10, resolution=5)
    loader._private_attr = 'should_not_appear'
    assert '_private_attr' not in loader.get_params()


def test_get_params_multiple(sample_loader_class_complex):
    loader = sample_loader_class_complex(time=15, resolution=20)
    params = loader.get_params()
    assert params['time'] == 15
    assert params['resolution'] == 20
    assert len(params) == 2


def test_loader_repr(sample_loader_class):
    r = repr(sample_loader_class())
    assert r.startswith('SampleLoader(')
    assert r.endswith(')')


def test_loader_repr_sorted_params(sample_loader_class_complex):
    r = repr(sample_loader_class_complex(time=10, resolution=5))
    assert r.index('resolution') < r.index('time')


def test_same_spec_same_path(sample_loader_class, sample_spatial_spec, tmp_path):
    with set_config(path_data_processed=tmp_path):
        p1 = sample_loader_class().get_processed_path(sample_spatial_spec)
        p2 = sample_loader_class().get_processed_path(sample_spatial_spec)
    assert p1 == p2


def test_different_specs_different_paths(sample_loader_class, sample_spatial_spec, tmp_path):
    d = asdict(sample_spatial_spec)
    d['shape'] = (2, 3)
    spec2 = SpatialSpec(**d)
    with set_config(path_data_processed=tmp_path):
        p1 = sample_loader_class().get_processed_path(sample_spatial_spec)
        p2 = sample_loader_class().get_processed_path(spec2)
    assert p1 != p2


def test_different_instances_same_path(sample_loader_class, sample_spatial_spec, tmp_path):
    with set_config(path_data_processed=tmp_path):
        p1 = sample_loader_class().get_processed_path(sample_spatial_spec)
        p2 = sample_loader_class().get_processed_path(sample_spatial_spec)
    assert p1 == p2


def test_is_processed_false_no_file(sample_loader_class, sample_spatial_spec, tmp_path):
    with set_config(path_data_processed=tmp_path):
        assert not sample_loader_class().is_processed(sample_spatial_spec)


def test_is_processed_false_no_hash_file(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert not loader.is_processed(sample_spatial_spec)


def test_is_processed_false_hash_mismatch(sample_loader_class, sample_spatial_spec, tmp_path):

    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        loader.get_state_hash_path(sample_spatial_spec).write_text(
            json.dumps({'state_hash': 'stale'}),
        )
        assert not loader.is_processed(sample_spatial_spec)


def test_is_processed_true_after_write_hash(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.mkdir(parents=True, exist_ok=True)
        path.touch()
        loader.write_state_hash(sample_spatial_spec)
        assert loader.is_processed(sample_spatial_spec)


def test_is_processed_symlink_valid(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = tmp_path / 'target.tif'
        target.touch()
        path.symlink_to(target)
        loader.write_state_hash(sample_spatial_spec)
        assert loader.is_processed(sample_spatial_spec)


def test_is_processed_symlink_broken(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(tmp_path / 'nonexistent.tif')
        assert not loader.is_processed(sample_spatial_spec)


def test_process_creates_file(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        loader.process(sample_spatial_spec)
        assert loader.get_processed_path(sample_spatial_spec).exists()


def test_load_returns_data(sample_loader_class, sample_spatial_spec, tmp_path):
    with set_config(path_data_processed=tmp_path):
        data = load(sample_loader_class(), spec=sample_spatial_spec)
        assert data.rio.crs == sample_spatial_spec.crs


def test_process_fn_calls_processor(sample_loader_class, sample_spatial_spec, tmp_path):
    calls = []

    class DummyProcessor:
        default_driver = None
        ext = 'tif'

        def __call__(self, path, spec):
            calls.append((path, spec))

    loader = sample_loader_class()
    loader.processor = DummyProcessor()

    with set_config(path_data_processed=tmp_path):
        expected_path = loader.get_processed_path(sample_spatial_spec)
        loader.process(sample_spatial_spec)

    assert calls == [
        (expected_path, sample_spatial_spec),
    ]


def test_write_and_read_state_hash(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        loader.get_processed_path(sample_spatial_spec)
        loader.write_state_hash(sample_spatial_spec)
        assert loader.read_state_hash(sample_spatial_spec) == loader.get_state_hash()


def test_read_state_hash_none_if_missing(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        loader.get_processed_path(sample_spatial_spec)
        assert loader.read_state_hash(sample_spatial_spec) is None


def test_driver_fallback_to_processor_default(sample_loader_class):
    assert callable(sample_loader_class().driver)


def test_driver_raises_if_no_processor():
    class Bare(DataLoader):
        pass

    with pytest.raises(NotImplementedError):
        _ = Bare().driver
