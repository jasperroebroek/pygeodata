from dataclasses import asdict

from pygeodata.config import set_config
from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec


def test_loader_initialization(sample_loader_class):
    """Test DataEntry initialization with default values."""
    loader = sample_loader_class()
    assert loader.name == 'sample'
    assert hasattr(loader, 'processor')
    assert hasattr(loader, 'driver')
    assert loader.class_name == 'Sample'
    assert loader.get_params() == {}


def test_loader_get_processed_path(sample_loader_class, sample_spatial_spec, tmp_path):
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)

        assert path.parent.exists()
        assert path.suffix == '.tif'
        assert str(tmp_path) in str(path)


def test_is_processed_false(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test is_processed when file doesn't exist."""
    with set_config(path_data_processed=tmp_path):
        assert not sample_loader_class().is_processed(sample_spatial_spec)


def test_is_processed_true(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test is_processed when file exists."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        assert loader.is_processed(sample_spatial_spec)


def test_process_creates_file(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test that process creates the expected output file."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        loader.process(sample_spatial_spec)
        path = loader.get_processed_path(sample_spatial_spec)
        assert path.exists()


def test_load_with_params(sample_loader_class_complex, sample_spatial_spec):
    """Test that parameters are included in the generated path."""
    assert set(sample_loader_class_complex(10, 10).get_params()) == {'time', 'resolution'}


def test_loader_repr(sample_loader_class):
    """Test string representation of loader."""
    loader = sample_loader_class()
    repr_str = repr(loader)
    assert repr_str.startswith('Sample(')
    assert repr_str.endswith(')')


def test_loader_repr_with_params(sample_loader_class_complex):
    """Test string representation includes parameters sorted alphabetically."""
    loader = sample_loader_class_complex(time=10, resolution=5)
    repr_str = repr(loader)
    assert 'ComplexSample(' in repr_str
    assert 'resolution=5' in repr_str
    assert 'time=10' in repr_str
    assert repr_str.index('resolution') < repr_str.index('time')


def test_name_conversion_simple(sample_loader_class):
    """Test class name to snake_case conversion for simple names."""
    loader = sample_loader_class()
    assert loader.name == 'sample'


def test_name_conversion_camel_case():
    """Test class name conversion handles camelCase properly."""

    class MyDataLoader(DataLoader):
        @property
        def processor(self):
            return lambda path, spec: None

    loader = MyDataLoader()
    assert loader.name == 'my_data'


def test_name_conversion_acronyms():
    """Test class name conversion handles acronyms properly."""

    class XMLHTTPLoader(DataLoader):
        @property
        def processor(self):
            return lambda path, spec: None

    loader = XMLHTTPLoader()
    assert loader.name == 'xmlhttp'


def test_name_conversion_mixed():
    """Test class name conversion handles mixed acronyms and camelCase."""

    class USGSElevationLoader(DataLoader):
        @property
        def processor(self):
            return lambda path, spec: None

    loader = USGSElevationLoader()
    assert loader.name == 'usgs_elevation'


def test_get_params_excludes_private_attributes(sample_loader_class_complex):
    """Test that get_params excludes private attributes."""
    loader = sample_loader_class_complex(time=10, resolution=5)
    loader._private_attr = 'should_not_appear'
    params = loader.get_params()
    assert '_private_attr' not in params


def test_get_params_multiple_params(sample_loader_class_complex):
    """Test get_params returns all instance attributes."""
    loader = sample_loader_class_complex(time=15, resolution=20)
    params = loader.get_params()
    assert params['time'] == 15
    assert params['resolution'] == 20
    assert len(params) == 2


def test_process_skips_if_already_processed(sample_loader_class, sample_spatial_spec, tmp_path, mocker):
    """Test that process doesn't reprocess if file already exists."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

        # Mock the processor to ensure it's not called
        mock_processor = mocker.patch.object(loader, 'processor')
        mock_processor.ext = 'tif'
        loader.process(sample_spatial_spec)
        mock_processor.assert_not_called()


def test_process_calls_processor_if_not_processed(sample_loader_class, sample_spatial_spec, tmp_path, mocker):
    """Test that process calls processor when file doesn't exist."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        # Mock the processor
        mock_processor = mocker.MagicMock()
        mocker.patch.object(loader, 'processor', mock_processor)

        loader.process(sample_spatial_spec)

        path = loader.get_processed_path(sample_spatial_spec)
        mock_processor.assert_called_once_with(path, sample_spatial_spec)


def test_load_returns_data(sample_loader_class, sample_spatial_spec, tmp_path, mocker):
    """Test that load returns data from driver."""
    loader = sample_loader_class()
    expected_data = {'test': 'data'}

    with set_config(path_data_processed=tmp_path):
        # Mock driver.load to return test data
        mock_load = mocker.MagicMock(return_value=expected_data)
        mocker.patch.object(loader.driver, 'load', mock_load)

        result = loader.load(sample_spatial_spec)

        assert result == expected_data
        mock_load.assert_called_once()


def test_load_processes_before_loading(sample_loader_class, sample_spatial_spec, tmp_path, mocker):
    """Test that load calls process before loading data."""
    loader = sample_loader_class()

    with set_config(path_data_processed=tmp_path):
        mock_process = mocker.patch.object(loader, 'process')
        mock_load = mocker.patch.object(loader.driver, 'load', return_value=None)

        loader.load(sample_spatial_spec)

        # Ensure process is called before load
        mock_process.assert_called_once_with(sample_spatial_spec)
        mock_load.assert_called_once()


def test_is_processed_with_valid_symlink(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test is_processed returns True for valid symlink."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create a target file and symlink to it
        target = tmp_path / 'target.tif'
        target.touch()
        path.symlink_to(target)

        assert loader.is_processed(sample_spatial_spec)


def test_is_processed_with_broken_symlink(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test is_processed behavior with broken symlink."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create a symlink to non-existent target
        target = tmp_path / 'nonexistent.tif'
        path.symlink_to(target)

        assert not loader.is_processed(sample_spatial_spec)


def test_driver_fallback_to_processor_default(sample_loader_class):
    """Test that driver property falls back to processor's default_driver."""
    loader = sample_loader_class()
    driver = loader.driver
    assert driver is not None
    assert hasattr(driver, 'load')


def test_get_processed_path_creates_parent_directory(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test that get_processed_path creates parent directories."""
    loader = sample_loader_class()
    with set_config(path_data_processed=tmp_path):
        path = loader.get_processed_path(sample_spatial_spec)
        # Parent should be created by get_processed_path
        assert path.parent.exists()


def test_multiple_specs_different_paths(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test that different spatial specs generate different paths."""
    loader = sample_loader_class()

    # Create a second spec with different properties
    spec2 = SpatialSpec(**asdict(sample_spatial_spec))
    spec2.shape = (2, 3)

    with set_config(path_data_processed=tmp_path):
        path1 = loader.get_processed_path(sample_spatial_spec)
        path2 = loader.get_processed_path(spec2)

        assert path1 != path2


def test_same_spec_same_path(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test that same spec always generates the same path."""
    loader = sample_loader_class()

    with set_config(path_data_processed=tmp_path):
        path1 = loader.get_processed_path(sample_spatial_spec)
        path2 = loader.get_processed_path(sample_spatial_spec)

        assert path1 == path2


def test_different_loader_instances_same_behavior(sample_loader_class, sample_spatial_spec, tmp_path):
    """Test that different instances of the same loader class behave identically."""
    loader1 = sample_loader_class()
    loader2 = sample_loader_class()

    with set_config(path_data_processed=tmp_path):
        path1 = loader1.get_processed_path(sample_spatial_spec)
        path2 = loader2.get_processed_path(sample_spatial_spec)

        assert path1 == path2
        assert loader1.name == loader2.name
        assert loader1.class_name == loader2.class_name
