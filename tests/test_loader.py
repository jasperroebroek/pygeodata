from pygeodata.config import set_config


def test_data_entry_initialization(sample_loader_class):
    """Test DataEntry initialization with default values."""
    loader = sample_loader_class()
    assert loader.name == 'sample'
    assert hasattr(loader, 'processor')
    assert hasattr(loader, 'driver')
    assert loader.class_name == 'Sample'
    assert loader.get_params() == {}


def test_get_processed_path(sample_loader_class, sample_spatial_spec, tmp_path):
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
