import json
from dataclasses import asdict
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys, set_config
from pygeodata.types import SpatialSpec
from tests.fixtures.data import (
    DummyLoader,
    EmptyLoader,
    MultiOutputLoader,
    SampleLoader,
    SimpleLoader,
    USGSElevationLoader,
    XMLHTTPLoader,
)





def test_artifact_initialization() -> None:
    loader = SimpleLoader()
    assert loader.get_file_stem() == 'simple_loader'
    assert hasattr(loader, 'processor')
    assert hasattr(loader, 'driver')
    assert loader.get_class_name() == 'SimpleLoader'
    assert loader.get_params() == {}


def test_artifact_no_processor_raises(sample_spatial_spec: SpatialSpec) -> None:
    with pytest.raises(NotImplementedError):
        EmptyLoader().process(sample_spatial_spec)


def test_name_conversion_acronyms() -> None:
    assert XMLHTTPLoader().get_file_stem() == 'xmlhttp_loader'


def test_name_conversion_mixed() -> None:
    assert USGSElevationLoader.get_file_stem() == 'usgs_elevation_loader'


def test_get_params_excludes_private(simple_data_loader: SampleLoader) -> None:
    setattr(simple_data_loader, '_private_attr', 'should_not_appear')  # noqa: B010
    assert '_private_attr' not in simple_data_loader.get_params()


def test_get_params_insertion(simple_data_loader: SampleLoader) -> None:
    setattr(simple_data_loader, 'public_attr', 'should_appear')  # noqa: B010
    assert 'public_attr' in simple_data_loader.get_params()


def test_get_params_multiple(simple_data_loader: SampleLoader) -> None:
    params = simple_data_loader.get_params()
    assert params['path'] == simple_data_loader.path
    assert params['scale'] == 2.0
    assert len(params) == 2


def test_artifact_repr() -> None:
    r = repr(EmptyLoader())
    assert r.startswith('EmptyLoader(')
    assert r.endswith(')')


def test_artifact_repr_sorted_params() -> None:
    r = repr(SampleLoader(scale=2.0, path='a.tif'))
    assert r.index('path') < r.index('scale')


def test_same_spec_same_path(sample_spatial_spec: SpatialSpec) -> None:
    p1 = SimpleLoader().get_processed_path(sample_spatial_spec)
    p2 = SimpleLoader().get_processed_path(sample_spatial_spec)
    assert p1 == p2


def test_different_specs_different_paths(sample_spatial_spec: SpatialSpec) -> None:
    d = asdict(sample_spatial_spec)
    d['shape'] = (2, 3)
    spec2 = SpatialSpec(**d)
    p1 = SimpleLoader().get_processed_path(sample_spatial_spec)
    p2 = SimpleLoader().get_processed_path(spec2)
    assert p1 != p2


def test_is_processed_false_no_file(sample_spatial_spec: SpatialSpec) -> None:
    assert not SimpleLoader().is_processed(sample_spatial_spec)


def test_is_processed_false_no_hash_file(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(sample_spatial_spec)
    path.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert not loader.is_processed(sample_spatial_spec)


def test_is_processed_false_hash_mismatch(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(sample_spatial_spec)
    path.mkdir(parents=True, exist_ok=True)
    path.touch()
    loader.resolve_cache_paths(sample_spatial_spec).state_hash_path.write_text(
        json.dumps({JSONKeys.STATE_HASH: 'stale'}),
    )
    assert not loader.is_processed(sample_spatial_spec)


def test_is_processed_true_after_write_hash(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(sample_spatial_spec)
    path.mkdir(parents=True, exist_ok=True)
    path.touch()
    loader.write_cache_metadata(sample_spatial_spec)
    assert loader.is_processed(sample_spatial_spec)


def test_is_processed_symlink_valid(sample_spatial_spec: SpatialSpec, tmp_path: Path) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(sample_spatial_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / 'target.tif'
    target.touch()
    path.symlink_to(target)
    loader.write_cache_metadata(sample_spatial_spec)
    assert loader.is_processed(sample_spatial_spec)


def test_is_processed_symlink_broken(sample_spatial_spec: SpatialSpec, tmp_path: Path) -> None:
    loader = SimpleLoader()
    path = loader.get_processed_path(sample_spatial_spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(tmp_path / 'nonexistent.tif')
    assert not loader.is_processed(sample_spatial_spec)


def test_process_creates_file(sample_spatial_spec: SpatialSpec, sample_geotiff: Path) -> None:
    loader = SampleLoader(path=sample_geotiff, scale=2.0)
    loader.process(sample_spatial_spec)
    assert loader.get_processed_path(sample_spatial_spec).exists()


def test_process_fn_calls_processor(sample_spatial_spec: SpatialSpec) -> None:
    loader = DummyLoader()

    expected_path = loader.get_processed_path(sample_spatial_spec)
    loader.process(sample_spatial_spec)

    assert loader.processor._calls == [(expected_path, sample_spatial_spec)]


def test_write_and_read_state_hash(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.get_processed_path(sample_spatial_spec)
    loader.write_cache_metadata(sample_spatial_spec)
    assert loader.read_state_hash(sample_spatial_spec) == loader.get_state_hash(sample_spatial_spec)


def test_read_state_hash_none_if_missing(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.get_processed_path(sample_spatial_spec)
    assert loader.read_state_hash(sample_spatial_spec) is None


def test_yielded_loaders_get_state_hash_written(
    sample_spatial_spec: SpatialSpec,
    multi_output_data_loader: MultiOutputLoader,
) -> None:
    multi_output_data_loader.process(sample_spatial_spec)
    assert multi_output_data_loader.loader_1.read_state_hash(
        sample_spatial_spec,
    ) == multi_output_data_loader.loader_1.get_state_hash(sample_spatial_spec)
    assert multi_output_data_loader.loader_2.read_state_hash(
        sample_spatial_spec,
    ) == multi_output_data_loader.loader_2.get_state_hash(sample_spatial_spec)


def test_yielded_loader_is_processed(
    sample_spatial_spec: SpatialSpec,
    multi_output_data_loader: MultiOutputLoader,
) -> None:
    primary = multi_output_data_loader.loader_1
    primary.get_processed_path(sample_spatial_spec).parent.mkdir(parents=True, exist_ok=True)
    primary.get_processed_path(sample_spatial_spec).touch()

    multi_output_data_loader.process(sample_spatial_spec)
    assert primary.is_processed(sample_spatial_spec)


def test_none_return_writes_self_hash(sample_spatial_spec: SpatialSpec, sample_geotiff: Path) -> None:
    loader = SampleLoader(sample_geotiff)
    loader.process(sample_spatial_spec)
    assert loader.read_state_hash(sample_spatial_spec) == loader.get_state_hash(sample_spatial_spec)


def test_yielded_loaders_get_parameters_written(
    sample_spatial_spec: SpatialSpec,
    multi_output_data_loader: MultiOutputLoader,
) -> None:
    multi_output_data_loader.process(sample_spatial_spec)
    path = multi_output_data_loader.loader_1.get_processed_path(sample_spatial_spec)
    assert (path.parent / f'.{path.stem}.params.json').exists()
    path = multi_output_data_loader.get_processed_path(sample_spatial_spec)
    assert (path.parent / f'.{path.stem}.params.json').exists()


def test_resolve_spec_raises_when_no_spec() -> None:
    with pytest.raises(ValueError, match='No spatial specification'):
        SimpleLoader().resolve_spec(None)


def test_resolve_spec_returns_spec_unchanged_when_fully_defined(sample_spatial_spec: SpatialSpec) -> None:
    assert SimpleLoader().resolve_spec(sample_spatial_spec) is sample_spatial_spec


def test_get_ext_falls_back_to_processor_ext() -> None:
    assert SimpleLoader().get_ext() == 'tif'


def test_get_ext_raises_when_no_ext_anywhere() -> None:
    with pytest.raises((ValueError, NotImplementedError)):
        EmptyLoader().get_ext()


def test_get_src_path_raises_when_no_processor() -> None:
    with pytest.raises(NotImplementedError, match='Processor must be implemented'):
        EmptyLoader().get_src_path()


def test_get_src_path_raises_when_processor_has_no_src_path() -> None:
    with pytest.raises(NotImplementedError, match='lacks src_path'):
        DummyLoader().get_src_path()


def test_get_src_path_returns_path(sample_geotiff: Path) -> None:
    loader = SampleLoader(path=sample_geotiff)
    assert loader.get_src_path() == sample_geotiff


def test_write_cache_metadata_content(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec)
    data = json.loads(loader.resolve_cache_paths(sample_spatial_spec).state_hash_path.read_text())
    assert data[JSONKeys.STATE_HASH] == loader.get_state_hash(sample_spatial_spec)
    assert data[JSONKeys.DEPENDENCY_TREE_HASH] == loader.get_dependency_tree_hash()
    assert data[JSONKeys.INSTANCE_HASH] == loader.get_instance_hash()
    assert data[JSONKeys.CLASS_NAME] == 'SimpleLoader'
    assert data[JSONKeys.CO_OUTPUTS] == []


def test_write_cache_metadata_co_outputs(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_cache_metadata(sample_spatial_spec, co_outputs=('abc123', 'def456'))
    data = json.loads(loader.resolve_cache_paths(sample_spatial_spec).state_hash_path.read_text())
    assert data[JSONKeys.CO_OUTPUTS] == ['abc123', 'def456']


def test_is_cache_valid_false_when_hash_key_missing(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.resolve_cache_paths(sample_spatial_spec).state_hash_path.parent.mkdir(parents=True, exist_ok=True)
    loader.resolve_cache_paths(sample_spatial_spec).state_hash_path.write_text(
        json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: 'something'}),
    )
    assert not loader.is_cache_valid(sample_spatial_spec)


def test_process_skips_when_already_processed(sample_spatial_spec: SpatialSpec, sample_geotiff: Path) -> None:
    loader = SampleLoader(path=sample_geotiff)
    loader.process(sample_spatial_spec)
    mtime = loader.get_processed_path(sample_spatial_spec).stat().st_mtime
    loader.process(sample_spatial_spec)
    assert loader.get_processed_path(sample_spatial_spec).stat().st_mtime == mtime


def test_process_prints_cache_invalid_message(
    sample_spatial_spec: SpatialSpec,
    sample_geotiff: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    loader = SampleLoader(path=sample_geotiff)
    loader.process(sample_spatial_spec)
    # Overwrite hash file with a stale hash to simulate invalidation
    loader.resolve_cache_paths(sample_spatial_spec).state_hash_path.write_text(
        json.dumps({JSONKeys.STATE_HASH: 'stale', JSONKeys.DEPENDENCY_TREE_HASH: 'stale'}),
    )
    loader.process(sample_spatial_spec)
    assert 'Cache invalid' in capsys.readouterr().out
