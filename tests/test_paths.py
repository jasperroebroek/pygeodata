from pygeodata.spec import SpatialSpec
from tests.fixtures.data import EmptyLoader, LoaderA, LoaderB, LoaderC, LoaderD


def test_exclude_and_private_params() -> None:
    loader = LoaderC(target='deforestation', n_jobs=8, _private_state=True)
    params = loader.get_params()
    assert 'target' in params
    assert 'n_jobs' in params
    assert '_private_state' not in params


def test_deterministic_sorting() -> None:
    loader1 = LoaderB(features=[LoaderA(year=1990), LoaderA(year=2020)])
    loader2 = LoaderB(features=[LoaderA(year=2020), LoaderA(year=1990)])
    assert loader1.get_params() == loader2.get_params()


def test_processed_dir_is_cache_root_slash_state_hash(sample_spatial_spec: SpatialSpec) -> None:
    loader = LoaderA(year=2000)
    expected = loader.get_cache_root() / loader.get_state_hash(sample_spatial_spec)
    assert loader.get_processed_dir(sample_spatial_spec) == expected


def test_mixed_list_state_hash_no_crash(sample_spatial_spec: SpatialSpec) -> None:
    h = LoaderD(items=[LoaderA(year=2000), 'europe']).get_state_hash(sample_spatial_spec)
    assert isinstance(h, str)
    assert len(h) == 64


def test_parameterless_loader_hash() -> None:
    EmptyLoader.get_dependency_tree_hash.cache_clear()
    h = EmptyLoader().get_instance_hash()
    assert isinstance(h, str)
    assert len(h) == 64
