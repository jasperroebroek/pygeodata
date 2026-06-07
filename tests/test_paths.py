import pytest

from pygeodata.config import get_config, set_config
from pygeodata.extraction import flatten_parameter_dict_for_path
from pygeodata.paths import generate_path
from tests.fixtures.data import DictLoader, DummyScenario, EmptyLoader, LoaderA, LoaderB, LoaderC, LoaderD


@pytest.fixture(
    params=[True, False],
    autouse=True,
    ids=['punctuation-on', 'punctuation-off'],
)
def use_both_filesystem_representations(request):
    with set_config(filesystem_allows_punctuation=request.param):
        yield


def test_exclude_and_private_params() -> None:
    loader = LoaderC(target='deforestation', n_jobs=8, _private_state=True)
    flat_params = flatten_parameter_dict_for_path(loader.get_params())
    assert 'target' in flat_params
    assert 'n_jobs' not in flat_params
    assert '_private_state' not in flat_params


def test_deterministic_sorting() -> None:
    loader1 = LoaderB(features=[LoaderA(year=1990), LoaderA(year=2020)])
    loader2 = LoaderB(features=[LoaderA(year=2020), LoaderA(year=1990)])

    flat_params_loader1 = flatten_parameter_dict_for_path(loader1.get_params())
    flat_params_loader2 = flatten_parameter_dict_for_path(loader2.get_params())

    assert flat_params_loader1 == flat_params_loader2


def test_nested_list_param_flattening() -> None:
    flat = flatten_parameter_dict_for_path(
        LoaderB(features=[LoaderA(year=2050), LoaderA(year=2100)]).get_params(),
    )
    correct_string = '[LoaderA, LoaderA]' if get_config().filesystem_allows_punctuation else 'seq--LoaderA _ LoaderA--'
    assert flat['features'] == correct_string
    assert flat['features_0__year'] == '2050'
    assert flat['features_1__year'] == '2100'


def test_mixed_list_no_crash() -> None:
    flat = flatten_parameter_dict_for_path(LoaderD(items=[LoaderA(year=2000), 'europe']).get_params())
    assert 'items' in flat


def test_mixed_list_state_hash_no_crash(sample_spatial_spec) -> None:
    h = LoaderD(items=[LoaderA(year=2000), 'europe']).get_state_hash(sample_spatial_spec)
    assert isinstance(h, str)
    assert len(h) == 64


def test_generate_path_hashing_and_enums(sample_spatial_spec, tmp_path) -> None:
    with set_config(max_path_param_depth=0):
        path = generate_path(
            spec=sample_spatial_spec,
            base_dir=tmp_path,
            name='LoaderC',
            target='deforestation',
            scenario=DummyScenario.SSP126,
        )
    assert path.parts[-2] == 'LoaderC'
    assert len(path.parts[-1]) == 64


def test_deeply_nested_flattening() -> None:
    inner = LoaderA(year=2000)
    middle = LoaderB(features=[inner])
    outer = LoaderB(features=[middle])
    flat = flatten_parameter_dict_for_path(outer.get_params())
    assert 'features_0__features_0__year' in flat


def test_dict_artifact_flattening() -> None:
    loader = DictLoader(mapping={'baseline': LoaderA(year=1990), 'current': LoaderA(year=2020)})
    flat = flatten_parameter_dict_for_path(loader.get_params())

    # Ensure the dictionary is flattened deterministically
    assert 'mapping_baseline__year' in flat
    assert flat['mapping_baseline__year'] == '1990'
    assert flat['mapping_current__year'] == '2020'


def test_parameterless_loader_hash() -> None:
    """A loader with no fields should still generate a valid, stable hash based on its code."""
    EmptyLoader.get_dependency_tree_hash.cache_clear()

    h1 = EmptyLoader().get_instance_hash()
    assert isinstance(h1, str)
    assert len(h1) == 64
