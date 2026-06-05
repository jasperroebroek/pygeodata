import pytest

from pygeodata.data import Data
from tests.fixtures.data import DummyLoader, EmptyLoader


def test_driver_fallback_to_processor_default(simple_data_loader: Data) -> None:
    assert callable(simple_data_loader.driver)


def test_driver_raises_if_no_processor() -> None:
    with pytest.raises(NotImplementedError):
        EmptyLoader().driver


def test_driver_raises_if_processor_has_no_default_driver() -> None:
    with pytest.raises(AttributeError):
        DummyLoader().driver
