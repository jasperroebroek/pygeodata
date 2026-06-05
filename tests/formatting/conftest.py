from collections.abc import Generator

import pytest

from pygeodata.tracked_object import TrackedObject


@pytest.fixture(autouse=True)
def restore_registry() -> Generator[None, None, None]:
    saved = dict(TrackedObject._registry)
    yield
    TrackedObject._registry = saved
    TrackedObject.clear_function_caches()
