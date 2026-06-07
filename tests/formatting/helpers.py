from enum import Enum

from pygeodata.data import Data


class Color(Enum):
    RED = 1
    GREEN = 2


class Size(Enum):
    RED = 1


class IntColor(int, Enum):
    ONE = 1
    TWO = 2


def make_artifact(class_name: str, params: dict | None = None) -> Data:
    _params = params or {}
    Data._registry.pop(class_name, None)
    cls = type(
        class_name,
        (Data,),
        {'get_params': lambda _self: _params},
    )
    return cls()
