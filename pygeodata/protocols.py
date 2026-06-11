from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from pygeodata.spec import SpatialSpec

T = TypeVar('T')


class Processor(Protocol):
    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None: ...


class Driver(Protocol):
    default_ext: str

    def __call__(self, path: str | Path) -> Any: ...


@runtime_checkable
class AllowsFormatting(Protocol):
    def format_as_json(self, spec: SpatialSpec | None = None) -> Any: ...
    def format_for_display(self) -> str: ...
