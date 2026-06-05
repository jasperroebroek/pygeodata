from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

import fiona
import rasterio as rio
from affine import Affine
from pyproj import CRS
from rasterio.coords import BoundingBox

T = TypeVar('T')
RasterShape = tuple[float, float]


class SpecKeys(StrEnum):
    SPEC = 'spec'
    CRS = 'crs'
    TRANSFORM = 'transform'
    RESOLUTION = 'resolution'
    SHAPE = 'shape'
    BOUNDS = 'bounds'
    BOUNDS_LATLON = 'bounds_latlon'


@dataclass(frozen=True)
class SpatialSpec:
    crs: CRS = field(default_factory=lambda: CRS.from_epsg(4326))
    transform: Affine | None = None
    shape: RasterShape | None = None

    @property
    def is_fully_defined(self) -> bool:
        return self.transform is not None and self.shape is not None

    @property
    def resolution(self) -> tuple[int, int]:
        if self.transform is None:
            raise ValueError('No transform provided')
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def bounds(self) -> BoundingBox:
        if self.transform is None:
            if self.crs.area_of_use is None:
                raise ValueError('CRS area of use not defined and no transform provided')
            return BoundingBox(*self.crs.area_of_use.bounds)

        height, width = self.shape
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (width, height)
        return BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        bounds = self.bounds
        return (bounds.left, bounds.right, bounds.bottom, bounds.top)

    @classmethod
    def from_raster_file(cls, path: str | Path) -> Self:
        with rio.open(path) as src:
            return cls(crs=src.crs, transform=src.transform, shape=src.shape)

    @classmethod
    def from_shape_file(cls, path: str | Path) -> Self:
        with fiona.open(path) as src:
            return cls(crs=src.crs)

    def __repr__(self):
        transform_str = (
            (
                f'Affine('
                f'{self.transform.a:.2f}, '
                f'{self.transform.b:.2f}, '
                f'{self.transform.c:.2f}, '
                f'{self.transform.d:.2f}, '
                f'{self.transform.e:.2f}, '
                f'{self.transform.f:.2f})'
            )
            if self.transform is not None
            else 'None'
        )
        return f'SpatialSpec(crs={self.crs.to_string()}, transform={transform_str}, shape={self.shape})'

    def to_dict(self) -> dict[str, Any]:
        transform_dict = (
            None
            if self.transform is None
            else {
                'a': self.transform.a,
                'b': self.transform.b,
                'c': self.transform.c,
                'd': self.transform.d,
                'e': self.transform.e,
                'f': self.transform.f,
            }
        )

        return {
            SpecKeys.CRS: self.crs.to_string(),
            SpecKeys.TRANSFORM: transform_dict,
            SpecKeys.SHAPE: self.shape,
            SpecKeys.RESOLUTION: self.resolution if self.transform is not None else None,
            SpecKeys.BOUNDS: self.bounds if self.transform is not None or self.crs.area_of_use is not None else None,
        }


class Processor(Protocol):
    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None: ...


class Driver(Protocol):
    default_ext: str

    def __call__(self, path: str | Path) -> Any: ...


@runtime_checkable
class AllowsFormatting(Protocol):
    def format_as_json(self, spec: SpatialSpec | None = None) -> Any: ...
    def format_for_display(self) -> str: ...


@runtime_checkable
class HasParameters(Protocol):
    def get_params(self, exclude: bool = True) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ClassNode:
    cls: type
    name: str
    color: str = '#ffffff'


@dataclass
class DependencyGraph:
    nodes: set[ClassNode] = field(default_factory=set)
    call_edges: set[tuple[ClassNode, ClassNode]] = field(default_factory=set)
    inheritance_edges: set[tuple[ClassNode, ClassNode]] = field(default_factory=set)


@dataclass(frozen=True)
class RuntimeNode:
    node_id: str
    cls: type
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    call_dependencies: tuple[type, ...] = ()
    inheritance_dependencies: tuple[type, ...] = ()


@dataclass(frozen=True)
class RuntimeParamEdge:
    src_id: str
    dst_id: str
    param_name: str


@dataclass
class RuntimeDependencyGraph:
    nodes: dict[str, RuntimeNode] = field(default_factory=dict)
    param_edges: set[RuntimeParamEdge] = field(default_factory=set)


@dataclass(frozen=True)
class SymbolTables:
    imported_objects: dict[str, str]
    module_aliases: dict[str, str]
    local_defs: set[str]


@dataclass(frozen=True)
class Shape:
    shape_tuple: tuple[int]

    def __iter__(self):
        return iter(self.shape_tuple)
