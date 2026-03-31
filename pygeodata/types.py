from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Self, TypeVar

import fiona
import rasterio as rio
from affine import Affine
from pyproj import CRS
from rasterio.coords import BoundingBox

T = TypeVar('T')
RasterShape = tuple[float, float]


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
        return f'RasterSpec(crs={self.crs.to_string()}, transform={transform_str}, shape={self.shape})'


class Processor(Protocol):
    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None: ...


class Driver(Protocol):
    default_ext: str

    def __call__(self, path: str | Path) -> Any: ...
