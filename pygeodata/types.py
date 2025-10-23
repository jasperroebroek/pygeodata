from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from affine import Affine
from pyproj import CRS
from rasterio.coords import BoundingBox

RasterShape = tuple[int, int]


@dataclass
class SpatialSpec:
    crs: CRS
    transform: Affine
    shape: RasterShape

    def resolution(self) -> tuple[int, int]:
        return (abs(self.transform.a), abs(self.transform.e))

    def bounds(self) -> BoundingBox:
        height, width = self.shape
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (width, height)
        return BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    def __repr__(self):
        transform_str = (
            f'Affine('
            f'{self.transform.a:.2f}, '
            f'{self.transform.b:.2f}, '
            f'{self.transform.c:.2f}, '
            f'{self.transform.d:.2f}, '
            f'{self.transform.e:.2f}, '
            f'{self.transform.f:.2f})'
        )
        return f'SpatialSpec(crs={self.crs.to_string()}, transform={transform_str}, shape={self.shape})'


class Processor(Protocol):
    def __call__(self, dst_path: str | Path, spec: SpatialSpec) -> None: ...


class Driver(Protocol):
    def load(self, path: str | Path) -> Any: ...
