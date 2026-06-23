from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from pyproj import CRS


class BoundingBox(NamedTuple):
    left: float
    bottom: float
    right: float
    top: float


class SpecKeys(StrEnum):
    SPEC = 'spec'
    CRS = 'crs'
    TRANSFORM = 'transform'
    RESOLUTION = 'resolution'
    SHAPE = 'shape'
    BOUNDS = 'bounds'
    BOUNDS_LATLON = 'bounds_latlon'


RasterShape = tuple[float, float]


def format_resolution(resolution: Any, crs: Any) -> str | None:
    """Format a resolution value (scalar or 2-tuple) as a human-readable string with units."""
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    if not resolution:
        return None
    try:
        vals = list(resolution) if isinstance(resolution, (list, tuple)) else None
        if not vals:
            return str(resolution)

        unit = 'm'
        if crs:
            try:
                crs_obj = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)
                axis_unit = crs_obj.axis_info[0].unit_name.lower()
                if 'degree' in axis_unit:
                    unit = '°'
                elif 'foot' in axis_unit or 'feet' in axis_unit:
                    unit = 'ft'
            except CRSError:
                pass

        def fmt(v: Any) -> str:
            return str(int(v)) if float(v) == int(float(v)) else f'{float(v):.4g}'

        if len(vals) >= 2 and vals[0] == vals[1]:
            return f'{fmt(vals[0])}{unit}'
        if len(vals) >= 2:
            return f'{fmt(vals[0])} × {fmt(vals[1])}{unit}'
        return f'{fmt(vals[0])}{unit}'
    except (TypeError, ValueError):
        return str(resolution)


@lru_cache(maxsize=32)
def _get_transformer(crs_wkt: str):
    from pyproj import CRS, Transformer

    crs_obj = CRS.from_user_input(crs_wkt)
    return Transformer.from_crs(crs_obj, 'EPSG:4326', always_xy=True)


def compute_bounds_latlon(
    bounds: Any,
    crs: CRS | str,
) -> tuple[float, float, float, float] | None:
    """Reproject a native bounding box to (lat_min, lon_min, lat_max, lon_max).

    Returns None if the inputs are missing or the projection fails.
    """
    from pyproj import CRS
    from pyproj.exceptions import ProjError

    if not bounds or not crs:
        return None
    try:
        coords = list(bounds) if isinstance(bounds, (list, tuple)) else None
        if not coords or len(coords) != 4:
            return None

        crs_key = crs.to_wkt() if isinstance(crs, CRS) else crs
        t = _get_transformer(crs_key)
        xmin, ymin, xmax, ymax = coords
        lon_min, lat_min = t.transform(xmin, ymin)
        lon_max, lat_max = t.transform(xmax, ymax)
        return (
            round(lat_min, 1),
            round(lon_min, 1),
            round(lat_max, 1),
            round(lon_max, 1),
        )
    except ProjError:
        return None


@dataclass(frozen=True)
class Shape:
    shape_tuple: tuple[int]

    def __iter__(self):
        return iter(self.shape_tuple)


@dataclass(frozen=True)
class SpatialSpec:
    crs: Any
    transform: Any | None = None
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

    @property
    def bounds_latlon(self) -> tuple[float, float, float, float] | None:
        try:
            b = self.bounds
            return compute_bounds_latlon(
                [b.left, b.bottom, b.right, b.top],
                self.crs,
            )
        except ValueError:
            return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialSpec:
        from affine import Affine
        from pyproj import CRS

        crs_raw = data.get(SpecKeys.CRS)
        crs = CRS.from_user_input(crs_raw) if crs_raw else None
        transform_d = data.get(SpecKeys.TRANSFORM)
        transform = (
            Affine(
                transform_d['a'],
                transform_d['b'],
                transform_d['c'],
                transform_d['d'],
                transform_d['e'],
                transform_d['f'],
            )
            if transform_d
            else None
        )
        shape = data.get(SpecKeys.SHAPE)
        return cls(crs=crs, transform=transform, shape=tuple(shape) if shape else None)

    @classmethod
    def from_raster_file(cls, path: str | Path) -> SpatialSpec:
        import rasterio as rio

        with rio.open(path) as src:
            return cls(crs=src.crs, transform=src.transform, shape=src.shape)

    @classmethod
    def from_shape_file(cls, path: str | Path) -> SpatialSpec:
        import fiona

        with fiona.open(path) as src:
            return cls(crs=src.crs)

    @classmethod
    def from_file(cls, path: str | Path) -> SpatialSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))

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
        crs_str = self.crs.to_string() if self.crs is not None else 'None'
        return f'SpatialSpec(crs={crs_str}, transform={transform_str}, shape={self.shape})'

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
            SpecKeys.CRS: self.crs.to_string() if self.crs is not None else None,
            SpecKeys.TRANSFORM: transform_dict,
            SpecKeys.SHAPE: self.shape,
            SpecKeys.RESOLUTION: self.resolution if self.transform is not None else None,
            SpecKeys.BOUNDS: self.bounds
            if self.transform is not None or (self.crs is not None and self.crs.area_of_use is not None)
            else None,
        }
