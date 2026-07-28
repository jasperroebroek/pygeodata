from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from pygeodata.hash import calculate_dict_hash

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
    RESOLUTION_DISPLAY = 'resolution_display'
    SHAPE = 'shape'
    BOUNDS = 'bounds'
    BOUNDS_LATLON = 'bounds_latlon'
    BOUNDS_DISPLAY = 'bounds_display'


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
        precision = 0
        if crs:
            try:
                crs_obj = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)
                axis_unit = crs_obj.axis_info[0].unit_name.lower()
                if 'degree' in axis_unit:
                    unit = '°'
                    precision = 2
                elif 'foot' in axis_unit or 'feet' in axis_unit:
                    unit = 'ft'
                    precision = 2
            except CRSError:
                pass

        def fmt(v: Any) -> str:
            v = round(float(v), precision)
            return str(int(v)) if v == int(v) else str(v)

        if len(vals) >= 2 and math.isclose(vals[0], vals[1], rel_tol=1e-9):
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


def _bounds_latlon_from_outline(
    t: Any,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    n: int = 100,
) -> tuple[float, float, float, float] | None:
    """Sample points densely along the rectangle's outline and reproject each.

    Used when the corners themselves fall outside the projection's valid
    domain (e.g. a global raster in a lens-shaped equal-area projection) —
    the outline still crosses into the valid region, so this recovers the
    true reachable lat/lon extent instead of giving up.
    """
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    edge_x = np.concatenate([xs, xs, np.full(n, xmin), np.full(n, xmax)])
    edge_y = np.concatenate([np.full(n, ymax), np.full(n, ymin), ys, ys])

    lons, lats = t.transform(edge_x, edge_y)
    finite = np.isfinite(lons) & np.isfinite(lats)
    if not np.any(finite):
        return None

    lat_min, lat_max = float(lats[finite].min()), float(lats[finite].max())
    lon_min, lon_max = float(lons[finite].min()), float(lons[finite].max())
    return (round(lat_min, 1), round(lon_min, 1), round(lat_max, 1), round(lon_max, 1))


def compute_bounds_latlon(
    bounds: Any,
    crs: CRS | str,
) -> tuple[float, float, float, float] | None:
    """Reproject a native bounding box to (lat_min, lon_min, lat_max, lon_max).

    Returns None if the inputs are missing or the projection fails everywhere.
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
        result = (lat_min, lon_min, lat_max, lon_max)

        if np.all(np.isfinite(result)):
            return tuple(round(v, 1) for v in result)

        return _bounds_latlon_from_outline(t, xmin, ymin, xmax, ymax)
    except ProjError:
        return None


def _fmt_coord(v: float, pos: str, neg: str) -> str:
    return f'{abs(v)}° {pos if v >= 0 else neg}'


def format_bounds_latlon(bl: tuple[float, float, float, float] | None) -> str | None:
    """Format a (lat_min, lon_min, lat_max, lon_max) tuple as two corner points with N/S/E/W."""
    if not bl or len(bl) != 4:
        return None
    lat_min, lon_min, lat_max, lon_max = bl
    sw = f'{_fmt_coord(lat_min, "N", "S")}, {_fmt_coord(lon_min, "E", "W")}'
    ne = f'{_fmt_coord(lat_max, "N", "S")}, {_fmt_coord(lon_max, "E", "W")}'
    return f'{sw} → {ne}'


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
    def resolution(self) -> tuple[float, float] | None:
        if self.transform is None:
            return None
        return (abs(self.transform.a), abs(self.transform.e))

    @property
    def resolution_display(self) -> str | None:
        """Human-readable resolution: formatted value, or None if unset."""
        return format_resolution(self.resolution, self.crs)

    @property
    def bounds(self) -> BoundingBox | None:
        if self.transform is None:
            return None

        height, width = self.shape
        x0, y0 = self.transform * (0, 0)
        x1, y1 = self.transform * (width, height)
        return BoundingBox(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @property
    def extent(self) -> tuple[float, float, float, float] | None:
        bounds = self.bounds
        if bounds is None:
            return None
        return (bounds.left, bounds.right, bounds.bottom, bounds.top)

    @property
    def bounds_latlon(self) -> tuple[float, float, float, float] | None:
        b = self.bounds
        if b is None:
            return None
        return compute_bounds_latlon(
            [b.left, b.bottom, b.right, b.top],
            self.crs,
        )

    @property
    def bounds_display(self) -> str | None:
        """Human-readable bounds for display: lat/lon with N/S/E/W when available,
        else the raw native tuple, else None if unset.
        """
        formatted = format_bounds_latlon(self.bounds_latlon)
        if formatted is not None:
            return formatted
        b = self.bounds
        return str(tuple(b)) if b else None

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
            SpecKeys.BOUNDS: self.bounds,
        }

    def get_hash(self) -> str:
        return calculate_dict_hash(self.to_dict())
