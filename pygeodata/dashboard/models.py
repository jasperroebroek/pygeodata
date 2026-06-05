from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FileRef:
    label: str
    path: str
    kind: str


@dataclass(slots=True)
class SpecInfo:
    crs: str | None = None
    resolution: str | None = None
    shape: str | None = None
    bounds: str | None = None
    bounds_latlon: tuple | None = None

    @classmethod
    def from_spec_json(cls, spec: dict) -> SpecInfo:
        """Build a SpecInfo from a raw .spec.json dict."""
        from pyproj import CRS, Transformer

        raw_crs = spec.get('crs')
        resolution = spec.get('resolution')
        shape = spec.get('shape')
        bounds = spec.get('bounds')

        crs_str = None if raw_crs in ('', None) else str(raw_crs)

        # Resolution → human-readable string
        resolution_fmt = cls._format_resolution(resolution, crs_str)

        # Bounds → lat/lon tuple
        bounds_latlon = None
        if bounds and crs_str:
            try:
                coords = list(bounds) if isinstance(bounds, (list, tuple)) else None
                if coords and len(coords) == 4:
                    t = Transformer.from_crs(crs_str, 'EPSG:4326', always_xy=True)
                    xmin, ymin, xmax, ymax = coords
                    lon_min, lat_min = t.transform(xmin, ymin)
                    lon_max, lat_max = t.transform(xmax, ymax)
                    bounds_latlon = (
                        round(lat_min, 1), round(lon_min, 1),
                        round(lat_max, 1), round(lon_max, 1),
                    )
            except Exception:
                pass

        return cls(
            crs=crs_str,
            resolution=resolution_fmt,
            shape=None if shape in ('', None) else str(shape),
            bounds=None if bounds in ('', None) else str(bounds),
            bounds_latlon=bounds_latlon,
        )

    @staticmethod
    def _format_resolution(resolution: Any, crs: str | None) -> str | None:
        if not resolution:
            return None
        try:
            from pyproj import CRS as ProjCRS
            vals = list(resolution) if isinstance(resolution, (list, tuple)) else None
            if not vals:
                return str(resolution)

            unit = 'm'
            if crs:
                try:
                    c = ProjCRS.from_user_input(crs)
                    axis_unit = c.axis_info[0].unit_name.lower() if c.axis_info else ''
                    if 'degree' in axis_unit:
                        unit = '°'
                    elif 'foot' in axis_unit or 'feet' in axis_unit:
                        unit = 'ft'
                except Exception:
                    pass

            def fmt(v: Any) -> str:
                return str(int(v)) if float(v) == int(float(v)) else f'{float(v):.4g}'

            if len(vals) >= 2 and vals[0] == vals[1]:
                return f'{fmt(vals[0])}{unit}'
            if len(vals) >= 2:
                return f'{fmt(vals[0])} × {fmt(vals[1])}{unit}'
            return f'{fmt(vals[0])}{unit}'
        except Exception:
            return str(resolution)


@dataclass(slots=True)
class ParamRow:
    path: str
    key_group: str
    final_key: str
    value_text: str       # plain text, unescaped — HTML escaping is the frontend's job
    value_type: str
    search_blob: str
    depth: int


@dataclass(slots=True)
class LinkedEntry:
    """A Data/TrackedObject reference embedded in another entry's params."""
    param_name: str
    class_name: str
    state_hash: str | None
    params_summary: dict[str, str]   # plain-text key→value pairs


@dataclass
class EntryInfo:
    record_id: str
    class_name: str
    object_type: str
    params_path: str
    spec_path: str | None
    state_hash_path: str | None
    execution_graph_path: str | None
    state_hash: str | None
    instance_hash: str | None
    params: dict[str, Any]
    spec: SpecInfo
    rows: list[ParamRow]
    linked_entries: list[LinkedEntry] = field(default_factory=list)
    co_output_hashes: list[str] = field(default_factory=list)
    co_outputs: list[EntryInfo] = field(default_factory=list)
    primary_file: FileRef | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    dep_hash_stale: bool = False


@dataclass(slots=True)
class GroupInfo:
    class_name: str
    object_type: str
    record_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ClassInfo:
    class_name: str
    object_type: str
    loaded: bool
    dependency_names: list[str] = field(default_factory=list)
    class_source_path: str | None = None
    class_graph_path: str | None = None
    class_registry_path: str | None = None
    source_stale: bool = False   # live source hash ≠ stored source hash
    deps_stale: bool = False     # live dependency_tree_hash ≠ stored dependency_tree_hash
