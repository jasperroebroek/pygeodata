from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pygeodata.spec import SpatialSpec, format_resolution

if TYPE_CHECKING:
    from pygeodata.records import SpecRecord


@dataclass(slots=True)
class SpecInfo:
    crs: str | None = None
    resolution: str | None = None
    resolution_display: str | None = None
    shape: str | None = None
    bounds: str | None = None
    bounds_latlon: tuple | None = None
    bounds_display: str | None = None


@functools.cache
def spec_info(record: SpecRecord | None) -> SpecInfo:
    """The display form of a spec; cached, as entries share a handful of distinct specs."""
    if record is None or record.crs is None:
        return SpecInfo()
    spec = SpatialSpec.from_record(record)
    bounds = spec.bounds
    return SpecInfo(
        crs=record.crs,
        resolution=format_resolution(spec.resolution, spec.crs),
        resolution_display=spec.resolution_display,
        shape=None if spec.shape is None else str(spec.shape),
        bounds=str([bounds.left, bounds.bottom, bounds.right, bounds.top]) if bounds else None,
        bounds_latlon=spec.bounds_latlon,
        bounds_display=spec.bounds_display,
    )


@dataclass(slots=True)
class CodeClassState:
    """Per-class state at a specific version, with live-staleness annotation."""

    class_name: str
    object_type: str
    source_hash: str
    is_loaded: bool
    is_stale: bool


@dataclass(slots=True)
class ClassInfo:
    class_name: str
    object_type: str | None
    # whether its current code is known (from source or imports), or only its registered snapshots
    known: bool
    call_dependency_names: list[str] = field(default_factory=list)
    inheritance_dependency_names: list[str] = field(default_factory=list)
    class_source_path: str | None = None
    class_graph_path: str | None = None
    class_registry_path: str | None = None
    class_tree_path: str | None = None
    source_stale: bool = False  # the current source has not run yet (no code snapshot)
    deps_stale: bool = False  # the current dependency tree has not run yet (no tree snapshot)
