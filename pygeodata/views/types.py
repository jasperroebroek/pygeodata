from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pygeodata._version import FORMAT_VERSION
from pygeodata.enums import SpecKeys
from pygeodata.spec import SpatialSpec, format_resolution

if TYPE_CHECKING:
    from pygeodata.catalog.types import FileRef, LinkedEntry, ParamRow
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


@dataclass
class EntryInfo:
    class_name: str
    object_type: str | None
    entry_dir: str
    params_path: str
    spec_path: str | None
    state_hash_path: str | None
    execution_graph_path: str | None
    state_hash: str | None
    instance_hash: str | None
    params_hash: str | None
    spec_hash: str | None
    params: dict[str, Any]
    spec: SpecInfo
    rows: list[ParamRow]
    linked_entries: list[LinkedEntry] = field(default_factory=list)
    co_output_hashes: list[str] = field(default_factory=list)
    primary_file: FileRef | None = None
    sidecar_files: list[FileRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    format_version: int = field(default=0)
    # Fields filled during assembly by discover_entries
    # record_id == state_hash (the key in EntryRegistry.records)
    record_id: str = ''
    dep_hash: str | None = None
    dep_hash_stale: bool = False
    # Resolved back-references — excluded from serialisation to avoid infinite recursion
    co_outputs: list[EntryInfo] = field(default_factory=list)

    @property
    def format_version_stale(self) -> bool:
        return self.format_version != FORMAT_VERSION

    def file_path(self, ref: FileRef) -> Path:
        return Path(self.entry_dir) / ref.relpath

    def _file_payload(self, ref: FileRef) -> dict[str, str]:
        return {'label': Path(ref.relpath).name, 'path': str(self.file_path(ref)), 'kind': ref.kind}

    def to_dict(self) -> dict[str, Any]:
        # co_outputs is intentionally excluded — only hashes are serialised.
        return {
            'record_id': self.record_id,
            'class_name': self.class_name,
            'object_type': self.object_type,
            'params_path': self.params_path,
            'spec_path': self.spec_path,
            'state_hash_path': self.state_hash_path,
            'execution_graph_path': self.execution_graph_path,
            'state_hash': self.state_hash,
            'instance_hash': self.instance_hash,
            'params_hash': self.params_hash,
            'spec_hash': self.spec_hash,
            'params': self.params,
            SpecKeys.SPEC: dataclasses.asdict(self.spec),
            'rows': [dataclasses.asdict(r) for r in self.rows],
            'linked_entries': [dataclasses.asdict(le) for le in self.linked_entries],
            'co_output_hashes': self.co_output_hashes,
            'primary_file': self._file_payload(self.primary_file) if self.primary_file else None,
            'sidecar_files': [self._file_payload(f) for f in self.sidecar_files],
            'warnings': list(self.warnings),
            'error': self.error,
            'format_version': self.format_version,
            'dep_hash': self.dep_hash,
            'dep_hash_stale': self.dep_hash_stale,
        }


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
    loaded: bool
    call_dependency_names: list[str] = field(default_factory=list)
    inheritance_dependency_names: list[str] = field(default_factory=list)
    class_source_path: str | None = None
    class_graph_path: str | None = None
    class_registry_path: str | None = None
    class_tree_path: str | None = None
    source_stale: bool = False  # live source hash ≠ stored source hash
    deps_stale: bool = False  # live dependency_tree_hash ≠ stored dependency_tree_hash
