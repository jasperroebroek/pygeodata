from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from pygeodata.config import FORMAT_VERSION
from pygeodata.spec import SpatialSpec, SpecKeys, compute_bounds_latlon, format_resolution


@dataclass(slots=True)
class RegistryClassInfo:
    object_type: str | None = None
    call_dependency_names: list[str] = field(default_factory=list)
    inheritance_dependency_names: list[str] = field(default_factory=list)
    stored_source_hash: str | None = None
    stored_dependency_tree_hash: str | None = None
    source_path: str | None = None
    graph_path: str | None = None
    registry_path: str | None = None
    tree_path: str | None = None


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
    def from_spec(cls, spec: SpatialSpec) -> SpecInfo:
        """Build a SpecInfo from a SpatialSpec."""
        try:
            bounds = spec.bounds
            bounds_list = [bounds.left, bounds.bottom, bounds.right, bounds.top]
            bounds_str = str(bounds_list)
        except ValueError:
            bounds_list = None
            bounds_str = None
        try:
            resolution = list(spec.resolution)
        except ValueError:
            resolution = None
        crs_str = spec.crs.to_string()
        return cls(
            crs=crs_str,
            resolution=format_resolution(resolution, spec.crs),
            shape=None if spec.shape in ('', None) else str(spec.shape),
            bounds=bounds_str,
            bounds_latlon=compute_bounds_latlon(bounds_list, spec.crs),
        )


@dataclass(slots=True)
class ParamRow:
    path: str
    key_group: str
    final_key: str
    value_text: str  # plain text, unescaped — HTML escaping is the frontend's job
    value_type: str
    search_blob: str
    depth: int


@dataclass(slots=True)
class LinkedEntry:
    """A Data/TrackedObject reference embedded in another entry's params."""

    param_name: str
    class_name: str
    state_hash: str | None
    params_summary: dict[str, str]  # plain-text key→value pairs


@dataclass
class EntryInfo:
    # Fields set during _process_params_path (always present)
    class_name: str
    object_type: str | None
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
    primary_file: FileRef | None = None
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
            'params': self.params,
            SpecKeys.SPEC: dataclasses.asdict(self.spec),
            'rows': [dataclasses.asdict(r) for r in self.rows],
            'linked_entries': [dataclasses.asdict(le) for le in self.linked_entries],
            'co_output_hashes': self.co_output_hashes,
            'primary_file': dataclasses.asdict(self.primary_file) if self.primary_file else None,
            'warnings': list(self.warnings),
            'error': self.error,
            'format_version': self.format_version,
            'dep_hash': self.dep_hash,
            'dep_hash_stale': self.dep_hash_stale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EntryInfo:
        spec_d = data.get(SpecKeys.SPEC, {})
        pf_d = data.get('primary_file')
        return cls(
            record_id=data.get('record_id', ''),
            class_name=data['class_name'],
            object_type=data.get('object_type'),
            params_path=data['params_path'],
            spec_path=data.get('spec_path'),
            state_hash_path=data.get('state_hash_path'),
            execution_graph_path=data.get('execution_graph_path'),
            state_hash=data.get('state_hash'),
            instance_hash=data.get('instance_hash'),
            params=data.get('params', {}),
            spec=SpecInfo(**spec_d) if spec_d else SpecInfo(),
            rows=[ParamRow(**r) for r in data.get('rows', [])],
            linked_entries=[LinkedEntry(**le) for le in data.get('linked_entries', [])],
            co_output_hashes=data.get('co_output_hashes', []),
            primary_file=FileRef(**pf_d) if pf_d else None,
            warnings=data.get('warnings', []),
            error=data.get('error'),
            # accept old cache blobs that stored format_version_stale bool
            format_version=data.get(
                'format_version',
                FORMAT_VERSION if not data.get('format_version_stale') else FORMAT_VERSION - 1,
            ),
            dep_hash=data.get('dep_hash'),
            dep_hash_stale=data.get('dep_hash_stale', False),
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
    loaded: bool
    call_dependency_names: list[str] = field(default_factory=list)
    inheritance_dependency_names: list[str] = field(default_factory=list)
    class_source_path: str | None = None
    class_graph_path: str | None = None
    class_registry_path: str | None = None
    class_tree_path: str | None = None
    source_stale: bool = False  # live source hash ≠ stored source hash
    deps_stale: bool = False  # live dependency_tree_hash ≠ stored dependency_tree_hash
