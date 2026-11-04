from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec

from pygeodata._version import FORMAT_VERSION
from pygeodata.catalog.store import index_entry
from pygeodata.fileio import read_format_version, write_json_atomic
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CachePathConstructor
from pygeodata.records import EntryRecord

if TYPE_CHECKING:
    from pathlib import Path

    from pygeodata.artifact import Artifact
    from pygeodata.spec import SpatialSpec


def write_entry(artifact: Artifact, spec: SpatialSpec, co_outputs: tuple[str, ...] = ()) -> None:
    """Write ``parameters.json``, ``spec.json`` and ``meta.json`` for this entry and index it in the catalog.

    ``co_outputs`` are the state hashes of artifacts produced in the same ``_process`` call.
    """
    spec = artifact.resolve_spec(spec)
    state_hash = artifact.get_state_hash(spec)
    paths = CachePathConstructor.from_state_hash(state_hash, artifact.get_cache_root())
    paths.mkdir()
    params_json = artifact.get_params_as_json(spec=spec)
    write_json_atomic(paths.params_path, params_json, indent=4)
    write_json_atomic(paths.spec_path, spec.to_dict(), indent=4)
    record = EntryRecord(
        class_name=artifact.get_class_name(),
        source_hash=calculate_cls_source_hash(type(artifact)),
        dependency_tree_hash=artifact.get_dependency_tree_hash(),
        instance_hash=artifact.get_instance_hash(),
        params_hash=artifact.get_params_hash(),
        spec_hash=spec.get_hash(),
        state_hash=state_hash,
        object_type=artifact.object_type.get_class_name(),
        co_output_hashes=list(co_outputs),
        format_version=FORMAT_VERSION,
    )
    record.dump(paths.state_hash_path)
    index_entry(paths.directory, record, spec.to_record(), params_json)


def _read_current_record(meta_path: Path) -> EntryRecord | None:
    """The entry's record, or None when it is absent, unreadable or in an older format."""
    if read_format_version(meta_path) != FORMAT_VERSION:
        return None
    try:
        return EntryRecord.from_file(meta_path)
    except (OSError, msgspec.DecodeError):
        return None


def read_state_hash(artifact: Artifact, spec: SpatialSpec) -> str | None:
    record = _read_current_record(artifact.resolve_cache_paths(spec).state_hash_path)
    return record.state_hash if record is not None else None


def is_cache_valid(artifact: Artifact, spec: SpatialSpec) -> bool:
    """``meta.json`` exists, is in the current format and records the artifact's current state hash."""
    record = _read_current_record(artifact.resolve_cache_paths(spec).state_hash_path)
    return record is not None and record.state_hash == artifact.get_state_hash(spec)
