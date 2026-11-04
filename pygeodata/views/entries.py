import json
from pathlib import Path

import msgspec

from pygeodata._version import FORMAT_VERSION
from pygeodata.catalog import store
from pygeodata.catalog.display import find_output_files, param_rows
from pygeodata.catalog.types import CatalogEntry, FileRole
from pygeodata.enums import Staleness
from pygeodata.paths import CachePathConstructor
from pygeodata.records import EntryRecord, SpecRecord
from pygeodata.registries.registry import EntryRegistry
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.staleness import dependency_staleness
from pygeodata.views.types import EntryInfo, spec_info

# ---------------------------------------------------------------------------
# Per-entry display enrichment
# ---------------------------------------------------------------------------


def _enrich_params_path(params_path: Path, record: EntryRecord | None = None) -> EntryInfo:
    """Read the display-layer files for one entry.

    Returns a partial EntryInfo with all display fields populated but
    assembly-stage fields (record_id, dep_hash_stale, co_outputs) left
    at their defaults — those are filled by discover_entries.

    ``record`` is the EntryRegistry record for this entry, which discover_entries
    already holds in memory; passing it avoids re-reading meta.json per entry.
    When omitted the record is read from disk, so standalone callers still work.
    """
    resolver = CachePathConstructor.from_path(params_path)
    entry_dir = resolver.directory.resolve()

    params = json.loads(params_path.read_text(encoding='utf-8')) if params_path.exists() else {}
    rows, linked_entries = param_rows(params)

    warnings: list[str] = []
    if record is None:
        try:
            record = EntryRecord.from_file(resolver.state_hash_path)
        except (OSError, msgspec.DecodeError):
            warnings.append('meta.json missing or unreadable')
    try:
        spec = SpecRecord.from_file(resolver.spec_path)
    except (OSError, msgspec.DecodeError):
        spec = None

    class_name = record.class_name if record else resolver.directory.name
    object_type = record.object_type if record else None
    files = find_output_files(entry_dir, record.output_ext if record else None)

    return EntryInfo(
        class_name=class_name,
        object_type=object_type,
        entry_dir=str(entry_dir),
        params_path=str(params_path.resolve()),
        spec_path=str(resolver.spec_path) if resolver.spec_path.exists() else None,
        state_hash_path=str(resolver.state_hash_path) if resolver.state_hash_path.exists() else None,
        execution_graph_path=str(resolver.execution_graph_path) if resolver.execution_graph_path.exists() else None,
        state_hash=record.state_hash if record else None,
        instance_hash=record.instance_hash if record else None,
        params_hash=record.params_hash if record else None,
        spec_hash=record.spec_hash if record else None,
        params=params,
        spec=spec_info(spec),
        rows=rows,
        linked_entries=linked_entries,
        co_output_hashes=record.co_output_hashes if record else [],
        primary_file=next((f for f in files if f.role is FileRole.PRIMARY), None),
        sidecar_files=[f for f in files if f.role is FileRole.SIDECAR],
        warnings=warnings,
        format_version=record.format_version if record else FORMAT_VERSION,
        dep_hash=record.dependency_tree_hash if record else None,
    )


def _entry_info_from_catalog(entry: CatalogEntry, spec: SpecRecord | None) -> EntryInfo:
    rec = entry.record
    resolver = CachePathConstructor(store.entry_dir(entry))
    return EntryInfo(
        class_name=rec.class_name,
        object_type=rec.object_type,
        entry_dir=str(resolver.directory),
        params_path=str(resolver.params_path),
        spec_path=str(resolver.spec_path),
        state_hash_path=str(resolver.state_hash_path),
        execution_graph_path=str(resolver.execution_graph_path),
        state_hash=rec.state_hash,
        instance_hash=rec.instance_hash,
        params_hash=rec.params_hash,
        spec_hash=rec.spec_hash,
        params={},
        spec=spec_info(spec),
        rows=entry.rows,
        linked_entries=entry.linked_entries,
        co_output_hashes=rec.co_output_hashes,
        primary_file=entry.primary_file,
        sidecar_files=entry.sidecar_files,
        format_version=rec.format_version,
        dep_hash=rec.dependency_tree_hash,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def discover_entries(
    progress: dict | None = None,
    version_registry: VersionRegistry | None = None,
) -> tuple[dict[str, EntryInfo], EntryRegistry, dict]:
    """Assemble the browser-enriched view of all cache entries.

    Display data comes from the catalog when it has the entry — import reads no
    per-entry files. Records missing from the catalog (fresh cache, post-wipe) fall
    back to per-entry disk enrichment, which also repopulates the catalog via the
    write path on the next render or via `catalog build`.
    """
    if version_registry is None:
        version_registry = VersionRegistry()

    entry_registry = EntryRegistry()
    catalog = store.read_catalog()

    records = sorted(entry_registry.records.values(), key=lambda r: r.state_hash)
    if progress is not None:
        progress['done'] = 0
        progress['total'] = len(records)

    entries: dict[str, EntryInfo] = {}
    diagnostics = dict(entry_registry.diagnostics())

    for rec in records:
        indexed = catalog.entries.get(rec.state_hash) if catalog is not None else None
        if indexed is not None:
            entry = _entry_info_from_catalog(indexed, catalog.specs.get(rec.spec_hash))
        else:
            entry_dir = entry_registry.entry_dir(rec.state_hash)
            entry = _enrich_params_path(CachePathConstructor(entry_dir).params_path, rec)
        entry.record_id = rec.state_hash
        entry.dep_hash_stale = (
            dependency_staleness(entry.class_name, entry.dep_hash, version_registry) is Staleness.STALE
        )
        entries[entry.record_id] = entry
        if progress is not None:
            progress['done'] += 1

    # Second pass — resolve co_output_hashes to EntryInfo references
    for entry in entries.values():
        entry.co_outputs = [entries[h] for h in entry.co_output_hashes if h in entries]

    diagnostics['created_entries'] = len(entries)
    return entries, entry_registry, diagnostics
