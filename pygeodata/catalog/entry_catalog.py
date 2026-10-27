import ast
import json
from functools import cache
from pathlib import Path

from affine import Affine
from sqlalchemy import select
from sqlalchemy.orm import Session

from pygeodata.catalog import models, store
from pygeodata.catalog.display import find_primary_and_sidecar_files
from pygeodata.catalog.params_index import flatten_params
from pygeodata.catalog.types import (
    EntryInfo,
    FileRef,
    FileRole,
    LinkedEntry,
    ParamRow,
    ParamRowType,
    SpecInfo,
)
from pygeodata.config import FORMAT_VERSION, catalog_db_path, get_config
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.registry import EntryRegistry
from pygeodata.registries.registry_types import EntryRecord
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.spec import SpatialSpec, _crs_from_string
from pygeodata.tracked_object import TrackedObject

# ---------------------------------------------------------------------------
# Per-entry display enrichment
# ---------------------------------------------------------------------------


def _enrich_params_path(params_path: Path, record: EntryRecord | None = None) -> EntryInfo:
    """Read the display-layer files for one entry. Never raises.

    Returns a partial EntryInfo with all display fields populated but
    assembly-stage fields (record_id, dep_hash_stale, co_outputs) left
    at their defaults — those are filled by discover_entries.

    ``record`` is the EntryRegistry record for this entry, which discover_entries
    already holds in memory; passing it avoids re-reading meta.json per entry.
    When omitted the record is read from disk, so standalone callers still work.
    """
    params_path_str = str(params_path.resolve())
    resolver = CachePathConstructor.from_path(params_path)

    params = json.loads(params_path.read_text(encoding='utf-8')) if params_path.exists() else {}
    spec = json.loads(resolver.spec_path.read_text(encoding='utf-8')) if resolver.spec_path.exists() else {}

    linked_entries: list[LinkedEntry] = []
    rows = flatten_params(params, linked_entries=linked_entries)

    if record is None and resolver.state_hash_path.exists():
        record = EntryRecord.from_file(resolver.state_hash_path)
    class_name = (record.class_name if record else None) or resolver.directory.name
    state_hash = record.state_hash if record else None
    instance_hash = record.instance_hash if record else None
    params_hash = record.params_hash if record else None
    spec_hash = record.spec_hash if record else None
    stored_dep_hash = record.dependency_tree_hash if record else None

    co_output_hashes = record.co_output_hashes if record else []
    format_version = record.format_version if record else FORMAT_VERSION

    object_type = record.object_type if record else None

    primary_file, sidecar_files = find_primary_and_sidecar_files(resolver.directory, class_name, object_type)

    warnings: list[str] = []
    if not state_hash:
        warnings.append('Missing state hash in meta.json')

    spatial_spec = SpatialSpec.from_dict(spec) if spec else None

    return EntryInfo(
        class_name=class_name,
        object_type=object_type,
        params_path=params_path_str,
        spec_path=str(resolver.spec_path) if resolver.spec_path.exists() else None,
        state_hash_path=str(resolver.state_hash_path) if resolver.state_hash_path.exists() else None,
        execution_graph_path=str(resolver.execution_graph_path) if resolver.execution_graph_path.exists() else None,
        state_hash=state_hash,
        instance_hash=instance_hash,
        params_hash=params_hash,
        spec_hash=spec_hash,
        params=params,
        spec=SpecInfo.from_spec(spatial_spec) if spatial_spec else SpecInfo(),
        rows=rows,
        linked_entries=linked_entries,
        co_output_hashes=co_output_hashes,
        primary_file=primary_file,
        sidecar_files=sidecar_files,
        warnings=warnings,
        format_version=format_version,
        dep_hash=stored_dep_hash,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _spec_info(crs_hash: str | None, transform_hash: str | None, shape_hash: str | None) -> SpecInfo:
    """Build the spec display from dimension rows, memoized per distinct spec triple.

    The reconstructed SpatialSpec covers bounds, resolution and their display
    strings, so only distinct (crs, transform, shape) triples reproject.
    """
    return _spec_info_cached(crs_hash, transform_hash, shape_hash)


@cache
def _spec_info_cached(crs_hash: str | None, transform_hash: str | None, shape_hash: str | None) -> SpecInfo:
    engine = store.get_engine(str(get_config().path_registry.resolve()))
    with Session(engine) as session:
        crs_row = session.get(models.Crs, crs_hash) if crs_hash is not None else None
        res_row = session.get(models.Resolution, transform_hash) if transform_hash is not None else None
        shape_row = session.get(models.Shape, shape_hash) if shape_hash is not None else None

    if crs_row is None:
        return SpecInfo()
    # Partial specs are common (figures and tables usually carry only a CRS): keep what is there.
    spec = SpatialSpec(
        crs=_crs_from_string(crs_row.crs),
        transform=Affine(res_row.a, res_row.b, res_row.c, res_row.d, res_row.e, res_row.f) if res_row else None,
        shape=ast.literal_eval(shape_row.shape) if shape_row else None,
    )
    return SpecInfo.from_spec(spec)


def _load_display_from_catalog() -> dict[str, dict] | None:
    """Per-state_hash display payload from the catalog; None when the catalog is unusable."""
    if not catalog_db_path().exists():
        return None
    source_root = get_config().path_registry
    base = source_root.resolve().parent
    engine = store.get_engine(str(source_root.resolve()))
    display: dict[str, dict] = {}
    with Session(engine) as session:
        files_by_hash: dict[str, dict[FileRole, list[FileRef]]] = {}
        for f in session.execute(select(models.EntryFile)).scalars():
            ref = FileRef(label=f.label, path=str(base / f.relpath), kind=f.kind)
            files_by_hash.setdefault(f.state_hash, {}).setdefault(f.role, []).append(ref)

        for row in session.execute(select(models.Entry)).scalars():
            entry_dir = store.entry_abs_path(source_root, row.dir_relpath)
            files = files_by_hash.get(row.state_hash, {})
            rows = [ParamRow(**{**r, 'row_type': ParamRowType(r['row_type'])}) for r in (row.rows or [])]
            linked = [LinkedEntry(**le) for le in (row.linked_entries or [])]
            display[row.state_hash] = {
                'entry_dir': entry_dir,
                'spec': _spec_info(row.crs_hash, row.transform_hash, row.shape_hash),
                'rows': rows,
                'linked_entries': linked,
                'primary_file': files.get(FileRole.PRIMARY, [None])[0],
                'sidecar_files': files.get(FileRole.SIDECAR, []),
            }
    return display or None


def discover_entries(
    progress: dict | None = None,
    version_registry: VersionRegistry | None = None,
) -> tuple[dict[str, EntryInfo], EntryRegistry, dict]:
    """Assemble the browser-enriched view of all cache entries.

    Display data comes from the catalog DB when it has entries — import reads no
    per-entry files. Records missing from the catalog (fresh cache, post-wipe) fall
    back to per-entry disk enrichment, which also repopulates the catalog via the
    write path on the next render or via `catalog build`.
    """
    if version_registry is None:
        version_registry = VersionRegistry()

    entry_registry = EntryRegistry()
    display_by_hash = _load_display_from_catalog()

    records = sorted(entry_registry.records.values(), key=lambda r: r.state_hash or '')
    if progress is not None:
        progress['done'] = 0
        progress['total'] = len(records)

    entries: dict[str, EntryInfo] = {}
    diagnostics = dict(entry_registry.diagnostics())
    diagnostics['created_entries'] = 0

    for rec in records:
        if rec.state_hash is None:
            continue
        disp = display_by_hash.get(rec.state_hash) if display_by_hash is not None else None
        entry = (
            _entry_info_from_display(rec, disp)
            if disp is not None
            else _enrich_params_path(CachePathConstructor.from_path(Path(rec.hash_path)).params_path, rec)
        )
        entry.record_id = rec.state_hash
        if entry.dep_hash:
            obj_cls = TrackedObject.find_object_class(entry.class_name)
            if obj_cls is not None:
                entry.dep_hash_stale = obj_cls.get_dependency_tree_hash() != entry.dep_hash
            else:
                entry.dep_hash_stale = version_registry.is_dependency_hash_stale(entry.dep_hash)
        entries[entry.record_id] = entry
        if progress is not None:
            progress['done'] += 1

    # Second pass — resolve co_output_hashes to EntryInfo references
    for entry in entries.values():
        entry.co_outputs = [entries[h] for h in entry.co_output_hashes if h in entries]

    diagnostics['created_entries'] = len(entries)
    return entries, entry_registry, diagnostics


def _entry_info_from_display(rec: EntryRecord, disp: dict) -> EntryInfo:
    entry_dir = Path(disp['entry_dir']).resolve()
    resolver = CachePathConstructor(entry_dir)
    return EntryInfo(
        class_name=rec.class_name or entry_dir.name,
        object_type=rec.object_type,
        params_path=str(resolver.params_path),
        spec_path=str(resolver.spec_path),
        state_hash_path=str(resolver.state_hash_path),
        execution_graph_path=str(resolver.execution_graph_path),
        state_hash=rec.state_hash,
        instance_hash=rec.instance_hash,
        params_hash=rec.params_hash,
        spec_hash=rec.spec_hash,
        params={},
        spec=disp['spec'],
        rows=disp['rows'],
        linked_entries=disp['linked_entries'],
        co_output_hashes=rec.co_output_hashes,
        primary_file=disp['primary_file'],
        sidecar_files=disp['sidecar_files'],
        format_version=rec.format_version or FORMAT_VERSION,
        dep_hash=rec.dependency_tree_hash,
    )
