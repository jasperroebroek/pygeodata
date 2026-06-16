from pygeodata.config import JSONKeys
from pygeodata.formatting.json import format_json
from pygeodata.registry import SourceRegistry
from pygeodata.registry_browser.filters import Filter, entry_matches_filters, matching_rows, parse_filters
from pygeodata.registry_browser.models import ClassInfo, EntryInfo, FileRef, LinkedEntry
from pygeodata.registry_browser.state import AppState
from pygeodata.spec import SpecKeys
from pygeodata.versioning import VersionRegistry

# ---------------------------------------------------------------------------
# Small serialisers — one responsibility each
# ---------------------------------------------------------------------------


def _row_payload(row) -> dict:
    return {
        'group': row.key_group,
        'parameter': row.final_key,
        'value': row.value_text,
        'value_type': row.value_type,
        'path': row.path,
        'depth': row.depth,
    }


def _file_payload(file_ref: FileRef) -> dict:
    return {
        'label': file_ref.label,
        'path': file_ref.path,
        'kind': file_ref.kind,
    }


def _linked_entry_payload(le: LinkedEntry) -> dict:
    return {
        'param_name': le.param_name,
        JSONKeys.CLASS_NAME: le.class_name,
        JSONKeys.STATE_HASH: le.state_hash,
        'params_summary': le.params_summary,
    }


def _spec_payload(spec) -> dict:
    return {
        SpecKeys.CRS: spec.crs,
        SpecKeys.RESOLUTION: spec.resolution,
        SpecKeys.SHAPE: spec.shape,
        SpecKeys.BOUNDS: spec.bounds,
        SpecKeys.BOUNDS_LATLON: spec.bounds_latlon,
    }


# ---------------------------------------------------------------------------
# Visibility / filtering
# ---------------------------------------------------------------------------


def _entry_matches_spec_filters(entry: EntryInfo, spec_filters: dict) -> bool:
    for dim in (SpecKeys.CRS, SpecKeys.RESOLUTION, SpecKeys.SHAPE):
        selected = spec_filters.get(dim)
        if not selected:
            continue
        val = getattr(entry.spec, dim)
        if isinstance(selected, list):
            if selected and val not in selected:
                return False
        elif val != selected:
            return False

    selected_bounds = spec_filters.get(SpecKeys.BOUNDS)
    if selected_bounds:
        entry_bounds = str(list(entry.spec.bounds_latlon)) if entry.spec.bounds_latlon else None
        if isinstance(selected_bounds, list):
            if selected_bounds and entry_bounds not in selected_bounds:
                return False
        elif entry_bounds != selected_bounds:
            return False

    return True


def _entry_is_visible(
    *,
    class_name: str,
    entry: EntryInfo,
    selected_classes: list[str],
    kind_filter: str,
    spec_filters: dict,
    filters: list[Filter],
    logic_mode: str,
    hide_stale: bool = False,
    snapshot_filter: list[str] | None = None,
) -> bool:
    if hide_stale and (entry.dep_hash_stale or entry.format_version_stale):
        return False
    if snapshot_filter and entry.dep_hash not in snapshot_filter:  # snapshot_filter is a resolved set of dep_hashes
        return False
    if selected_classes and class_name not in selected_classes:
        return False
    if kind_filter != 'all' and (entry.object_type or '').lower() != kind_filter and class_name not in selected_classes:
        return False
    if not _entry_matches_spec_filters(entry, spec_filters):
        return False
    return entry_matches_filters(class_name, entry, filters, logic_mode)


# ---------------------------------------------------------------------------
# Query helpers — build filtered views of state
# ---------------------------------------------------------------------------


def _sidebar_counts(
    state: AppState,
    *,
    kind_filter: str,
    spec_filters: dict,
    filters: list[Filter],
    logic_mode: str,
    hide_stale: bool = False,
    snapshot_filter: list[str] | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name, class_info in state.classes.items():
        if kind_filter != 'all' and (class_info.object_type or '').lower() != kind_filter:
            continue
        record_ids = state.get_state_hashes(class_name)
        n = sum(
            1
            for rid in record_ids
            if _entry_is_visible(
                class_name=class_name,
                entry=state.entries[rid],
                selected_classes=[],
                kind_filter=kind_filter,
                spec_filters=spec_filters,
                filters=filters,
                logic_mode=logic_mode,
                hide_stale=hide_stale,
                snapshot_filter=snapshot_filter,
            )
        )
        if n:
            counts[class_name] = n
    return counts


def _build_visible_groups(
    state: AppState,
    *,
    selected_classes: list[str],
    kind_filter: str,
    spec_filters: dict,
    filters: list[Filter],
    logic_mode: str,
    hide_stale: bool = False,
    snapshot_filter: list[str] | None = None,
) -> list[tuple[str, list[EntryInfo]]]:
    visible_groups: list[tuple[str, list[EntryInfo]]] = []

    for class_name, class_info in sorted(state.classes.items()):
        if (
            kind_filter != 'all'
            and (class_info.object_type or '').lower() != kind_filter
            and class_name not in selected_classes
        ):
            continue

        record_ids = state.get_state_hashes(class_name)

        visible_entries = [
            state.entries[rid]
            for rid in record_ids
            if _entry_is_visible(
                class_name=class_name,
                entry=state.entries[rid],
                selected_classes=selected_classes,
                kind_filter=kind_filter,
                spec_filters=spec_filters,
                filters=filters,
                logic_mode=logic_mode,
                hide_stale=hide_stale,
                snapshot_filter=snapshot_filter,
            )
        ]

        if visible_entries:
            visible_groups.append((class_name, visible_entries))

    return visible_groups


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _build_class_cards(
    state: AppState,
    *,
    sidebar_counts: dict[str, int],
    selected_classes: list[str],
    snapshot_dep_hashes: set[str] | None = None,
) -> list[dict]:
    # When a snapshot filter is active, derive stale status from the visible entries:
    # a class is stale if any of its snapshot entries have dep_hash_stale=True.
    snapshot_stale_classes: set[str] = set()
    if snapshot_dep_hashes is not None:
        for entry in state.entries.values():
            if entry.dep_hash in snapshot_dep_hashes and entry.dep_hash_stale:
                snapshot_stale_classes.add(entry.class_name)

    class_cards = []
    for class_name, class_info in sorted(state.classes.items()):
        record_ids = state.get_state_hashes(class_name)
        format_version_stale = any(
            state.entries[rid].format_version_stale for rid in record_ids if rid in state.entries
        )
        if snapshot_dep_hashes is not None:
            source_stale = class_name in snapshot_stale_classes
            deps_stale = False
        else:
            source_stale = class_info.source_stale
            deps_stale = class_info.deps_stale
        class_cards.append(
            {
                'class_name': class_name,
                'object_type': class_info.object_type,
                'loaded': class_info.loaded,
                'call_dependency_names': class_info.call_dependency_names,
                'inheritance_dependency_names': class_info.inheritance_dependency_names,
                'total_record_count': len(record_ids),
                'visible_record_count': sidebar_counts.get(class_name, 0),
                'selected': class_name in selected_classes,
                'source_stale': source_stale,
                'deps_stale': deps_stale,
                'format_version_stale': format_version_stale,
            },
        )
    return class_cards


def _build_table_rows(
    *,
    visible_groups: list[tuple[str, list[EntryInfo]]],
    classes: dict[str, ClassInfo],
    selected_entry: str | None,
    filters: list[Filter],
    logic_mode: str,
    row_display: str,
) -> list[dict]:
    table_rows: list[dict] = []

    for class_name, group_entries in visible_groups:
        ci = classes.get(class_name)
        for entry in group_entries:
            table_rows.append(
                {
                    'row_type': 'header',
                    'class_name': class_name,
                    'record_id': entry.record_id,
                    'object_type': entry.object_type,
                    'spec': _spec_payload(entry.spec),
                    'warning_count': len(entry.warnings),
                    'warnings': entry.warnings,
                    'error': entry.error or '',
                    'focused': entry.record_id == selected_entry,
                    'source_stale': ci.source_stale if ci else False,
                    'dep_hash_stale': entry.dep_hash_stale,
                    'format_version_stale': entry.format_version_stale,
                    'dep_hash': entry.dep_hash,
                },
            )

            if row_display == 'all':
                detail_rows = entry.rows
            elif row_display == 'selected':
                detail_rows = matching_rows(entry, filters, logic_mode)
            else:
                detail_rows = []

            for row in detail_rows:
                table_rows.append(
                    {
                        'row_type': 'detail',
                        'class_name': class_name,
                        'record_id': entry.record_id,
                        **_row_payload(row),
                    },
                )

    return table_rows


def _class_detail_payload(class_info: ClassInfo) -> dict:
    return {
        'class_name': class_info.class_name,
        'object_type': class_info.object_type,
        'loaded': class_info.loaded,
        'call_dependency_names': class_info.call_dependency_names,
        'inheritance_dependency_names': class_info.inheritance_dependency_names,
        'source_available': bool(class_info.class_source_path),
        'graph_available': bool(class_info.class_graph_path),
        'class_source_path': class_info.class_source_path,
        'class_graph_path': class_info.class_graph_path,
        'class_registry_path': class_info.class_registry_path,
        'class_tree_path': class_info.class_tree_path,
        'source_stale': class_info.source_stale,
        'deps_stale': class_info.deps_stale,
    }


def _co_output_diff_rows(co_entry: EntryInfo, main_entry: EntryInfo) -> list[dict]:
    main_by_path = {r.path: r.value_text for r in main_entry.rows}
    return [_row_payload(r) for r in co_entry.rows if main_by_path.get(r.path) != r.value_text]


def _co_output_payload(entry: EntryInfo, main_entry: EntryInfo) -> dict:
    return {
        'record_id': entry.record_id,
        JSONKeys.CLASS_NAME: entry.class_name,
        JSONKeys.OBJECT_TYPE: entry.object_type,
        JSONKeys.STATE_HASH: entry.state_hash,
        'primary_file': _file_payload(entry.primary_file) if entry.primary_file else None,
        'rows': [_row_payload(r) for r in entry.rows],
        'diff_rows': _co_output_diff_rows(entry, main_entry),
    }


def _same_instance_run_payload(sibling: EntryInfo) -> dict:
    return {
        'record_id': sibling.record_id,
        JSONKeys.CLASS_NAME: sibling.class_name,
        JSONKeys.STATE_HASH: sibling.state_hash,
        'primary_file': _file_payload(sibling.primary_file) if sibling.primary_file else None,
        SpecKeys.SPEC: _spec_payload(sibling.spec),
    }


def _entry_detail_payload(entry: EntryInfo, same_instance_runs: list[EntryInfo] | None = None) -> dict:
    return {
        'record_id': entry.record_id,
        JSONKeys.STATE_HASH: entry.state_hash,
        JSONKeys.INSTANCE_HASH: entry.instance_hash,
        JSONKeys.CLASS_NAME: entry.class_name,
        JSONKeys.OBJECT_TYPE: entry.object_type,
        'warnings': entry.warnings,
        'dep_hash_stale': entry.dep_hash_stale,
        'format_version_stale': entry.format_version_stale,
        'dep_hash': entry.dep_hash,
        'params_path': entry.params_path,
        'state_hash_path': entry.state_hash_path,
        'execution_graph_path': entry.execution_graph_path,
        SpecKeys.SPEC: _spec_payload(entry.spec),
        'primary_file': _file_payload(entry.primary_file) if entry.primary_file else None,
        'figure_previews': [entry.primary_file.path]
        if entry.primary_file and entry.primary_file.kind == 'image'
        else [],
        JSONKeys.CO_OUTPUTS: [_co_output_payload(e, entry) for e in entry.co_outputs],
        'same_instance_runs': [_same_instance_run_payload(s) for s in (same_instance_runs or [])],
        'rows': [_row_payload(r) for r in entry.rows],
        'params_tree': format_json(entry.params),
        'linked_entries': [_linked_entry_payload(le) for le in entry.linked_entries],
    }


def _build_instance_hash_index(entries: dict[str, 'EntryInfo']) -> dict[str, list['EntryInfo']]:
    index: dict[str, list[EntryInfo]] = {}
    for entry in entries.values():
        if entry.instance_hash:
            index.setdefault(entry.instance_hash, []).append(entry)
    return index


def version_groups_payload(vreg: VersionRegistry) -> list[dict]:
    """Serialize VersionRegistry.version_groups for the browser API.

    Adds cutoff_mtime / cutoff_exclusive (positional, derived from adjacent groups)
    and expands class_names to include newly-appearing classes so the JS can show
    them as 'added' in the version change summary.
    """
    groups = vreg.version_groups
    src = vreg.source_registry

    first_reg: dict[str, str] = {}
    for class_name in src.class_names:
        states = src.get_states(class_name)
        if states:
            first_reg[class_name] = states[0].registered_at

    added_by_group: dict[str, str] = {}
    initial_mtime = groups[-1].mtime if groups else ''
    non_initial_asc = list(reversed(groups[:-1]))
    for cn, reg in first_reg.items():
        if reg <= initial_mtime:
            continue
        for j, g in enumerate(non_initial_asc):
            lower = initial_mtime if j == 0 else non_initial_asc[j - 1].mtime
            upper = non_initial_asc[j + 1].mtime if j + 1 < len(non_initial_asc) else None
            if reg > lower and (upper is None or reg < upper):
                added_by_group[cn] = g.mtime
                break

    result = []
    for i, vi in enumerate(groups):
        cutoff_mtime = 'now' if i == 0 else groups[i - 1].mtime
        cutoff_exclusive = i > 0
        is_initial = i == len(groups) - 1
        added_classes = [] if is_initial else sorted(
            cn for cn, gm in added_by_group.items()
            if gm == vi.mtime and cn not in {e.class_name for e in vi.events}
        )
        result.append(
            {
                'mtime': vi.mtime,
                'label': vi.label,
                'class_names': sorted(set(vi.class_names) | set(added_classes)),
                'cutoff_mtime': cutoff_mtime,
                'cutoff_exclusive': cutoff_exclusive,
            },
        )
    return result


def _class_version_history(class_name: str, src: SourceRegistry) -> list[dict]:
    """Return code versions for a class sorted oldest-first.

    Each entry: {source_hash, mtime, is_version_change}.
    """
    states = src.get_states(class_name)
    return [
        {'source_hash': s.source_hash, 'mtime': s.registered_at, 'is_version_change': src.is_version_change(s)}
        for s in states
    ]


def _build_detail_payload(
    *,
    state: AppState,
    selected_entry_info: EntryInfo | None,
    selected_classes: list[str],
    vreg: VersionRegistry,
) -> dict | None:
    src = vreg.source_registry
    if selected_entry_info is not None:
        class_info = state.classes.get(selected_entry_info.class_name)
        if class_info is None:
            return None
        siblings: list[EntryInfo] = []
        if selected_entry_info.instance_hash:
            index = _build_instance_hash_index(state.entries)
            siblings = [
                e
                for e in index.get(selected_entry_info.instance_hash, [])
                if e.record_id != selected_entry_info.record_id
            ]
        entry_version_mtime = vreg.version_mtime_for_dep_hash(selected_entry_info.dep_hash) if selected_entry_info.dep_hash else None
        return {
            **_class_detail_payload(class_info),
            'code_versions': _class_version_history(selected_entry_info.class_name, src),
            'entry_version_mtime': entry_version_mtime,
            'selected_entry': _entry_detail_payload(selected_entry_info, same_instance_runs=siblings),
        }

    if len(selected_classes) == 1:
        class_info = state.classes.get(selected_classes[0])
        if class_info is not None:
            return {
                **_class_detail_payload(class_info),
                'code_versions': _class_version_history(selected_classes[0], src),
                'entry_version_mtime': None,
                'selected_entry': None,
            }

    return None


# ---------------------------------------------------------------------------
# Dep-hash (snapshot) options
# ---------------------------------------------------------------------------


def _dep_hashes_for_version(state: 'AppState', version_mtime: str, vreg: VersionRegistry) -> set[str]:
    """Return dep_hashes whose snapshot belongs to the selected version group."""
    return {dh for dh, identity in vreg.dep_hash_to_mtime.items() if identity == version_mtime}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def build_browser_payload(
    state: AppState,
    *,
    selected_classes: list[str],
    selected_entry: str | None,
    kind_filter: str,
    spec_filters: dict,
    filters: list[dict],
    logic_mode: str,
    row_display: str,
    hide_stale: bool = False,
    version_filter: str | None = None,  # mtime of the selected version, or None for all
) -> dict:
    vreg = state.version_registry
    parsed_filters = parse_filters(filters)
    dep_hash_set = _dep_hashes_for_version(state, version_filter, vreg) if version_filter else None

    visible_groups = _build_visible_groups(
        state,
        selected_classes=selected_classes,
        kind_filter=kind_filter,
        spec_filters=spec_filters,
        filters=parsed_filters,
        logic_mode=logic_mode,
        hide_stale=hide_stale,
        snapshot_filter=dep_hash_set,
    )

    sidebar_counts = _sidebar_counts(
        state,
        kind_filter=kind_filter,
        spec_filters=spec_filters,
        filters=parsed_filters,
        logic_mode=logic_mode,
        hide_stale=hide_stale,
        snapshot_filter=dep_hash_set,
    )

    # Validate selected_entry is still visible
    if selected_entry and selected_entry not in state.entries:
        selected_entry = None

    visible_entry_ids = [e.record_id for _, entries in visible_groups for e in entries]
    selected_entry_info = state.entries.get(selected_entry) if selected_entry else None

    stale_hidden = sum(1 for e in state.entries.values() if e.dep_hash_stale) if hide_stale else 0
    diagnostics = {**state.diagnostics, 'stale_hidden': stale_hidden}

    return {
        'selected_classes': selected_classes,
        'selected_entry': selected_entry,
        'class_cards': _build_class_cards(
            state, sidebar_counts=sidebar_counts, selected_classes=selected_classes, snapshot_dep_hashes=dep_hash_set
        ),
        'table_rows': _build_table_rows(
            visible_groups=visible_groups,
            classes=state.classes,
            selected_entry=selected_entry,
            filters=parsed_filters,
            logic_mode=logic_mode,
            row_display=row_display,
        ),
        'detail': _build_detail_payload(
            state=state,
            selected_entry_info=selected_entry_info,
            selected_classes=selected_classes,
            vreg=vreg,
        ),
        'diagnostics': diagnostics,
        'spec_options': state.spec_options,
        'version_options': version_groups_payload(vreg),
        'counts': {
            'classes': len(state.classes),
            'classes_loaded': sum(1 for c in state.classes.values() if c.loaded),
            'entries': len(state.entries),
            'visible_classes': len(visible_groups),
            'visible_entries': len(visible_entry_ids),
        },
        'visible_entry_ids': visible_entry_ids,
    }
