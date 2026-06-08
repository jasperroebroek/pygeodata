from pygeodata.config import JSONKeys
from pygeodata.formatting.json import format_json
from pygeodata.registry_browser.filters import Filter, entry_matches_filters, matching_rows, parse_filters
from pygeodata.registry_browser.models import ClassInfo, EntryInfo, FileRef, LinkedEntry
from pygeodata.registry_browser.state import AppState
from pygeodata.types import SpecKeys

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
) -> bool:
    if selected_classes and class_name not in selected_classes:
        return False
    if (
        kind_filter != 'all'
        and (entry.object_type or '').lower() != kind_filter
        and class_name not in selected_classes
    ):
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
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_name, class_info in state.classes.items():
        if kind_filter != 'all' and (class_info.object_type or '').lower() != kind_filter:
            continue
        group = state.groups.get(class_name)
        record_ids = group.record_ids if group else []
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
) -> list[tuple[str, list[EntryInfo]]]:
    visible_groups: list[tuple[str, list[EntryInfo]]] = []

    for class_name, class_info in sorted(state.classes.items()):
        if (
            kind_filter != 'all'
            and (class_info.object_type or '').lower() != kind_filter
            and class_name not in selected_classes
        ):
            continue

        group = state.groups.get(class_name)
        record_ids = group.record_ids if group else []

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
) -> list[dict]:
    class_cards = []
    for class_name, class_info in sorted(state.classes.items()):
        group = state.groups.get(class_name)
        class_cards.append(
            {
                'class_name': class_name,
                'object_type': class_info.object_type,
                'loaded': class_info.loaded,
                'call_dependency_names': class_info.call_dependency_names,
                'inheritance_dependency_names': class_info.inheritance_dependency_names,
                'total_record_count': len(group.record_ids) if group else 0,
                'visible_record_count': sidebar_counts.get(class_name, 0),
                'selected': class_name in selected_classes,
                'source_stale': class_info.source_stale,
                'deps_stale': class_info.deps_stale,
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


def _build_detail_payload(
    *,
    state: AppState,
    selected_entry_info: EntryInfo | None,
    selected_classes: list[str],
) -> dict | None:
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
        return {
            **_class_detail_payload(class_info),
            'selected_entry': _entry_detail_payload(selected_entry_info, same_instance_runs=siblings),
        }

    if len(selected_classes) == 1:
        class_info = state.classes.get(selected_classes[0])
        if class_info is not None:
            return {
                **_class_detail_payload(class_info),
                'selected_entry': None,
            }

    return None


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
) -> dict:
    parsed_filters = parse_filters(filters)

    visible_groups = _build_visible_groups(
        state,
        selected_classes=selected_classes,
        kind_filter=kind_filter,
        spec_filters=spec_filters,
        filters=parsed_filters,
        logic_mode=logic_mode,
    )

    sidebar_counts = _sidebar_counts(
        state,
        kind_filter=kind_filter,
        spec_filters=spec_filters,
        filters=parsed_filters,
        logic_mode=logic_mode,
    )

    # Validate selected_entry is still visible
    if selected_entry and selected_entry not in state.entries:
        selected_entry = None

    visible_entry_ids = [e.record_id for _, entries in visible_groups for e in entries]
    selected_entry_info = state.entries.get(selected_entry) if selected_entry else None

    return {
        'selected_classes': selected_classes,
        'selected_entry': selected_entry,
        'class_cards': _build_class_cards(state, sidebar_counts=sidebar_counts, selected_classes=selected_classes),
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
        ),
        'diagnostics': state.diagnostics,
        'spec_options': state.spec_options,
        'counts': {
            'classes': len(state.classes),
            'classes_loaded': sum(1 for c in state.classes.values() if c.loaded),
            'entries': len(state.entries),
            'visible_classes': len(visible_groups),
            'visible_entries': len(visible_entry_ids),
        },
        'visible_entry_ids': visible_entry_ids,
    }
