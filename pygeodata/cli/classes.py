import click

from pygeodata.cli.common import (
    _apply_registry_path,
    _entry_registry,
    _fmt_hash,
    _full_hash_option,
    _import_options,
    _legend,
    _marker,
    _project_source,
    _registry_option,
    _tree_hash_lookup,
)
from pygeodata.enums import ObjectType
from pygeodata.registries.registry import SourceRegistry
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.views.classes import imported_classes, merge_unknown_classes, source_classes

# ---------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------


@click.command('classes')
@_import_options
@click.option(
    '--type',
    'type_filter',
    default=None,
    help='Filter to one object type: Data, Table, Figure, or TrackedObject (case-insensitive).',
)
@click.option('--hide-current', is_flag=True, default=False, help='Omit classes with no staleness.')
@_full_hash_option
@_registry_option
def classes_cmd(
    do_import: bool,
    verbose_import: bool,
    type_filter: str | None,
    hide_current: bool,
    full_hash: bool,
    registry_path: str | None,
) -> None:
    """Show staleness for the project's classes and those with cache entries."""
    _apply_registry_path(registry_path)

    analysis = _project_source(do_import, verbose_import)
    vreg = VersionRegistry()
    ereg = _entry_registry()
    src = SourceRegistry()
    classes = source_classes(analysis) if analysis is not None else imported_classes()
    classes = merge_unknown_classes(classes, ereg, _tree_hash_lookup(analysis), version_registry=vreg, src=src)

    # Apply type filter (case-insensitive exact match against object_type)
    if type_filter:
        tf = type_filter.upper()
        classes = {k: v for k, v in classes.items() if (v.object_type or '').upper() == tf}

    # Apply hide-current filter
    if hide_current:
        classes = {k: v for k, v in classes.items() if v.source_stale or v.deps_stale}

    if not classes:
        click.echo('No classes found.')
        return

    # Build source-hash lookup: class_name -> stored source hash
    stored_hashes: dict[str, str] = {}
    for class_name in classes:
        state = src.get_latest_state_for_class(class_name)
        if state:
            stored_hashes[class_name] = state.source_hash

    # Group by object type — Data, Table, Figure, then others alphabetically
    by_type: dict[str, list[str]] = {}
    for class_name, info in classes.items():
        by_type.setdefault(info.object_type or 'Unknown', []).append(class_name)

    type_order = list(ObjectType)
    other_types = sorted(t for t in by_type if t not in type_order)
    ordered_types = [t for t in type_order if t in by_type] + other_types

    any_stale = False
    markers: set[str] = set()
    for object_type in ordered_types:
        class_names = sorted(by_type[object_type])
        click.echo(f'── {object_type} ({len(class_names)}) ──')
        col = max(len(cn) for cn in class_names)
        for class_name in class_names:
            info = classes[class_name]
            if not info.known:
                marker = '?'
            elif info.source_stale:
                marker = 'C'
            elif info.deps_stale:
                marker = 'D'
            else:
                marker = ' '
            any_stale = any_stale or marker in ('C', 'D')
            markers.add(marker)
            src_hash = stored_hashes.get(class_name)
            src_str = f'src:{_fmt_hash(src_hash, full_hash)}' if src_hash else 'src:—'
            all_deps = info.call_dependency_names + info.inheritance_dependency_names
            deps_str = f'deps:[{", ".join(all_deps)}]' if all_deps else 'deps:—'
            click.echo(f'  {_marker(marker)} {class_name:<{col}}  {src_str}  {deps_str}')
        click.echo()

    legend = _legend(markers)
    if legend:
        click.echo(legend)

    if any_stale:
        raise SystemExit(1)
