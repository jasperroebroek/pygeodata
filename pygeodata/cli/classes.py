from pathlib import Path

import click

from pygeodata.catalog.class_catalog import discover_loaded_classes, merge_unloaded_classes
from pygeodata.cli.common import (
    _apply_registry_path,
    _fmt_hash,
    _full_hash_option,
    _import_options,
    _import_project_modules,
    _registry_option,
)
from pygeodata.enums import ObjectType
from pygeodata.registries.registry import EntryRegistry, SourceRegistry
from pygeodata.registries.versioning import VersionRegistry

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
    """Show staleness for all loaded and known classes."""
    _apply_registry_path(registry_path)

    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')

    vreg = VersionRegistry()
    ereg = EntryRegistry()
    src = SourceRegistry()
    classes = discover_loaded_classes()
    classes = merge_unloaded_classes(classes, ereg, version_registry=vreg, src=src)

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
    for object_type in ordered_types:
        class_names = sorted(by_type[object_type])
        click.echo(f'── {object_type} ({len(class_names)}) ──')
        col = max(len(cn) for cn in class_names)
        for class_name in class_names:
            info = classes[class_name]
            if info.source_stale:
                any_stale = True
                indicator = click.style('C', fg='red')
            elif info.deps_stale:
                any_stale = True
                indicator = click.style('D', fg='yellow')
            elif not info.loaded:
                indicator = click.style('N', fg='bright_black')
            else:
                indicator = ' '
            loaded_label = 'loaded' if info.loaded else 'unloaded'
            src_hash = stored_hashes.get(class_name)
            src_str = f'src:{_fmt_hash(src_hash, full_hash)}' if src_hash else 'src:—'
            all_deps = info.call_dependency_names + info.inheritance_dependency_names
            deps_str = f'deps:[{", ".join(all_deps)}]' if all_deps else 'deps:—'
            click.echo(f'  {indicator} {class_name:<{col}}  {loaded_label}  {src_str}  {deps_str}')
        click.echo()

    if any_stale:
        raise SystemExit(1)
