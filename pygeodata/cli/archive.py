from pathlib import Path

import click

from pygeodata.archive import CACHE_DIR, collect_export_files, import_archive, write_export_tar
from pygeodata.cli.common import _entry_registry, _import_options, _import_project_modules, _resolve_entry
from pygeodata.enums import RegistryDir


@click.command('import')
@click.argument('archive', type=click.Path(exists=True))
def import_archive_cmd(archive: str) -> None:
    """Import a pygeodata export archive into the current project."""
    counts = import_archive(Path(archive))
    click.echo(
        f'Imported: {counts[CACHE_DIR]} cache files, '
        f'{counts[RegistryDir.CODE]} code snapshot files, '
        f'{counts[RegistryDir.SNAPSHOTS]} tree snapshot files.',
    )


@click.command('export')
@click.argument('archive', type=click.Path())
@click.argument('hash_prefixes', nargs=-1)
@_import_options
@click.option('--class', 'class_name', default=None, help='Export all entries of this class.')
@click.option(
    '--type',
    'type_filter',
    default=None,
    help='Export all entries of one object type: Data, Table, or Figure (case-insensitive).',
)
@click.option('--all', 'export_all', is_flag=True, default=False, help='Export every cache entry.')
@click.option(
    '--snapshots/--no-snapshots',
    'include_snapshots',
    default=True,
    help='Include code/dep-tree snapshots needed to reproduce the entries (default: included).',
)
def export_archive(
    archive: str,
    hash_prefixes: tuple[str, ...],
    do_import: bool,
    verbose_import: bool,
    class_name: str | None,
    type_filter: str | None,
    export_all: bool,
    include_snapshots: bool,
) -> None:
    """Export cache entries to a pygeodata archive (a counterpart to `import`).

    Select entries by one or more HASH_PREFIXES, by --class, by --type, or with --all.
    """
    n_selectors = sum((bool(hash_prefixes), bool(class_name), bool(type_filter), export_all))
    if n_selectors == 0:
        raise click.UsageError('Provide HASH_PREFIXES, --class, --type, or --all.')
    if n_selectors > 1:
        raise click.UsageError('Provide only one of HASH_PREFIXES, --class, --type, or --all.')

    if do_import:
        _import_project_modules(Path.cwd(), verbose=verbose_import)

    registry = _entry_registry()
    entries = registry.entries

    if export_all:
        record_ids = list(entries)
    elif class_name:
        record_ids = [h for h, e in entries.items() if e.class_name == class_name]
        if not record_ids:
            raise click.ClickException(f'No entries found for class {class_name!r}.')
    elif type_filter:
        tf = type_filter.upper()
        record_ids = [h for h, e in entries.items() if e.object_type.upper() == tf]
        if not record_ids:
            raise click.ClickException(f'No entries found for type {type_filter!r}.')
    else:
        record_ids = [_resolve_entry(registry, prefix).state_hash for prefix in hash_prefixes]

    files, warnings = collect_export_files(record_ids, registry, include_snapshots)
    write_export_tar(files, Path(archive))
    click.echo(
        f'Exported {len(record_ids)} entr{"y" if len(record_ids) == 1 else "ies"} ({len(files)} files) to {archive}',
    )
    for warning in warnings:
        click.echo(click.style(f'  warning: {warning}', fg='yellow'))
