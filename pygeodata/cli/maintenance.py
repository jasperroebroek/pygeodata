from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import click

from pygeodata._version import FORMAT_VERSION
from pygeodata.cache import clean_cache, clean_lock_files, clean_source_registry
from pygeodata.catalog.store import refresh
from pygeodata.cli.common import (
    _check_roots_readable,
    _echo_import_report,
    _entry_registry,
    _import_options,
    _import_project_modules,
)
from pygeodata.config import get_config
from pygeodata.project import import_project_modules


@click.group('catalog')
def catalog_group() -> None:
    """Inspect and rebuild the .catalog.json cache catalog."""


@catalog_group.command('build')
@click.option(
    '--cache-root',
    'cache_roots',
    multiple=True,
    type=click.Path(),
    help='Cache root to index (repeatable). Defaults to the configured cache roots.',
)
@click.option(
    '--prune/--no-prune',
    default=True,
    help='Also drop entries whose cache entry no longer exists (default: prune).',
)
@_import_options
def catalog_build_cmd(
    cache_roots: tuple[str, ...],
    prune: bool,
    do_import: bool,
    verbose_import: bool,
) -> None:
    """Index every cache entry into the catalog JSON."""
    config = get_config()
    if do_import:
        _import_project_modules(Path.cwd(), verbose=verbose_import)
    if not config.catalog_enabled:
        raise click.ClickException('Catalog writes are disabled: catalog_enabled is False.')
    roots = [Path(r) for r in cache_roots] if cache_roots else list(config.cache_roots)
    _check_roots_readable((*roots, config.path_registry))
    result = refresh(roots, format_version=FORMAT_VERSION, prune=prune)
    indexed, pruned = result.indexed, result.prune.pruned
    click.echo(
        f'Indexed {indexed} entr{"y" if indexed == 1 else "ies"}; pruned {pruned} entr{"y" if pruned == 1 else "ies"}',
    )
    for root in result.prune.unavailable_roots:
        click.echo(f'Not pruned, cache root missing or empty: {root}', err=True)
    click.echo(f'Catalog: {config.path_catalog}')


@catalog_group.command('info')
def catalog_info_cmd() -> None:
    """Show where the catalog was written, what it holds, and how it differs from disk."""
    registry = _entry_registry()
    header, status = registry.header, registry.status
    writer = header.writer
    written_at = datetime.fromtimestamp(writer.written_at, tz=UTC).astimezone().strftime('%Y-%m-%d %H:%M')
    click.echo(f'Catalog      {get_config().path_catalog}')
    click.echo(f'ID           {header.catalog_id}  (version {header.catalog_version})')
    click.echo(f'Written      {written_at} by pygeodata {writer.pygeodata} (Python {writer.python}) on {writer.host}')
    click.echo(f'From root    {writer.root}')
    mode = 'read-only: written from another project root' if status.read_only else 'writable'
    click.echo(f'Mode         {mode}')
    counts = Counter(entry.object_type for entry in registry.entries.values())
    click.echo(f'Entries      {len(registry.entries)}  ({", ".join(f"{t}: {n}" for t, n in sorted(counts.items()))})')
    diff = status.diff
    if diff.unindexed:
        click.echo(f'Not indexed  {len(diff.unindexed)} entry dir(s) on disk')
    if diff.missing:
        click.echo(
            f'Missing      {len(diff.missing)} entr{"y" if len(diff.missing) == 1 else "ies"} without an entry dir',
        )
    for root in diff.unavailable_roots:
        click.echo(f'Unavailable  {root} (missing or empty; its entries are kept)')


@click.command('clean-cache')
@click.option('--verbose-import', is_flag=True, default=False, help='Print each module imported.')
@click.option(
    '--dry-run/--no-dry-run',
    default=True,
    help='Actually delete files with --no-dry-run (default is dry run).',
)
@click.option(
    '--delete-unregistered/--no-delete-unregistered',
    default=True,
    help='Skip entries whose class is not in the registry with --no-delete-unregistered.',
)
@click.option(
    '--check-class-params',
    is_flag=True,
    default=False,
    help=(
        'Also validate TrackedObject subclasses held as parameters. Catches entries the '
        'default check reports as current; entries without a parameters.json count as stale.'
    ),
)
def clean_cache_cmd(
    verbose_import: bool,
    dry_run: bool,
    delete_unregistered: bool,
    check_class_params: bool,
) -> None:
    """Remove stale or invalid cache entries.

    Imports the project first: deciding what is stale needs its current classes. Nothing is deleted
    when any project module fails to import, since its entries would look unregistered.
    """
    report = import_project_modules(Path.cwd())
    _echo_import_report(report, verbose_import)
    if report.failed:
        failures = '\n'.join(f'  {name}: {error}' for name, error in report.failed.items())
        if not dry_run:
            raise click.ClickException(f'Not deleting anything: these project modules failed to import:\n{failures}')
        click.echo(
            f'Warning: these project modules failed to import, so a real run would refuse:\n{failures}',
            err=True,
        )
    clean_cache(
        dry_run=dry_run,
        delete_unregistered=delete_unregistered,
        check_class_params=check_class_params,
    )


@click.command('clean-locks')
@click.option(
    '--dry-run/--no-dry-run',
    default=True,
    help='Actually delete files with --no-dry-run (default is dry run).',
)
def clean_locks_cmd(dry_run: bool) -> None:
    """Remove process.lock files left behind by terminated runs.

    Locks held by a running process are skipped.  A dry run cannot detect
    held locks and lists every lock file found.
    """
    clean_lock_files(dry_run=dry_run)


@click.command('clean-source')
@_import_options
@click.option(
    '--dry-run/--no-dry-run',
    default=True,
    help='Actually delete files with --no-dry-run (default is dry run).',
)
def clean_source_cmd(do_import: bool, verbose_import: bool, dry_run: bool) -> None:
    """Remove orphaned code snapshots and dependency trees from .source/.

    Keeps the latest snapshot per class and anything referenced by a live
    cache entry.  Everything else is prunable.  Runs as dry run by default.
    """
    if do_import:
        _import_project_modules(Path.cwd(), verbose=verbose_import)
    clean_source_registry(dry_run=dry_run)
