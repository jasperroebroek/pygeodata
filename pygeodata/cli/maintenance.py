from pathlib import Path

import click

from pygeodata._version import FORMAT_VERSION
from pygeodata.cache import clean_cache, clean_lock_files, clean_source_registry
from pygeodata.catalog.store import refresh
from pygeodata.cli.common import _check_roots_readable, _import_options, _import_project_modules
from pygeodata.config import get_config


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
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s) from {Path.cwd()}')
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


@click.command('clean-cache')
@_import_options
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
    do_import: bool,
    verbose_import: bool,
    dry_run: bool,
    delete_unregistered: bool,
    check_class_params: bool,
) -> None:
    """Remove stale or invalid cache entries."""
    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
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
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
    clean_source_registry(dry_run=dry_run)
