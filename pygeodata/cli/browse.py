from pathlib import Path

import click

from pygeodata.cli.common import _check_roots_readable, _import_options, _import_project_modules
from pygeodata.config import get_config


@click.command('browse')
@click.option('--port', default=0, show_default=True, help='Port to listen on (0 = random free port).')
@click.option(
    '--fast/--no-fast',
    'fast',
    default=False,
    help='Serve the UI immediately from the catalog bootstrap while the full state builds (default: block until ready).',
)
@click.option(
    '--reconcile-interval',
    default=5.0,
    show_default=True,
    type=float,
    help='Minutes between automatic catalog reconciles (0 disables).',
)
@click.option(
    '--reimport-on-reconcile/--no-reimport-on-reconcile',
    default=True,
    help='Also reimport project modules on the periodic reconcile.',
)
@_import_options
def browse(
    port: int,
    fast: bool,
    reconcile_interval: float,
    reimport_on_reconcile: bool,
    do_import: bool,
    verbose_import: bool,
) -> None:
    """Launch the registry browser."""
    try:
        from pygeodata_browser.serve import open_registry_browser  # optional package that itself imports pygeodata
    except ImportError as err:
        raise click.ClickException(
            f'The registry browser lives in the pygeodata-browser package: pip install pygeodata[browser]. '
            f'Original error: {err}',
        ) from err

    if do_import:
        root = Path.cwd()
        n = _import_project_modules(root, verbose=verbose_import)
        click.echo(f'Imported {n} module(s) from {root}')

    if not get_config().catalog_enabled:
        click.echo('Note: catalog_enabled is False — the browser shows nothing until a catalog is built.')

    _check_roots_readable((*get_config().cache_roots, get_config().path_registry))

    open_registry_browser(
        port,
        fast=fast,
        reconcile_interval=reconcile_interval * 60,
        reimport_on_reconcile=reimport_on_reconcile,
    )
