import click

from pygeodata.cli.common import _check_roots_readable
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
    help='Minutes between automatic catalog checks and, with --import-all, project imports (0 disables).',
)
@click.option(
    '--import-all',
    'import_all',
    is_flag=True,
    default=False,
    help=(
        'Read the classes by importing the project in a subprocess instead of from source. '
        'Imports EVERY .py file at the project root and in its packages: a root-level script '
        'without an if __name__ == "__main__" guard RUNS, on every import.'
    ),
)
@click.option(
    '--read-only/--writable',
    'read_only',
    default=None,
    help=(
        'Never write into the project (a mirror), or allow it. Default: read-only when the catalog was written '
        'from another project root or catalog_enabled is False.'
    ),
)
def browse(port: int, fast: bool, reconcile_interval: float, import_all: bool, read_only: bool | None) -> None:
    """Launch the registry browser."""
    try:
        from pygeodata_browser.serve import open_registry_browser  # optional package that itself imports pygeodata
    except ImportError as err:
        raise click.ClickException(
            f'The registry browser lives in the pygeodata-browser package: '
            f'pip install pygeodata[browser] or conda install pygeodata-browser. '
            f'Original error: {err}',
        ) from err

    if read_only is False and not get_config().catalog_enabled:
        raise click.UsageError('--writable needs catalog_enabled: it is False, so nothing can be written.')

    if not get_config().catalog_enabled:
        click.echo('Note: catalog_enabled is False — the browser shows nothing until a catalog is built.')

    _check_roots_readable((*get_config().cache_roots, get_config().path_registry))

    open_registry_browser(
        port,
        fast=fast,
        reconcile_interval=reconcile_interval * 60,
        import_all=import_all,
        read_only=read_only,
    )
