from pathlib import Path

import click

from pygeodata.cli._lazy import LazyGroup

COMMANDS: dict[str, tuple[str, str]] = {
    'browse': ('pygeodata.cli.browse:browse', 'Launch the registry browser.'),
    'catalog': ('pygeodata.cli.maintenance:catalog_group', 'Inspect and rebuild the .catalog.json cache catalog.'),
    'classes': (
        'pygeodata.cli.classes:classes_cmd',
        "Show staleness for the project's classes and those with cache entries.",
    ),
    'clean-cache': ('pygeodata.cli.maintenance:clean_cache_cmd', 'Remove stale or invalid cache entries.'),
    'clean-locks': (
        'pygeodata.cli.maintenance:clean_locks_cmd',
        'Remove process.lock files left behind by terminated runs.',
    ),
    'clean-source': (
        'pygeodata.cli.maintenance:clean_source_cmd',
        'Remove orphaned code snapshots and dependency trees from .source/.',
    ),
    'code': ('pygeodata.cli.source:code', 'Inspect the .source/ code registry.'),
    'entry': ('pygeodata.cli.entries:entry', 'Inspect cache entries.'),
    'export': (
        'pygeodata.cli.archive:export_archive',
        'Export cache entries to a pygeodata archive (a counterpart to `import`).',
    ),
    'import': (
        'pygeodata.cli.archive:import_archive_cmd',
        'Import a pygeodata export archive into the current project.',
    ),
    'snapshot': ('pygeodata.cli.source:snapshot', 'Inspect the .source/ dependency-tree snapshot registry.'),
    'versions': (
        'pygeodata.cli.source:versions_cmd',
        'Show the full version timeline: groups, changed classes, and dep-tree snapshots.',
    ),
}


@click.group(cls=LazyGroup, lazy_commands=COMMANDS)
@click.option(
    '--settings',
    'settings_path',
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        'Import this settings module before running the command, alongside any '
        'pygeodata_settings.py and src/pygeodata_settings.py in the current directory.'
    ),
)
def cli(settings_path: Path | None) -> None:
    """Pygeodata command-line interface.

    Every command first imports the settings modules — from ``--settings`` when given,
    plus ``pygeodata_settings.py`` and ``src/pygeodata_settings.py`` in the working
    directory — so the CLI and Python code share one configuration point.
    """
    from pygeodata.config import load_project_settings  # noqa: PLC0415 — config loads numpy; keep it off `--help`

    try:
        loaded = load_project_settings(settings_path=settings_path)
    except Exception as exc:
        raise click.ClickException(f'settings module failed: {exc}') from exc
    for path in loaded:
        click.echo(f'Loaded settings: {path}')
