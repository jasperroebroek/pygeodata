from __future__ import annotations

import functools
from datetime import datetime
from os import scandir
from pathlib import Path
from typing import TYPE_CHECKING

import click

from pygeodata.catalog.types import CatalogNewerError, CatalogOutdatedError
from pygeodata.config import get_config
from pygeodata.project import ImportReport, import_project_modules
from pygeodata.registries.registry import EntryRegistry
from pygeodata.static import analyse_project, compare_with_imports
from pygeodata.tracked_object import live_tree_hash

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pygeodata.analysis import ProjectAnalysis
    from pygeodata.catalog.types import CatalogEntry


def _fmt_mtime(mtime: str) -> str:
    try:
        return datetime.fromisoformat(mtime).strftime('%Y-%m-%d %H:%M')
    except (ValueError, AttributeError):
        return mtime


def _import_options(f):
    @click.option(
        '--import-all',
        'do_import',
        is_flag=True,
        default=False,
        help='Import all .py files from the current directory to populate the class registry.',
    )
    @click.option('--verbose-import', is_flag=True, default=False, help='Print each module imported.')
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


def _echo_import_report(report: ImportReport, verbose: bool) -> None:
    summary = f'Imported {len(report.imported)} module(s)'
    if report.already_loaded:
        summary += f' ({len(report.already_loaded)} more already loaded)'
    click.echo(summary)
    if verbose:
        for name in report.imported:
            click.echo(f'  imported {name}')
        for name in report.already_loaded:
            click.echo(f'  already loaded {name}')


def _import_project_modules(root: Path, verbose: bool = False) -> None:
    report = import_project_modules(root)
    _echo_import_report(report, verbose)
    if verbose:
        for name, error in report.failed.items():
            click.echo(f'  skip {name}: {error}')


def _project_source(do_import: bool, verbose: bool) -> ProjectAnalysis | None:
    """The project's classes read from source; None with ``--import-all``, whose imported classes serve instead.

    Importing also reports every class where imports and source disagree.
    """
    analysis = analyse_project(Path.cwd())
    for problem in analysis.problems:
        click.echo(f'Warning: {problem}', err=True)
    if not do_import:
        return analysis
    _import_project_modules(Path.cwd(), verbose=verbose)
    for message in compare_with_imports(analysis):
        click.echo(f'Warning: {message}', err=True)
    return None


def _tree_hash_lookup(analysis: ProjectAnalysis | None) -> Callable[[str], str | None]:
    return analysis.tree_hashes.get if analysis is not None else live_tree_hash


# One legend for every staleness marker the CLI prints.
LEGEND = {
    'F': ('red', 'written by another pygeodata format'),
    'S': ('yellow', 'stale: its dependency tree changed'),
    'C': ('red', 'source changed since it last ran'),
    'D': ('yellow', 'dependencies changed since it last ran'),
    '?': ('bright_black', 'unknown: class not found in the project'),
}


def _marker(code: str) -> str:
    return click.style(code, fg=LEGEND[code][0]) if code in LEGEND else ' '


def _legend(codes: Iterable[str]) -> str | None:
    present = set(codes)
    used = [code for code in LEGEND if code in present]
    if not used:
        return None
    return '  '.join(f'{_marker(code)} {LEGEND[code][1]}' for code in used)


def _check_roots_readable(roots: tuple[Path, ...]) -> None:
    """Error on roots that exist but cannot be read; warn about roots that do not exist yet."""
    unreadable: list[tuple[Path, str]] = []
    missing: list[Path] = []
    for root in dict.fromkeys(roots):
        try:
            with scandir(root):
                pass
        except PermissionError as err:  # noqa: PERF203
            unreadable.append((root, str(err.strerror)))
        except OSError:
            missing.append(root)

    if unreadable:
        lines = '\n'.join(f'  {root} ({reason})' for root, reason in unreadable)
        raise click.ClickException(
            f'Cache root is not readable:\n{lines}\nCheck the mount and privacy settings (network volumes) '
            'before continuing.',
        )
    for root in missing:
        click.echo(f'Warning: {root} does not exist yet; nothing is indexed from it.')


def _apply_registry_path(registry_path: str | None) -> Path | None:
    """Override path_registry from CLI arg if provided, return the resolved Path."""
    if registry_path:
        p = Path(registry_path)
        get_config().update(path_registry=p)
        return p
    return None


def _fmt_hash(h: str, full: bool, width: int = 12) -> str:
    return h if full else h[:width]


_registry_option = click.option(
    '--registry',
    'registry_path',
    default=None,
    help='Path to .source/ registry root (defaults to config value).',
)

_full_hash_option = click.option(
    '--full-hash',
    'full_hash',
    is_flag=True,
    default=False,
    help='Print full hashes instead of truncated prefixes.',
)


def _entry_registry() -> EntryRegistry:
    try:
        return EntryRegistry()
    except (CatalogOutdatedError, CatalogNewerError) as exc:
        raise click.ClickException(str(exc)) from exc


def _resolve_entry(reg: EntryRegistry, hash_prefix: str) -> CatalogEntry:
    """The entry for a partial or full state hash. Raises ClickException on ambiguity/miss."""
    matches = reg.match_prefix(hash_prefix)
    if not matches:
        raise click.ClickException(f'No entry found matching {hash_prefix!r}.')
    if len(matches) > 1:
        raise click.ClickException(
            f'{len(matches)} entries match {hash_prefix!r}: ' + ', '.join(h[:12] for h in matches),
        )
    return reg.entries[matches[0]]
