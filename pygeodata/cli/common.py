from __future__ import annotations

import functools
from datetime import datetime
from os import scandir
from pathlib import Path
from typing import TYPE_CHECKING

import click

from pygeodata.catalog.types import CatalogNewerError, CatalogOutdatedError
from pygeodata.config import get_config
from pygeodata.project import import_project_modules
from pygeodata.registries.registry import EntryRegistry

if TYPE_CHECKING:
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


def _import_project_modules(root: Path, verbose: bool = False) -> int:
    report = import_project_modules(root)
    if verbose:
        for name in report.imported:
            click.echo(f'  imported {name}')
        for name, error in report.failed.items():
            click.echo(f'  skip {name}: {error}')
    return len(report.imported)


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
