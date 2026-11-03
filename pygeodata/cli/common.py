import functools
import importlib.util
import sys
from datetime import datetime
from os import scandir
from pathlib import Path

import click

from pygeodata.config import get_config


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
    """Import all .py files under root, silently skipping failures. Returns import count."""
    if root not in sys.path:
        sys.path.insert(0, str(root))
    cfg = get_config()
    _config_dirs = set()
    for _p in (*cfg.cache_roots, cfg.path_registry):
        if _p.parts and not _p.is_absolute():
            _config_dirs.add(_p.parts[0])
    _skip_dirs = {'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test', 'cache'} | _config_dirs
    count = 0
    for py_file in sorted(root.rglob('*.py')):
        parts = py_file.relative_to(root).parts
        if any(p.startswith(('.', '__pycache__')) or p in _skip_dirs for p in parts):
            continue
        # Settings files are imported by the cli group callback, honouring the
        # --settings > pygeodata_settings.py > src/pygeodata_settings.py priority;
        # --import-all must never import a second settings file.
        if parts[-1].startswith(('test_', 'conftest', 'setup', 'manage')) or parts[-1] == 'pygeodata_settings.py':
            continue
        stem = '.'.join(py_file.relative_to(root).with_suffix('').parts)
        module_name = f'pygeodata_project_{stem}'
        if module_name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                count += 1
                if verbose:
                    click.echo(f'  imported {module_name}')
        except Exception as e:  # noqa: BLE001
            if verbose:
                click.echo(f'  skip {module_name}: {e}')
    return count


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


def _resolve_record(reg, hash_prefix: str):
    """Return (state_hash, EntryRecord) for a partial or full hash. Raises ClickException on ambiguity/miss."""
    matches = [(h, r) for h, r in reg.records.items() if h.startswith(hash_prefix)]
    if not matches:
        raise click.ClickException(f'No entry found matching {hash_prefix!r}.')
    if len(matches) > 1:
        raise click.ClickException(
            f'{len(matches)} entries match {hash_prefix!r}: ' + ', '.join(h[:12] for h, _ in matches),
        )
    return matches[0]
