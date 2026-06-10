import importlib.util
import json
import shutil
import sys
import tarfile
from pathlib import Path

import click

from pygeodata.cache import clean_cache
from pygeodata.config import JSONKeys, get_config
from pygeodata.data import Data
from pygeodata.figure import Figure
from pygeodata.registry_browser.serve import open_registry_browser


@click.group()
def cli():
    pass


_FAMILY_BY_NAME = {'Data': Data, 'Figure': Figure}


def _cache_root_from_tar(tar: tarfile.TarFile, hash_dir: str) -> Path | None:
    """Read OBJECT_TYPE from the hash.json and return the matching Artifact subclass cache root."""
    for member in tar.getmembers():
        p = Path(member.name)
        if p.parts[:1] == ('cache',) and p.parts[1] == hash_dir and p.name.endswith('.hash.json'):
            f = tar.extractfile(member)
            if f is None:
                return None
            try:
                data = json.loads(f.read())
            except (json.JSONDecodeError, OSError):
                return None
            object_type = data.get(JSONKeys.OBJECT_TYPE, '')
            family = _FAMILY_BY_NAME.get(object_type)
            return family.get_cache_root() if family else None
    return None


def _resolve_dest(
    parts: tuple[str, ...],
    cache_roots: dict[str, Path | None],
    source_root: Path,
    existing_dirs: set[Path],
) -> Path | None:
    if parts[0] == 'cache' and len(parts) >= 3:
        cache_root = cache_roots.get(parts[1])
        if cache_root is None:
            return None
        if (cache_root / parts[1]) in existing_dirs:
            return None
        return cache_root / Path(*parts[1:])
    if parts[0] in ('code', 'snapshots') and len(parts) >= 3:
        if (source_root / parts[0] / parts[1]) in existing_dirs:
            return None
        return source_root / Path(*parts)
    return None


def _existing_entry_dirs(cache_roots: dict[str, Path | None], source_root: Path) -> set[Path]:
    dirs: set[Path] = set()
    for hash_dir, cache_root in cache_roots.items():
        if cache_root is not None:
            d = cache_root / hash_dir
            if d.exists():
                dirs.add(d)
    for top in ('code', 'snapshots'):
        d = source_root / top
        if d.exists():
            dirs.update(d.iterdir())
    return dirs


def _extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        return
    with src, dest.open('wb') as out:
        shutil.copyfileobj(src, out)


def _import_project_modules(root: Path) -> int:
    """Import all .py files under root, silently skipping failures. Returns import count."""
    if root not in sys.path:
        sys.path.insert(0, str(root))
    count = 0
    for py_file in sorted(root.rglob('*.py')):
        # Skip hidden dirs, virtual envs, build dirs, and test files
        parts = py_file.relative_to(root).parts
        _skip_dirs = {'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test'}
        if any(p.startswith(('.', '__pycache__')) or p in _skip_dirs for p in parts):
            continue
        if parts[-1].startswith(('test_', 'conftest', 'setup', 'manage')):
            continue
        module_name = '.'.join(py_file.relative_to(root).with_suffix('').parts)
        if module_name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                count += 1
        except Exception:  # noqa: BLE001
            pass
    return count


@cli.command('browse')
@click.option('--port', default=0, show_default=True, help='Port to listen on (0 = random free port).')
@click.option(
    '--no-import',
    'do_import',
    is_flag=True,
    flag_value=False,
    default=True,
    help='Skip auto-importing .py files from the current directory.',
)
def browse(port: int, do_import: bool) -> None:
    """Launch the registry browser. Auto-imports .py files to populate the class registry."""
    if do_import:
        root = Path.cwd()
        n = _import_project_modules(root)
        if n:
            click.echo(f'Imported {n} module(s) from {root}')

    open_registry_browser(port)


@cli.command('clean-cache')
@click.option(
    '--no-dry-run',
    'dry_run',
    is_flag=True,
    flag_value=False,
    default=True,
    help='Actually delete files (default is dry run).',
)
@click.option(
    '--no-delete-unregistered',
    'delete_unregistered',
    is_flag=True,
    flag_value=False,
    default=True,
    help='Skip entries whose class is not in the registry.',
)
def clean_cache_cmd(dry_run: bool, delete_unregistered: bool) -> None:
    """Remove stale or invalid cache entries."""
    clean_cache(dry_run=dry_run, delete_unregistered=delete_unregistered)


@cli.command('import')
@click.argument('archive', type=click.Path(exists=True))
def import_archive(archive: str) -> None:
    """Import a pygeodata export archive into the current project."""
    source_root = Path(get_config().path_registry)
    counts = {'cache': 0, 'code': 0, 'snapshots': 0}

    with tarfile.open(archive, 'r:gz') as tar:
        cache_roots: dict[str, Path | None] = {}
        for member in tar.getmembers():
            p = Path(member.name)
            parts = p.parts
            if parts[:1] == ('cache',) and len(parts) >= 3 and p.name.endswith('.hash.json'):
                hash_dir = parts[1]
                if hash_dir not in cache_roots:
                    cache_roots[hash_dir] = _cache_root_from_tar(tar, hash_dir)

        existing_dirs = _existing_entry_dirs(cache_roots, source_root)

        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not parts or member.isdir():
                continue
            dest = _resolve_dest(parts, cache_roots, source_root, existing_dirs)
            if dest is None or dest.exists():
                continue
            _extract_member(tar, member, dest)
            counts[parts[0]] = counts.get(parts[0], 0) + 1

    click.echo(
        f'Imported: {counts["cache"]} cache files, '
        f'{counts["code"]} code snapshot files, '
        f'{counts["snapshots"]} tree snapshot files.',
    )
