import json
import shutil
import tarfile
from pathlib import Path

import click

from pygeodata.catalog.entry_catalog import discover_entries
from pygeodata.catalog.export import collect_export_files, write_export_tar
from pygeodata.cli.common import _import_options, _import_project_modules, _resolve_record
from pygeodata.config import get_config
from pygeodata.enums import JSONKeys, ObjectType


def _cache_root_from_tar(tar: tarfile.TarFile, hash_dir: str) -> Path | None:
    """Read OBJECT_TYPE from the meta.json and return the matching Artifact subclass cache root."""
    for member in tar.getmembers():
        p = Path(member.name)
        if p.parts[:1] == ('cache',) and p.parts[1] == hash_dir and p.name == 'meta.json':
            f = tar.extractfile(member)
            if f is None:
                return None
            try:
                data = json.loads(f.read())
            except (json.JSONDecodeError, OSError):
                return None
            try:
                object_type = ObjectType(data.get(JSONKeys.OBJECT_TYPE, ''))
            except ValueError:
                return None
            return get_config().get_cache_root(object_type)
    return None


def _resolve_dest(
    parts: tuple[str, ...],
    cache_roots: dict[str, Path | None],
    source_root: Path,
) -> tuple[Path, Path] | None:
    if any(p in ('..', '') or Path(p).is_absolute() for p in parts):
        return None
    if parts[0] == 'cache' and len(parts) >= 3:
        cache_root = cache_roots.get(parts[1])
        if cache_root is None:
            return None
        return cache_root / Path(*parts[1:]), cache_root
    if parts[0] in ('code', 'snapshots') and len(parts) >= 3:
        return source_root / Path(*parts), source_root
    return None


def _extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path, root: Path) -> None:
    resolved = dest.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    if src is None:
        return
    with src, resolved.open('wb') as out:
        shutil.copyfileobj(src, out)


@click.command('import')
@click.argument('archive', type=click.Path(exists=True))
def import_archive(archive: str) -> None:
    """Import a pygeodata export archive into the current project."""
    source_root = Path(get_config().path_registry)
    counts = {'cache': 0, 'code': 0, 'snapshots': 0}

    with tarfile.open(archive, 'r:*') as tar:
        cache_roots: dict[str, Path | None] = {}
        for member in tar.getmembers():
            p = Path(member.name)
            parts = p.parts
            if parts[:1] == ('cache',) and len(parts) >= 3 and p.name == 'meta.json':
                hash_dir = parts[1]
                if hash_dir not in cache_roots:
                    cache_roots[hash_dir] = _cache_root_from_tar(tar, hash_dir)

        for member in tar.getmembers():
            parts = Path(member.name).parts
            if not parts or member.isdir():
                continue
            resolved = _resolve_dest(parts, cache_roots, source_root)
            if resolved is None:
                continue
            dest, root = resolved
            if dest.exists():
                continue
            _extract_member(tar, member, dest, root)
            counts[parts[0]] = counts.get(parts[0], 0) + 1

    click.echo(
        f'Imported: {counts["cache"]} cache files, '
        f'{counts["code"]} code snapshot files, '
        f'{counts["snapshots"]} tree snapshot files.',
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
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')

    entries, entry_registry, _ = discover_entries()

    if export_all:
        record_ids = list(entries)
    elif class_name:
        record_ids = [rid for rid, e in entries.items() if e.class_name == class_name]
        if not record_ids:
            raise click.ClickException(f'No entries found for class {class_name!r}.')
    elif type_filter:
        tf = type_filter.upper()
        record_ids = [rid for rid, e in entries.items() if (e.object_type or '').upper() == tf]
        if not record_ids:
            raise click.ClickException(f'No entries found for type {type_filter!r}.')
    else:
        record_ids = [_resolve_record(entry_registry, prefix)[0] for prefix in hash_prefixes]

    files, warnings = collect_export_files(record_ids, entries, include_snapshots)
    write_export_tar(files, Path(archive))
    click.echo(
        f'Exported {len(record_ids)} entr{"y" if len(record_ids) == 1 else "ies"} ({len(files)} files) to {archive}',
    )
    for warning in warnings:
        click.echo(click.style(f'  warning: {warning}', fg='yellow'))
