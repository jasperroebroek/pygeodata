import dataclasses
import functools
import importlib.util
import json
import shutil
import sys
import tarfile
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

import click

from pygeodata.cache import clean_cache, clean_source_registry
from pygeodata.catalog.class_catalog import discover_loaded_classes, merge_unloaded_classes
from pygeodata.catalog.entry_catalog import _enrich_params_path
from pygeodata.catalog.filters import Filter, FilterOperator, FilterTarget, prescan_text
from pygeodata.catalog.types import SpecInfo
from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config
from pygeodata.data import Data
from pygeodata.figure import Figure
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.registry import EntryRegistry, SourceRegistry, TreeRegistry
from pygeodata.registries.registry_types import Version
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.spec import SpatialSpec
from pygeodata.tracked_object import TrackedObject


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


@click.group()
def cli():
    pass


def _cache_root_from_tar(tar: tarfile.TarFile, hash_dir: str) -> Path | None:
    """Read OBJECT_TYPE from the meta.json and return the matching Artifact subclass cache root."""
    family_by_name = {'Data': Data, 'Figure': Figure}
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
            object_type = data.get(JSONKeys.OBJECT_TYPE, '')
            family = family_by_name.get(object_type)
            return family.get_cache_root() if family else None
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


def _import_project_modules(root: Path, verbose: bool = False) -> int:
    """Import all .py files under root, silently skipping failures. Returns import count."""
    if root not in sys.path:
        sys.path.insert(0, str(root))
    cfg = get_config()
    _config_dirs = set()
    for _p in (cfg.path_cache, cfg.path_figures, cfg.path_registry):
        if _p.parts and not _p.is_absolute():
            _config_dirs.add(_p.parts[0])
    _skip_dirs = {'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test', 'cache'} | _config_dirs
    count = 0
    for py_file in sorted(root.rglob('*.py')):
        parts = py_file.relative_to(root).parts
        if any(p.startswith(('.', '__pycache__')) or p in _skip_dirs for p in parts):
            continue
        if parts[-1].startswith(('test_', 'conftest', 'setup', 'manage')):
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


@cli.command('browse')
@click.option('--port', default=0, show_default=True, help='Port to listen on (0 = random free port).')
@_import_options
def browse(port: int, do_import: bool, verbose_import: bool) -> None:
    """Launch the registry browser."""
    from pygeodata.registry_browser.serve import open_registry_browser

    if do_import:
        root = Path.cwd()
        n = _import_project_modules(root, verbose=verbose_import)
        click.echo(f'Imported {n} module(s) from {root}')

    open_registry_browser(port)


@cli.command('clean-cache')
@_import_options
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
def clean_cache_cmd(do_import: bool, verbose_import: bool, dry_run: bool, delete_unregistered: bool) -> None:
    """Remove stale or invalid cache entries."""
    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
    clean_cache(dry_run=dry_run, delete_unregistered=delete_unregistered)


@cli.command('clean-source')
@_import_options
@click.option(
    '--no-dry-run',
    'dry_run',
    is_flag=True,
    flag_value=False,
    default=True,
    help='Actually delete files (default is dry run).',
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


# ---------------------------------------------------------------------------
# shared
# ---------------------------------------------------------------------------

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


def _apply_registry_path(registry_path: str | None) -> Path | None:
    """Override path_registry from CLI arg if provided, return the resolved Path."""
    if registry_path:
        p = Path(registry_path)
        get_config().update(path_registry=p)
        return p
    return None


def _fmt_hash(h: str, full: bool, width: int = 12) -> str:
    return h if full else h[:width]


def _resolve_source_hash(reg: SourceRegistry, prefix: str) -> str:
    """Resolve a source-hash prefix to a full hash, or raise ClickException."""
    h = reg.resolve_hash_prefix(prefix)
    if h is None:
        raise click.ClickException(f'No source hash found matching {prefix!r}.')
    return h


def _resolve_dep_hash(trees: TreeRegistry, prefix: str) -> str:
    """Resolve a dep-hash prefix to a full hash, or raise ClickException."""
    h = trees.resolve_hash_prefix(prefix)
    if h is None:
        raise click.ClickException(f'No dep-tree hash found matching {prefix!r}.')
    return h


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


@cli.group()
def code():
    """Inspect the .source/ code registry."""


@code.command('list')
@_registry_option
@_full_hash_option
@click.option('--verbose', '-v', is_flag=True, default=False, help='Show all snapshots per class.')
def code_list(registry_path: str | None, full_hash: bool, verbose: bool) -> None:
    """List all tracked classes."""
    reg = SourceRegistry(_apply_registry_path(registry_path))

    by_type: dict[str, list[str]] = {}
    for class_name in sorted(reg.class_names):
        states = reg.get_states(class_name)
        object_type = states[-1].object_type if states else 'Unknown'
        by_type.setdefault(object_type, []).append(class_name)

    if not by_type:
        click.echo('No snapshots found.')
        return

    for object_type in sorted(by_type):
        class_names = by_type[object_type]
        click.echo(f'{object_type} ({len(class_names)} {"class" if len(class_names) == 1 else "classes"})')
        click.echo()
        for class_name in class_names:
            states = list(reversed(reg.get_states(class_name)))
            n = len(states)
            click.echo(f'  {class_name} ({n} {"snapshot" if n == 1 else "snapshots"})')
            if verbose:
                for s in states:
                    click.echo(f'    {_fmt_mtime(s.registered_at)}  {_fmt_hash(s.source_hash, full_hash)}')
        click.echo()


@code.command('show')
@_registry_option
@_full_hash_option
@click.option('--class', 'class_name', default=None, help='Class name to show snapshots for.')
@click.option('--hash', 'source_hash', default=None, help='Source hash to look up.')
def code_show(registry_path: str | None, full_hash: bool, class_name: str | None, source_hash: str | None) -> None:
    """Show snapshot metadata for a class or hash."""
    if not class_name and not source_hash:
        raise click.UsageError('Provide --class or --hash.')

    reg = SourceRegistry(_apply_registry_path(registry_path))

    if source_hash:
        source_hash = _resolve_source_hash(reg, source_hash)
        state = reg.get_state_by_hash(source_hash)
        if state is None:
            raise click.ClickException(f'Hash {source_hash!r} not found.')
        click.echo(f'Class      {state.class_name}')
        click.echo(f'Registered {_fmt_mtime(state.registered_at)}')
        click.echo(f'Type       {state.object_type}')
        click.echo(f'Hash       {state.source_hash}')
        return

    states = reg.get_states(class_name)
    if not states:
        raise click.ClickException(f'Class {class_name!r} not found.')
    click.echo(f'{"Registered At":<20}  Hash')
    for s in sorted(states, key=lambda s: s.registered_at, reverse=True):
        click.echo(f'{_fmt_mtime(s.registered_at):<20}  {_fmt_hash(s.source_hash, full_hash)}')


@code.command('versions')
@_registry_option
@click.option('--class', 'class_name', default=None, help='List version groups that include this class.')
@click.option('--hash', 'source_hash', default=None, help='Show which version group owns this source hash (prefix ok).')
def code_versions(registry_path: str | None, class_name: str | None, source_hash: str | None) -> None:
    """Look up which version group contains a given class or source hash."""
    root = _apply_registry_path(registry_path)
    vreg = VersionRegistry(root)

    if source_hash:
        source_hash = _resolve_source_hash(vreg.source_registry, source_hash)
        match = vreg.version_for_source_hash(source_hash)
        if match is None:
            raise click.ClickException(f'Hash {source_hash!r} not found.')
        click.echo(vreg.label(match))
        return

    if not vreg.versions:
        click.echo('No version groups found.')
        return

    for vi in vreg.versions:
        if class_name and class_name not in vi.class_names:
            continue
        click.echo(vreg.label(vi))


@code.command('source')
@_registry_option
@_full_hash_option
@click.option('--class', 'class_name', default=None, help='Show latest source for this class.')
@click.option('--hash', 'source_hashes', multiple=True, help='Source hash(es). One to show, two to diff.')
@click.option('--diff', 'do_diff', is_flag=True, default=False, help='Diff latest vs previous (with --class).')
@click.option('--expand', is_flag=True, default=False, help='Show full file context in diffs.')
@click.option('--no-color', 'color', is_flag=True, flag_value=False, default=True, help='Disable diff colors.')
def code_source(
    registry_path: str | None,
    full_hash: bool,
    class_name: str | None,
    source_hashes: tuple[str, ...],
    do_diff: bool,
    expand: bool,
    color: bool,
) -> None:
    """Print or diff source code."""
    if not class_name and not source_hashes:
        raise click.UsageError('Provide --class or --hash.')
    if class_name and source_hashes:
        raise click.UsageError('Provide --class or --hash, not both.')
    if len(source_hashes) > 2:
        raise click.UsageError('Provide at most two --hash values.')

    reg = SourceRegistry(_apply_registry_path(registry_path))

    if len(source_hashes) == 2:
        ha = _resolve_source_hash(reg, source_hashes[0])
        hb = _resolve_source_hash(reg, source_hashes[1])
        _echo_diff_hashes(reg, ha, hb, expand=expand, color=color, full_hash=full_hash)
    elif len(source_hashes) == 1:
        h = _resolve_source_hash(reg, source_hashes[0])
        _echo_source_by_hash(reg, h, do_diff, expand=expand, color=color, full_hash=full_hash)
    else:
        _echo_source_by_class(reg, class_name, do_diff, expand=expand, color=color, full_hash=full_hash)


def _colorize_diff(lines: list[str]) -> str:
    result = []
    for line in lines:
        if line.startswith(('+++', '---')):
            result.append(click.style(line, bold=True))
        elif line.startswith('+'):
            result.append(click.style(line, fg='green'))
        elif line.startswith('-'):
            result.append(click.style(line, fg='red'))
        elif line.startswith('@@'):
            result.append(click.style(line, fg='cyan'))
        else:
            result.append(line)
    return ''.join(result)


def _make_diff(text_a: str, text_b: str, label_a: str, label_b: str, expand: bool, color: bool) -> str:
    kwargs = {} if expand else {'n': 3}
    lines = list(
        unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            **kwargs,
        ),
    )
    return _colorize_diff(lines) if color else ''.join(lines)


def _echo_diff_hashes(
    reg: SourceRegistry,
    hash_a: str,
    hash_b: str,
    expand: bool,
    color: bool,
    full_hash: bool = False,
) -> None:
    text_a = reg.get_source(hash_a)
    text_b = reg.get_source(hash_b)
    if text_a is None:
        raise click.ClickException(f'Hash {hash_a!r} not found.')
    if text_b is None:
        raise click.ClickException(f'Hash {hash_b!r} not found.')
    click.echo(
        _make_diff(text_a, text_b, _fmt_hash(hash_a, full_hash, 8), _fmt_hash(hash_b, full_hash, 8), expand, color),
        nl=False,
    )


def _echo_source_by_hash(
    reg: SourceRegistry,
    source_hash: str,
    do_diff: bool,
    expand: bool,
    color: bool,
    full_hash: bool = False,
) -> None:
    if not do_diff:
        text = reg.get_source(source_hash)
        if text is None:
            raise click.ClickException(f'Hash {source_hash!r} not found.')
        click.echo(text, nl=False)
        return
    prev_state = reg.get_previous_state(source_hash)
    if prev_state is None:
        raise click.ClickException('No previous snapshot to diff against.')
    prev_hash = prev_state.source_hash
    text_a = reg.get_source(prev_hash)
    text_b = reg.get_source(source_hash)
    if text_a is None or text_b is None:
        raise click.ClickException('Source file missing.')
    click.echo(
        _make_diff(
            text_a,
            text_b,
            _fmt_hash(prev_hash, full_hash, 8),
            _fmt_hash(source_hash, full_hash, 8),
            expand,
            color,
        ),
        nl=False,
    )


def _echo_source_by_class(
    reg: SourceRegistry,
    class_name: str,
    do_diff: bool,
    expand: bool,
    color: bool,
    full_hash: bool = False,
) -> None:
    states = reg.get_states(class_name)
    if not states:
        raise click.ClickException(f'Class {class_name!r} not found.')
    if not do_diff:
        text = reg.get_source(states[-1].source_hash)
        if text is None:
            raise click.ClickException(f'Source file missing for {class_name!r}.')
        click.echo(text, nl=False)
        return
    if len(states) < 2:
        raise click.ClickException(f'{class_name!r} has only one snapshot, nothing to diff.')
    prev, latest = states[-2], states[-1]
    text_a = reg.get_source(prev.source_hash)
    text_b = reg.get_source(latest.source_hash)
    if text_a is None or text_b is None:
        raise click.ClickException('Source file missing.')
    click.echo(
        _make_diff(
            text_a,
            text_b,
            _fmt_hash(prev.source_hash, full_hash, 8),
            _fmt_hash(latest.source_hash, full_hash, 8),
            expand,
            color,
        ),
        nl=False,
    )


# -----------------------------------------------------------------------------
# versions  (cross-cutting: code groups + dep-tree snapshots per version group)
# -----------------------------------------------------------------------------


def _echo_version_group(vi: Version, vreg: VersionRegistry, full_hash: bool = False) -> None:
    click.echo(click.style(vreg.label(vi), bold=True))

    if vi.class_names:
        click.echo('  Classes')
        col = max(len(cn) for cn in vi.class_names)
        for cn, sh in zip(vi.class_names, vi.source_hashes):
            click.echo(f'    {cn:<{col}}  {_fmt_hash(sh, full_hash)}')

    snapshot_hashes = sorted(vreg.dep_hashes_for_version(vi))
    tree_rows = []
    for dep_hash in snapshot_hashes:
        tree = vreg.tree_registry.get_snapshot_from_hash(dep_hash)
        if tree is None:
            continue
        root_class = tree.root_class or '?'
        n = len(tree.nodes)
        tree_rows.append((root_class, n, dep_hash))
    if tree_rows:
        click.echo('  Trees')
        col = max(len(r[0]) for r in tree_rows)
        for root_class, n, dep_hash in tree_rows:
            n_label = f'{n} {"class" if n == 1 else "classes"}'
            click.echo(f'    {root_class:<{col}}  {n_label:<12}  {_fmt_hash(dep_hash, full_hash)}')

    click.echo()


@cli.command('versions')
@_registry_option
@_full_hash_option
@click.option('--class', 'class_name', default=None, help='Show version history for a specific class.')
def versions_cmd(registry_path: str | None, full_hash: bool, class_name: str | None) -> None:
    """Show the full version timeline: groups, changed classes, and dep-tree snapshots.

    With --class, show that class's full snapshot history instead (mirrors the
    browser's per-class Versions card).
    """
    root = _apply_registry_path(registry_path)
    vreg = VersionRegistry(root)

    if class_name:
        src = vreg.source_registry
        states = src.get_states(class_name)
        if not states:
            raise click.ClickException(f'Class {class_name!r} not found.')
        sorted_states = sorted(states, key=lambda s: s.registered_at, reverse=True)
        click.echo(f'{class_name}  ({len(sorted_states)} {"state" if len(sorted_states) == 1 else "states"})')
        click.echo()
        for s in sorted_states:
            change_marker = '*' if src.is_version_change(s) else ' '
            click.echo(f'  {change_marker} {_fmt_mtime(s.registered_at)}  {_fmt_hash(s.source_hash, full_hash)}')
        return

    if not vreg.versions:
        click.echo('No version groups found.')
        return

    for vi in vreg.versions:
        _echo_version_group(vi, vreg, full_hash=full_hash)


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


@cli.group()
def snapshot():
    """Inspect the .source/ dependency-tree snapshot registry."""


@snapshot.command('list')
@_registry_option
@_full_hash_option
def snapshot_list(registry_path: str | None, full_hash: bool) -> None:
    """List all dep-tree snapshots."""
    trees = TreeRegistry(_apply_registry_path(registry_path))
    hashes = sorted(trees.dependency_hashes)
    if not hashes:
        click.echo('No snapshots found.')
        return
    for dep_hash in hashes:
        tree = trees.get_snapshot_from_hash(dep_hash)
        if tree is None:
            continue
        n = len(tree.nodes)
        root_class = tree.root_class or '?'
        click.echo(f'{root_class} ({n} {"class" if n == 1 else "classes"})  {_fmt_hash(dep_hash, full_hash)}')


@snapshot.command('show')
@_registry_option
@_full_hash_option
@click.option('--hash', 'dep_hash', required=True, help='Dep-tree hash to inspect.')
@click.option('--json', 'as_json', is_flag=True, default=False, help='Print raw tree.json.')
def snapshot_show(registry_path: str | None, full_hash: bool, dep_hash: str, as_json: bool) -> None:
    """Show contents of a dep-tree snapshot."""
    trees = TreeRegistry(_apply_registry_path(registry_path))
    dep_hash = _resolve_dep_hash(trees, dep_hash)
    tree = trees.get_snapshot_from_hash(dep_hash)
    if tree is None:
        raise click.ClickException(f'Dep hash {dep_hash!r} not found.')

    if as_json:
        click.echo(json.dumps(dataclasses.asdict(tree), indent=4))
        return

    click.echo(f'Root       {tree.root_class or "?"}')
    click.echo(f'Classes    {len(tree.nodes)}')
    click.echo(f'Dep hash   {dep_hash}')
    if tree.nodes:
        click.echo()
        click.echo(f'  {"Class":<40}  {"Type":<12}  Source hash')
        for class_name in sorted(tree.nodes):
            src_hash = tree.get_source_hash(class_name) or ''
            obj_type = tree.get_object_type(class_name) or ''
            click.echo(f'  {class_name:<40}  {obj_type:<12}  {_fmt_hash(src_hash, full_hash)}')

    call_deps = tree.get_call_dependencies()
    inh_deps = tree.get_inheritance_dependencies()
    if call_deps:
        click.echo()
        click.echo(f'Call deps       {", ".join(call_deps)}')
    if inh_deps:
        click.echo(f'Inherited deps  {", ".join(inh_deps)}')


@snapshot.command('version')
@_registry_option
@click.option('--hash', 'dep_hash', required=True, help='Dep-tree hash to look up.')
def snapshot_version(registry_path: str | None, dep_hash: str) -> None:
    """Show which version group a dep-tree snapshot belongs to."""
    root = _apply_registry_path(registry_path)
    vreg = VersionRegistry(root)
    dep_hash = _resolve_dep_hash(vreg.tree_registry, dep_hash)
    tree = vreg.tree_registry.get_snapshot_from_hash(dep_hash)
    if tree is None:
        raise click.ClickException(f'Dep hash {dep_hash!r} not found.')

    match = vreg.version_for_dep_hash(dep_hash)
    if match:
        click.echo(vreg.label(match))
        click.echo()
        click.echo(f'Classes  {", ".join(match.class_names)}')
    else:
        click.echo('(not assigned to any version group)')


# ---------------------------------------------------------------------------
# classes
# ---------------------------------------------------------------------------


@cli.command('classes')
@_import_options
@click.option(
    '--type',
    'type_filter',
    default=None,
    help='Filter to one object type, e.g. DATA or FIGURE.',
)
@click.option('--hide-current', is_flag=True, default=False, help='Omit classes with no staleness.')
@_full_hash_option
@_registry_option
def classes_cmd(
    do_import: bool,
    verbose_import: bool,
    type_filter: str | None,
    hide_current: bool,
    full_hash: bool,
    registry_path: str | None,
) -> None:
    """Show staleness for all loaded and known classes."""
    _apply_registry_path(registry_path)

    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')

    vreg = VersionRegistry()
    ereg = EntryRegistry()
    src = SourceRegistry()
    classes = discover_loaded_classes()
    classes = merge_unloaded_classes(classes, ereg, version_registry=vreg, src=src)

    # Apply type filter (case-insensitive prefix match against object_type)
    if type_filter:
        tf = type_filter.upper()
        classes = {k: v for k, v in classes.items() if (v.object_type or '').upper() == tf}

    # Apply hide-current filter
    if hide_current:
        classes = {k: v for k, v in classes.items() if v.source_stale or v.deps_stale}

    if not classes:
        click.echo('No classes found.')
        return

    # Build source-hash lookup: class_name -> stored source hash
    stored_hashes: dict[str, str] = {}
    for class_name in classes:
        state = src.get_latest_state_for_class(class_name)
        if state:
            stored_hashes[class_name] = state.source_hash

    # Group by object type — Data first, then Figure, then others alphabetically
    by_type: dict[str, list[str]] = {}
    for class_name, info in classes.items():
        by_type.setdefault(info.object_type or 'Unknown', []).append(class_name)

    type_order = ['Data', 'Figure']
    other_types = sorted(t for t in by_type if t not in type_order)
    ordered_types = [t for t in type_order if t in by_type] + other_types

    any_stale = False
    for object_type in ordered_types:
        class_names = sorted(by_type[object_type])
        click.echo(f'── {object_type} ({len(class_names)}) ──')
        col = max(len(cn) for cn in class_names)
        for class_name in class_names:
            info = classes[class_name]
            if info.source_stale:
                any_stale = True
                indicator = click.style('C', fg='red')
            elif info.deps_stale:
                any_stale = True
                indicator = click.style('D', fg='yellow')
            elif not info.loaded:
                indicator = click.style('N', fg='bright_black')
            else:
                indicator = ' '
            loaded_label = 'loaded' if info.loaded else 'unloaded'
            src_hash = stored_hashes.get(class_name)
            src_str = f'src:{_fmt_hash(src_hash, full_hash)}' if src_hash else 'src:—'
            all_deps = info.call_dependency_names + info.inheritance_dependency_names
            deps_str = f'deps:[{", ".join(all_deps)}]' if all_deps else 'deps:—'
            click.echo(f'  {indicator} {class_name:<{col}}  {loaded_label}  {src_str}  {deps_str}')
        click.echo()

    if any_stale:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


@cli.group()
@click.option('--full-hash', 'full_hash', is_flag=True, default=False, help='Print full hashes.')
@click.pass_context
def entry(ctx: click.Context, full_hash: bool) -> None:
    """Inspect cache entries."""
    ctx.ensure_object(dict)
    ctx.obj['full_hash'] = full_hash


def _stale_indicator(rec, vreg: VersionRegistry | None = None) -> str:
    """Single-char staleness marker: F=format stale, S=dep stale, N=not loaded, blank=ok."""
    if rec.format_version != FORMAT_VERSION:
        return 'F'
    obj_cls = TrackedObject.find_object_class(rec.class_name)
    if rec.dependency_tree_hash:
        if obj_cls is not None:
            dep_stale = obj_cls.get_dependency_tree_hash() != rec.dependency_tree_hash
        else:
            dep_stale = (vreg or VersionRegistry()).is_dependency_hash_stale(rec.dependency_tree_hash)
        if dep_stale:
            return 'S'
    if obj_cls is None:
        return 'N'
    return ' '




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


def _parse_filter_expr(expr: str) -> list[tuple[list, str]]:
    """Parse a single --filter expression into (filters, logic_mode) pairs.

    Syntax:
      [!] [target:]value  (| [!] [target:]value)*

    Examples
    --------
      "RF"                       → ALL contains RF, AND
      "value:RF|value:KMEANS"    → value contains RF OR value contains KMEANS
      "!ElevationLoader"         → ALL not-contains ElevationLoader
      "!RF|KMEANS"               → NOT(RF OR KMEANS)  [! applies to whole OR group]

    Returns a list of (filters, logic_mode) to be ANDed together by the caller.
    """
    _target_map = {
        'class': FilterTarget.CLASS,
        'crs': FilterTarget.CRS,
        'key': FilterTarget.KEY,
        'value': FilterTarget.VALUE,
        'path': FilterTarget.PATH,
        'key_group': FilterTarget.KEY_GROUP,
    }

    expr = expr.strip()
    negate = False
    if expr.startswith('!'):
        negate = True
        expr = expr[1:].strip()

    parts = [p.strip() for p in expr.split('|') if p.strip()]
    if not parts:
        return []

    filters = []
    for part in parts:
        if ':' in part:
            raw_target, _, value = part.partition(':')
            target = _target_map.get(raw_target.strip().lower(), FilterTarget.ALL)
            value = value.strip()
        else:
            target = FilterTarget.ALL
            value = part.strip()
        filters.append(Filter(target=target, operator=FilterOperator.CONTAINS, value=value))

    logic_mode = 'NOT' if negate else ('OR' if len(filters) > 1 else 'AND')
    return [(filters, logic_mode)]


def _entry_matches_all_exprs(
    class_name: str,
    entry_info,
    parsed_exprs: list[tuple[list, str]],
) -> bool:
    """Return True if entry satisfies all parsed filter expressions (AND between exprs)."""
    from pygeodata.catalog.filters import entry_matches_filters

    return all(
        entry_matches_filters(class_name, entry_info, filters, logic_mode) for filters, logic_mode in parsed_exprs
    )


def _format_param_rows(param_rows) -> list[str]:
    """Format ParamRow list into lines (for buffered or paged output)."""
    if not param_rows:
        return []
    col = max((len(r.final_key) for r in param_rows if r.depth == 0), default=8)
    col = max(col, 8)
    lines = []
    for row in param_rows:
        pad = '  ' * row.depth
        w = max(col - row.depth * 2, 4)
        if row.value_type == 'data_ref':
            key_str = click.style(f'→ {row.final_key}', fg='cyan')
            lines.append(f'    {pad}{key_str:<{w + 2}}  {row.value_text}')
        else:
            key_str = click.style(f'{row.final_key}', fg='bright_black')
            lines.append(f'    {pad}{key_str:<{w}}  {row.value_text}')
    return lines


def _echo_param_rows(param_rows) -> None:
    for line in _format_param_rows(param_rows):
        click.echo(line)


@entry.command('list')
@_import_options
@click.option('--class', 'class_name', default=None, help='Filter by class name.')
@click.option('--hide-stale', is_flag=True, default=False, help='Omit stale entries.')
@click.option('--params', 'show_params', is_flag=True, default=False, help='Show parsed parameters under each entry.')
@click.option(
    '--filter',
    'filters',
    multiple=True,
    metavar='EXPR',
    help=(
        'Filter entries. Syntax: [target:]value, optionally prefixed with ! (NOT) '
        'or using | between terms (OR). Multiple --filter flags are ANDed. '
        'Targets: class, crs, key, value, path, key_group (default: all). '
        'Examples: --filter RF  --filter "value:RF|value:KMEANS"  --filter "!ElevationLoader"'
    ),
)
@click.option('--full-hash', 'full_hash_sub', is_flag=True, default=False, help='Print full hashes.')
@click.option('--no-pager', is_flag=True, default=False, help='Disable pager (print directly to stdout).')
@click.pass_context
def entry_list(
    ctx: click.Context,
    do_import: bool,
    verbose_import: bool,
    class_name: str | None,
    hide_stale: bool,
    show_params: bool,
    filters: tuple[str, ...],
    full_hash_sub: bool,
    no_pager: bool,
) -> None:
    """List cache entries."""
    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub

    parsed_exprs = [pair for f in filters for pair in _parse_filter_expr(f)]
    needs_enrich = bool(parsed_exprs) or show_params

    reg = EntryRegistry()
    records = reg.records

    if not records:
        click.echo('No entries found.')
        return

    vreg = VersionRegistry()
    output_rows = []
    for state_hash, rec in sorted(records.items(), key=lambda kv: (kv[1].class_name or '', kv[0])):
        if class_name and rec.class_name != class_name:
            continue
        stale = _stale_indicator(rec, vreg)
        if hide_stale and stale.strip():
            continue

        # Fast pre-filter: cheap text scan before paying for full enrichment.
        params_path = CachePathConstructor.from_path(Path(rec.hash_path)).params_path if rec.hash_path else None
        if parsed_exprs and params_path:
            try:
                params_blob = params_path.read_text(encoding='utf-8').lower() if params_path.exists() else ''
            except OSError:
                params_blob = ''
            if not prescan_text(rec.class_name or '', params_blob, parsed_exprs):
                continue

        entry_info = None
        if needs_enrich and params_path:
            try:
                entry_info = _enrich_params_path(params_path)
            except Exception:  # noqa: BLE001
                pass

        if parsed_exprs:
            if entry_info is None:
                continue
            if not _entry_matches_all_exprs(rec.class_name or '', entry_info, parsed_exprs):
                continue

        if entry_info is not None:
            spec_info = entry_info.spec
        else:
            spec_path = CachePathConstructor.from_path(Path(rec.hash_path)).spec_path if rec.hash_path else None
            spec_dict = json.loads(spec_path.read_text(encoding='utf-8')) if spec_path and spec_path.exists() else {}
            try:
                spec_info = SpecInfo.from_spec(SpatialSpec.from_dict(spec_dict)) if spec_dict else SpecInfo()
            except Exception:  # noqa: BLE001
                spec_info = SpecInfo()
        crs_str = spec_info.crs or ''
        res_str = spec_info.resolution or ''
        bounds_str = spec_info.bounds_display or 'None'
        output_rows.append(
            (
                stale,
                rec.class_name or '?',
                rec.object_type or '',
                crs_str,
                res_str,
                bounds_str,
                _fmt_hash(state_hash, full_hash),
                entry_info,
            ),
        )

    if not output_rows:
        click.echo('No entries match.')
        return

    col_class = max(len(r[1]) for r in output_rows)
    col_type = max(len(r[2]) for r in output_rows)
    col_crs = max((len(r[3]) for r in output_rows), default=0)
    col_res = max((len(r[4]) for r in output_rows), default=0)
    col_bounds = max((len(r[5]) for r in output_rows), default=0)

    lines = []
    for stale, cls, obj_type, crs, res, bounds, h, entry_info in output_rows:
        if stale == 'F':
            indicator = click.style('F', fg='red')
        elif stale == 'S':
            indicator = click.style('S', fg='yellow')
        elif stale == 'N':
            indicator = click.style('N', fg='bright_black')
        else:
            indicator = ' '
        lines.append(
            f'{indicator} {cls:<{col_class}}  {obj_type:<{col_type}}  {crs:<{col_crs}}  {res:<{col_res}}  {bounds:<{col_bounds}}  {h}',
        )
        if (show_params or parsed_exprs) and entry_info is not None and entry_info.rows:
            lines.extend(_format_param_rows(entry_info.rows))

    output = '\n'.join(lines)
    if no_pager or not sys.stdout.isatty():
        click.echo(output)
    else:
        click.echo_via_pager(output + '\n')


@entry.command('show')
@click.argument('hash_prefix')
@_import_options
@click.option('--no-params', 'show_params', is_flag=True, flag_value=False, default=True, help='Omit params JSON.')
@click.option('--full-hash', 'full_hash_sub', is_flag=True, default=False, help='Print full hashes.')
@click.pass_context
def entry_show(
    ctx: click.Context,
    hash_prefix: str,
    do_import: bool,
    verbose_import: bool,
    show_params: bool,
    full_hash_sub: bool,
) -> None:
    """Show detail for one entry (partial hash ok)."""
    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub
    reg = EntryRegistry()
    state_hash, rec = _resolve_record(reg, hash_prefix)

    resolver = CachePathConstructor.from_path(Path(rec.hash_path)) if rec.hash_path else None

    entry_info = None
    if resolver:
        try:
            entry_info = _enrich_params_path(resolver.params_path)
        except Exception:  # noqa: BLE001
            pass

    stale = _stale_indicator(rec)
    stale_label = {'S': 'dep stale', 'F': 'format stale', '?': 'unknown (class not loaded)'}.get(stale, 'ok')

    click.echo(f'Class        {rec.class_name}')
    click.echo(f'Type         {rec.object_type or "unknown"}')
    click.echo(f'State hash   {_fmt_hash(state_hash, full_hash)}')
    if rec.instance_hash:
        click.echo(f'Instance     {_fmt_hash(rec.instance_hash, full_hash)}')
    if rec.dependency_tree_hash:
        click.echo(f'Dep hash     {_fmt_hash(rec.dependency_tree_hash, full_hash)}')
    click.echo(f'Staleness    {stale_label}')

    if entry_info:
        spec = entry_info.spec
        if spec.crs or spec.resolution or spec.shape:
            click.echo()
            if spec.crs:
                click.echo(f'CRS          {spec.crs}')
            click.echo(f'Resolution   {spec.resolution_display or "None"}')
            if spec.shape:
                click.echo(f'Shape        {spec.shape}')
            click.echo(f'Bounds       {spec.bounds_display or "None"}')

    if rec.co_output_hashes:
        click.echo()
        click.echo(f'Co-outputs   {", ".join(_fmt_hash(h, full_hash) for h in rec.co_output_hashes)}')

    if resolver:
        click.echo()
        if entry_info and entry_info.primary_file:
            click.echo(f'Output       {entry_info.primary_file.path}')
        click.echo(f'Params path  {resolver.params_path}')
        click.echo(f'Hash path    {rec.hash_path}')
        if resolver.spec_path.exists():
            click.echo(f'Spec path    {resolver.spec_path}')

    if show_params and entry_info and entry_info.params:
        click.echo()
        click.echo('Params')
        click.echo(json.dumps(entry_info.params, indent=2))
