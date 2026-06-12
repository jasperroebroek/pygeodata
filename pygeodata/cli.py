import importlib.util
import json
import shutil
import sys
import tarfile
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

import click

from pygeodata.config import get_config
from pygeodata.registry import SourceRegistry, TreeRegistry
from pygeodata.versioning import VersionInfo, VersionRegistry


def _fmt_mtime(mtime: str) -> str:
    try:
        return datetime.fromisoformat(mtime).strftime('%Y-%m-%d %H:%M')
    except (ValueError, AttributeError):
        return mtime


@click.group()
def cli():
    pass


def _cache_root_from_tar(tar: tarfile.TarFile, hash_dir: str) -> Path | None:
    """Read OBJECT_TYPE from the hash.json and return the matching Artifact subclass cache root."""
    from pygeodata.config import JSONKeys
    from pygeodata.data import Data
    from pygeodata.figure import Figure
    family_by_name = {'Data': Data, 'Figure': Figure}
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
            family = family_by_name.get(object_type)
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


def _import_project_modules(root: Path, verbose: bool = False) -> int:
    """Import all .py files under root, silently skipping failures. Returns import count."""
    if root not in sys.path:
        sys.path.insert(0, str(root))
    _skip_dirs = {'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test', 'cache'}
    count = 0
    for py_file in sorted(root.rglob('*.py')):
        parts = py_file.relative_to(root).parts
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
                if verbose:
                    click.echo(f'  imported {module_name}')
        except Exception as e:  # noqa: BLE001
            if verbose:
                click.echo(f'  skip {module_name}: {e}')
    return count


@cli.command('browse')
@click.option('--port', default=0, show_default=True, help='Port to listen on (0 = random free port).')
@click.option(
    '--import-all',
    'do_import',
    is_flag=True,
    default=False,
    help='Import all .py files from the current directory to populate the class registry.',
)
@click.option(
    '--verbose-import',
    is_flag=True,
    default=False,
    help='Print each module imported (and failures) when used with --import-all.',
)
def browse(port: int, do_import: bool, verbose_import: bool) -> None:
    """Launch the registry browser."""
    if do_import:
        root = Path.cwd()
        n = _import_project_modules(root, verbose=verbose_import)
        click.echo(f'Imported {n} module(s) from {root}')

    from pygeodata.registry_browser.serve import open_registry_browser
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
    from pygeodata.cache import clean_cache
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


def _get_root(registry_path: str | None) -> Path:
    return Path(registry_path) if registry_path else Path(get_config().path_registry)


def _fmt_hash(h: str, full: bool, width: int = 12) -> str:
    return h if full else h[:width]


def _resolve_source_hash(reg: SourceRegistry, prefix: str) -> str:
    """Resolve a source-hash prefix to a full hash, or raise ClickException."""
    matches = [h for h in reg._hash_index if h.startswith(prefix)]
    if not matches:
        raise click.ClickException(f'No source hash found matching {prefix!r}.')
    if len(matches) > 1:
        raise click.ClickException(
            f'{len(matches)} source hashes match {prefix!r}: ' + ', '.join(h[:12] for h in matches)
        )
    return matches[0]


def _resolve_dep_hash(trees: TreeRegistry, prefix: str) -> str:
    """Resolve a dep-hash prefix to a full hash, or raise ClickException."""
    matches = [h for h in trees.dep_hashes if h.startswith(prefix)]
    if not matches:
        raise click.ClickException(f'No dep-tree hash found matching {prefix!r}.')
    if len(matches) > 1:
        raise click.ClickException(
            f'{len(matches)} dep hashes match {prefix!r}: ' + ', '.join(h[:12] for h in matches)
        )
    return matches[0]


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
    reg = SourceRegistry(_get_root(registry_path))

    by_type: dict[str, list[str]] = {}
    for class_name in sorted(reg.class_names):
        states = sorted(reg.get_states(class_name), key=lambda s: s.registered_at)
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
            states = sorted(reg.get_states(class_name), key=lambda s: s.registered_at, reverse=True)
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

    reg = SourceRegistry(_get_root(registry_path))

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
@click.option('--class', 'class_name', default=None, help='Filter to groups containing this class.')
@click.option('--hash', 'source_hash', default=None, help='Which version group owns this source hash.')
def code_versions(registry_path: str | None, class_name: str | None, source_hash: str | None) -> None:
    """Show version groups, newest-first."""
    vreg = VersionRegistry(_get_root(registry_path))

    if source_hash:
        source_hash = _resolve_source_hash(vreg._source_registry, source_hash)
        identity = vreg.version_mtime_for_source_hash(source_hash)
        if identity is None:
            raise click.ClickException(f'Hash {source_hash!r} not found.')
        match = next((vi for vi in vreg.version_groups if vi.mtime == identity), None)
        click.echo(match.label if match else 'unknown')
        return

    if not vreg.version_groups:
        click.echo('No version groups found.')
        return

    for vi in vreg.version_groups:
        if class_name and class_name not in vi.class_names:
            continue
        click.echo(vi.label)


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

    reg = SourceRegistry(_get_root(registry_path))

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
    kwargs = {} if expand else {"n": 3}
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
    state = reg.get_state_by_hash(source_hash)
    if state is None:
        raise click.ClickException(f'Hash {source_hash!r} not found.')
    states = sorted(reg.get_states(state.class_name), key=lambda s: s.registered_at)
    idx = next((i for i, s in enumerate(states) if s.source_hash == source_hash), None)
    if idx is None or idx == 0:
        raise click.ClickException('No previous snapshot to diff against.')
    prev_hash = states[idx - 1].source_hash
    text_a = reg.get_source(prev_hash)
    text_b = reg.get_source(source_hash)
    if text_a is None or text_b is None:
        raise click.ClickException('Source file missing.')
    click.echo(
        _make_diff(
            text_a, text_b, _fmt_hash(prev_hash, full_hash, 8), _fmt_hash(source_hash, full_hash, 8), expand, color
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
    states = sorted(reg.get_states(class_name), key=lambda s: s.registered_at)
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


# ---------------------------------------------------------------------------
# versions  (cross-cutting: code groups + dep-tree snapshots per version group)
# ---------------------------------------------------------------------------



def _echo_version_group(vi: VersionInfo, vreg: VersionRegistry, full_hash: bool = False) -> None:
    click.echo(click.style(vi.label, bold=True))

    if vi.class_names:
        click.echo('  Classes')
        col = max(len(cn) for cn in vi.class_names)
        for cn, sh in zip(vi.class_names, vi.source_hashes):
            click.echo(f'    {cn:<{col}}  {_fmt_hash(sh, full_hash)}')

    snapshot_hashes = sorted(
        dh for dh, identity in vreg.dep_hash_to_mtime.items() if identity == vi.mtime
    )
    tree_rows = []
    for dep_hash in snapshot_hashes:
        tree = vreg._tree_registry.get_snapshot(dep_hash)
        if tree is None:
            continue
        root_class = next(iter(tree.tree), '?')
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
    """Show version groups with classes and dep-tree snapshots, newest-first.

    With --class, show that class's full state history instead (mirrors the
    browser's per-class Versions card).
    """
    vreg = VersionRegistry(_get_root(registry_path))

    if class_name:
        code_groups = vreg.code_groups
        states = code_groups.get(class_name)
        if not states:
            raise click.ClickException(f'Class {class_name!r} not found.')
        sorted_states = sorted(states, key=lambda e: e['mtime'], reverse=True)
        click.echo(f'{class_name}  ({len(sorted_states)} {"state" if len(sorted_states) == 1 else "states"})')
        click.echo()
        for e in sorted_states:
            change_marker = '*' if e['is_version_change'] else ' '
            click.echo(f'  {change_marker} {_fmt_mtime(e["mtime"])}  {_fmt_hash(e["source_hash"], full_hash)}')
        return

    if not vreg.version_groups:
        click.echo('No version groups found.')
        return

    for vi in vreg.version_groups:
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
    trees = TreeRegistry(_get_root(registry_path))
    hashes = sorted(trees.dep_hashes)
    if not hashes:
        click.echo('No snapshots found.')
        return
    for dep_hash in hashes:
        tree = trees.get_snapshot(dep_hash)
        if tree is None:
            continue
        n = len(tree.nodes)
        root_class = next(iter(tree.tree), '?')
        click.echo(f'{root_class} ({n} {"class" if n == 1 else "classes"})  {_fmt_hash(dep_hash, full_hash)}')


@snapshot.command('show')
@_registry_option
@_full_hash_option
@click.option('--hash', 'dep_hash', required=True, help='Dep-tree hash to inspect.')
@click.option('--json', 'as_json', is_flag=True, default=False, help='Print raw tree.json.')
def snapshot_show(registry_path: str | None, full_hash: bool, dep_hash: str, as_json: bool) -> None:
    """Show contents of a dep-tree snapshot."""
    trees = TreeRegistry(_get_root(registry_path))
    dep_hash = _resolve_dep_hash(trees, dep_hash)
    tree = trees.get_snapshot(dep_hash)
    if tree is None:
        raise click.ClickException(f'Dep hash {dep_hash!r} not found.')

    if as_json:
        click.echo(json.dumps(tree.to_dict(), indent=4))
        return

    click.echo(f'Root       {next(iter(tree.tree), "?")}')
    click.echo(f'Classes    {len(tree.nodes)}')
    click.echo(f'Dep hash   {dep_hash}')
    if tree.nodes:
        click.echo()
        click.echo(f'  {"Class":<40}  {"Type":<12}  Source hash')
        for class_name in sorted(tree.nodes):
            src_hash = tree.get_source_hash(class_name) or ''
            obj_type = tree.get_object_type(class_name) or ''
            click.echo(f'  {class_name:<40}  {obj_type:<12}  {_fmt_hash(src_hash, full_hash)}')

    call_deps = tree.get_call_deps()
    inh_deps = tree.get_inheritance_deps()
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
    vreg = VersionRegistry(_get_root(registry_path))
    dep_hash = _resolve_dep_hash(vreg._tree_registry, dep_hash)
    tree = vreg._tree_registry.get_snapshot(dep_hash)
    if tree is None:
        raise click.ClickException(f'Dep hash {dep_hash!r} not found.')

    identity = vreg.version_mtime_for_dep_hash(dep_hash)
    match = next((vi for vi in vreg.version_groups if vi.mtime == identity), None)
    if match:
        click.echo(match.label)
        click.echo()
        click.echo(f'Classes  {", ".join(match.class_names)}')
    else:
        click.echo('Initial')


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


@cli.group()
@click.option(
    '--import-all',
    'do_import',
    is_flag=True,
    default=False,
    help='Import all .py files from the current directory to resolve staleness.',
)
@click.option('--verbose-import', is_flag=True, default=False, help='Print each module imported.')
@click.option('--full-hash', 'full_hash', is_flag=True, default=False, help='Print full hashes.')
@click.pass_context
def entry(ctx: click.Context, do_import: bool, verbose_import: bool, full_hash: bool) -> None:
    """Inspect cache entries."""
    ctx.ensure_object(dict)
    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
    ctx.obj['full_hash'] = full_hash


def _stale_indicator(rec) -> str:
    """Single-char staleness marker: S=dep stale, F=format stale, ?=unknown, blank=ok."""
    from pygeodata.config import FORMAT_VERSION
    if rec.dep_hash_stale is True:
        return 'S'
    if rec.format_version != FORMAT_VERSION:
        return 'F'
    if rec.dep_hash_stale is None and rec.dep_hash:
        return '?'
    return ' '


def _fmt_coord(v: float, pos: str, neg: str) -> str:
    return f'{abs(v)}° {pos if v >= 0 else neg}'


def _fmt_bounds(bounds_latlon) -> str:
    if not bounds_latlon:
        return ''
    try:
        lat_min, lon_min, lat_max, lon_max = [float(v) for v in bounds_latlon]
        sw = f'{_fmt_coord(lat_min, "N", "S")}, {_fmt_coord(lon_min, "E", "W")}'
        ne = f'{_fmt_coord(lat_max, "N", "S")}, {_fmt_coord(lon_max, "E", "W")}'
        return f'{sw} → {ne}'
    except (TypeError, ValueError):
        return str(bounds_latlon)


def _spec_info_from_raw(spec: dict):
    """Build a SpatialSpec from raw spec JSON, silently returning None on failure."""
    from pygeodata.spec import SpatialSpec
    try:
        return SpatialSpec.from_dict(spec)
    except Exception:
        return None


def _read_json(path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding='utf-8')) if path and Path(path).exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_record(reg, hash_prefix: str):
    """Return (state_hash, EntryRecord) for a partial or full hash. Raises ClickException on ambiguity/miss."""
    matches = [(h, r) for h, r in reg.records.items() if h.startswith(hash_prefix)]
    if not matches:
        raise click.ClickException(f'No entry found matching {hash_prefix!r}.')
    if len(matches) > 1:
        raise click.ClickException(
            f'{len(matches)} entries match {hash_prefix!r}: ' + ', '.join(h[:12] for h, _ in matches)
        )
    return matches[0]


@entry.command('list')
@click.option('--class', 'class_name', default=None, help='Filter by class name.')
@click.option('--hide-stale', is_flag=True, default=False, help='Omit stale entries.')
@click.option('--full-hash', 'full_hash_sub', is_flag=True, default=False, help='Print full hashes.')
@click.pass_context
def entry_list(ctx: click.Context, class_name: str | None, hide_stale: bool, full_hash_sub: bool) -> None:
    """List cache entries."""
    from pygeodata.paths import CachePathResolver
    from pygeodata.registry import EntryRegistry
    from pygeodata.spec import format_resolution
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub
    reg = EntryRegistry.instance()
    records = reg.records

    if not records:
        click.echo('No entries found.')
        return

    rows = []
    for state_hash, rec in sorted(records.items(), key=lambda kv: (kv[1].class_name or '', kv[0])):
        if class_name and rec.class_name != class_name:
            continue
        stale = _stale_indicator(rec)
        if hide_stale and stale.strip():
            continue
        spec_raw = _read_json(CachePathResolver.from_path(Path(rec.hash_path)).spec_path if rec.hash_path else None)
        spatial = _spec_info_from_raw(spec_raw)
        crs_str = spec_raw.get('crs') or ''
        res_str = format_resolution(list(spatial.resolution), spatial.crs) if spatial and spatial.transform is not None else ''
        bounds_str = _fmt_bounds(spatial.bounds_latlon) if spatial else ''
        rows.append((stale, rec.class_name or '?', rec.object_type or '', crs_str, res_str, bounds_str, _fmt_hash(state_hash, full_hash)))

    if not rows:
        click.echo('No entries match.')
        return

    col_class = max(len(r[1]) for r in rows)
    col_type = max(len(r[2]) for r in rows)
    col_crs = max((len(r[3]) for r in rows), default=0)
    col_res = max((len(r[4]) for r in rows), default=0)
    col_bounds = max((len(r[5]) for r in rows), default=0)

    for stale, cls, obj_type, crs, res, bounds, h in rows:
        indicator = click.style(stale, fg='yellow') if stale.strip() else ' '
        click.echo(f'{indicator} {cls:<{col_class}}  {obj_type:<{col_type}}  {crs:<{col_crs}}  {res:<{col_res}}  {bounds:<{col_bounds}}  {h}')


@entry.command('show')
@click.argument('hash_prefix')
@click.option('--no-params', 'show_params', is_flag=True, flag_value=False, default=True, help='Omit params JSON.')
@click.option('--full-hash', 'full_hash_sub', is_flag=True, default=False, help='Print full hashes.')
@click.pass_context
def entry_show(ctx: click.Context, hash_prefix: str, show_params: bool, full_hash_sub: bool) -> None:
    """Show detail for one entry (partial hash ok)."""
    from pygeodata.paths import CachePathResolver
    from pygeodata.registry import EntryRegistry
    from pygeodata.spec import format_resolution
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub
    reg = EntryRegistry.instance()
    state_hash, rec = _resolve_record(reg, hash_prefix)

    resolver = CachePathResolver.from_path(Path(rec.hash_path)) if rec.hash_path else None
    spec = _read_json(resolver.spec_path if resolver else None)

    stale = _stale_indicator(rec)
    stale_label = {'S': 'dep stale', 'F': 'format stale', '?': 'unknown (class not loaded)'}.get(stale, 'ok')

    click.echo(f'Class        {rec.class_name}')
    click.echo(f'Type         {rec.object_type or "unknown"}')
    click.echo(f'State hash   {_fmt_hash(state_hash, full_hash)}')
    if rec.instance_hash:
        click.echo(f'Instance     {_fmt_hash(rec.instance_hash, full_hash)}')
    if rec.dep_hash:
        click.echo(f'Dep hash     {_fmt_hash(rec.dep_hash, full_hash)}')
    click.echo(f'Staleness    {stale_label}')

    spatial = _spec_info_from_raw(spec)
    if spatial or spec:
        click.echo()
        crs_str = spec.get('crs') or ''
        if crs_str:
            click.echo(f'CRS          {crs_str}')
        if spatial and spatial.transform is not None:
            click.echo(f'Resolution   {format_resolution(list(spatial.resolution), spatial.crs)}')
            click.echo(f'Shape        {list(spatial.shape)}')
            bounds_latlon = spatial.bounds_latlon
            if bounds_latlon:
                click.echo(f'Bounds       {_fmt_bounds(bounds_latlon)}')
        elif spec.get('shape'):
            click.echo(f'Shape        {spec["shape"]}')

    if rec.co_output_hashes:
        click.echo()
        click.echo(f'Co-outputs   {", ".join(_fmt_hash(h, full_hash) for h in rec.co_output_hashes)}')

    if resolver:
        click.echo()
        click.echo(f'Params path  {resolver.params_path}')
        click.echo(f'Hash path    {rec.hash_path}')
        if resolver.spec_path.exists():
            click.echo(f'Spec path    {resolver.spec_path}')

    if show_params:
        params = _read_json(resolver.params_path if resolver else None)
        if params:
            click.echo()
            click.echo('Params')
            click.echo(json.dumps(params, indent=2))
