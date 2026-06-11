import importlib.util
import json
import shutil
import sys
import tarfile
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

import click

from pygeodata.cache import clean_cache
from pygeodata.config import JSONKeys, get_config
from pygeodata.data import Data
from pygeodata.figure import Figure
from pygeodata.registry import SourceRegistry, TreeRegistry
from pygeodata.registry_browser.serve import open_registry_browser
from pygeodata.versioning import (
    build_version_groups,
    snapshot_version_identity,
    source_hash_version_identity,
    version_infos,
)


def _fmt_mtime(mtime: str) -> str:
    try:
        return datetime.fromisoformat(mtime).strftime('%Y-%m-%d %H:%M')
    except (ValueError, AttributeError):
        return mtime


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
        snaps = sorted(reg.get_states(class_name), key=lambda s: s.registered_at)
        object_type = snaps[-1].object_type if snaps else 'Unknown'
        by_type.setdefault(object_type, []).append(class_name)

    if not by_type:
        click.echo('No snapshots found.')
        return

    for object_type in sorted(by_type):
        class_names = by_type[object_type]
        click.echo(f'{object_type} ({len(class_names)} {"class" if len(class_names) == 1 else "classes"})')
        click.echo()
        for class_name in class_names:
            snaps = sorted(reg.get_states(class_name), key=lambda s: s.registered_at, reverse=True)
            n = len(snaps)
            click.echo(f'  {class_name} ({n} {"snapshot" if n == 1 else "snapshots"})')
            if verbose:
                for s in snaps:
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
        state = reg.get_state_by_hash(source_hash)
        if state is None:
            raise click.ClickException(f'Hash {source_hash!r} not found.')
        click.echo(f'Class      {state.class_name}')
        click.echo(f'Registered {_fmt_mtime(state.registered_at)}')
        click.echo(f'Type       {state.object_type}')
        click.echo(f'Hash       {state.source_hash}')
        return

    snaps = reg.get_states(class_name)
    if not snaps:
        raise click.ClickException(f'Class {class_name!r} not found.')
    click.echo(f'{"Registered At":<20}  Hash')
    for s in sorted(snaps, key=lambda s: s.registered_at, reverse=True):
        click.echo(f'{_fmt_mtime(s.registered_at):<20}  {_fmt_hash(s.source_hash, full_hash)}')


@code.command('versions')
@_registry_option
@click.option('--class', 'class_name', default=None, help='Filter to groups containing this class.')
@click.option('--hash', 'source_hash', default=None, help='Which version group owns this source hash.')
def code_versions(registry_path: str | None, class_name: str | None, source_hash: str | None) -> None:
    """Show version groups, newest-first."""
    reg = SourceRegistry(_get_root(registry_path))

    if source_hash:
        state = reg.get_state_by_hash(source_hash)
        if state is None:
            raise click.ClickException(f'Hash {source_hash!r} not found.')
        groups = build_version_groups(version_infos(reg), reg.code_groups_dict())
        identity = source_hash_version_identity(reg, source_hash, state.class_name)
        if identity is None:
            raise click.ClickException(f'Could not resolve version for {source_hash!r}.')
        match = next((g for g in groups if g['mtime'] == identity), None)
        if match:
            click.echo(f'{match["label"]}')
        else:
            click.echo('initial')
        return

    groups = build_version_groups(version_infos(reg), reg.code_groups_dict())
    if not groups:
        click.echo('No version groups found.')
        return

    for g in groups:
        if class_name and class_name not in g.get('class_names', []):
            continue
        click.echo(g['label'])


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
        _echo_diff_hashes(reg, source_hashes[0], source_hashes[1], expand=expand, color=color, full_hash=full_hash)
    elif len(source_hashes) == 1:
        _echo_source_by_hash(reg, source_hashes[0], do_diff, expand=expand, color=color, full_hash=full_hash)
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
    n = None if expand else 3
    lines = list(
        unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            n=n,
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
    snaps = sorted(reg.get_states(class_name), key=lambda s: s.registered_at)
    if not snaps:
        raise click.ClickException(f'Class {class_name!r} not found.')
    if not do_diff:
        text = reg.get_source(snaps[-1].source_hash)
        if text is None:
            raise click.ClickException(f'Source file missing for {class_name!r}.')
        click.echo(text, nl=False)
        return
    if len(snaps) < 2:
        raise click.ClickException(f'{class_name!r} has only one snapshot, nothing to diff.')
    prev, latest = snaps[-2], snaps[-1]
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


def _best_entry_at_cutoff(entries: list[dict], cutoff: str, excl: bool) -> dict | None:
    if cutoff == 'now':
        candidates = entries
    elif excl:
        candidates = [e for e in entries if e['mtime'] < cutoff]
    else:
        candidates = [e for e in entries if e['mtime'] <= cutoff]
    return max(candidates, key=lambda e: e['mtime']) if candidates else None


def _echo_version_group(
    g: dict, code_groups: dict, dep_hash_to_identity: dict, trees: TreeRegistry, full_hash: bool = False
) -> None:
    click.echo(click.style(g['label'], bold=True))

    changed = sorted({cn for cn in g['class_names'] if cn in code_groups})
    if changed:
        click.echo('  Classes')
        col = max(len(cn) for cn in changed)
        for cn in changed:
            best = _best_entry_at_cutoff(code_groups.get(cn, []), g['cutoff_mtime'], g['cutoff_exclusive'])
            src = _fmt_hash(best['source_hash'], full_hash) if best else '?'
            click.echo(f'    {cn:<{col}}  {src}')

    snapshot_hashes = sorted(dh for dh, identity in dep_hash_to_identity.items() if identity == g['mtime'])
    snap_rows = []
    for dep_hash in snapshot_hashes:
        snap = trees.get_snapshot(dep_hash)
        if snap is None:
            continue
        root_class = next(iter(tree.tree), '?')
        n = len(tree.nodes)
        snap_rows.append((root_class, n, dep_hash))
    if snap_rows:
        click.echo('  Trees')
        col = max(len(r[0]) for r in tree_rows)
        for root_class, n, dep_hash in snap_rows:
            n_label = f'{n} {"class" if n == 1 else "classes"}'
            click.echo(f'    {root_class:<{col}}  {n_label:<12}  {_fmt_hash(dep_hash, full_hash)}')

    click.echo()


@cli.command('versions')
@_registry_option
@_full_hash_option
@click.option('--class', 'class_name', default=None, help='Show version history for a specific class.')
def versions_cmd(registry_path: str | None, full_hash: bool, class_name: str | None) -> None:
    """Show version groups with classes and dep-tree snapshots, newest-first.

    With --class, show that class's full snapshot history instead (mirrors the
    browser's per-class Versions card).
    """
    root = _get_root(registry_path)
    reg = SourceRegistry(root)
    code_groups = reg.code_groups_dict()

    if class_name:
        entries = code_groups.get(class_name)
        if not entries:
            raise click.ClickException(f'Class {class_name!r} not found.')
        sorted_entries = sorted(entries, key=lambda e: e['mtime'], reverse=True)
        click.echo(f'{class_name}  ({len(sorted_entries)} {"snapshot" if len(sorted_entries) == 1 else "snapshots"})')
        click.echo()
        for e in sorted_entries:
            change_marker = '*' if e['is_version_change'] else ' '
            click.echo(f'  {change_marker} {_fmt_mtime(e["mtime"])}  {_fmt_hash(e["source_hash"], full_hash)}')
        return

    trees = TreeRegistry(root)
    groups = build_version_groups(version_infos(reg), code_groups)
    if not groups:
        click.echo('No version groups found.')
        return

    dep_hash_to_identity: dict[str, str] = {
        dh: snapshot_version_identity(reg, dh, tree_registry=trees) for dh in trees.dep_hashes
    }

    for g in groups:
        _echo_version_group(g, code_groups, dep_hash_to_identity, trees, full_hash=full_hash)


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
    tree = trees.get_snapshot(dep_hash)
    if tree is None:
        raise click.ClickException(f'Dep hash {dep_hash!r} not found.')

    if as_json:
        click.echo(json.dumps(tree.to_dict(), indent=2))
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
    root = _get_root(registry_path)
    reg = SourceRegistry(root)
    trees = TreeRegistry(root)

    tree = trees.get_snapshot(dep_hash)
    if tree is None:
        raise click.ClickException(f'Dep hash {dep_hash!r} not found.')

    identity = snapshot_version_identity(reg, dep_hash, tree_registry=trees)
    groups = build_version_groups(version_infos(reg), reg.code_groups_dict())
    match = next((g for g in groups if g['mtime'] == identity), None)
    if match:
        click.echo(match['label'])
        click.echo()
        click.echo(f'Classes  {", ".join(match["class_names"])}')
    else:
        click.echo('Initial')
