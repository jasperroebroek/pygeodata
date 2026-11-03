import dataclasses
import json
from difflib import unified_diff

import click

from pygeodata.cli.common import _apply_registry_path, _fmt_hash, _fmt_mtime, _full_hash_option, _registry_option
from pygeodata.registries.registry import SourceRegistry, TreeRegistry
from pygeodata.registries.registry_types import Version
from pygeodata.registries.versioning import VersionRegistry


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


@click.group()
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
@click.option('--color/--no-color', default=True, help='Disable diff colors with --no-color.')
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


@click.command('versions')
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


@click.group()
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
