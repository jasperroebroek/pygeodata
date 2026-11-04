from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import click

from pygeodata.catalog.types import ParamRowType
from pygeodata.cli.common import (
    LEGEND,
    _entry_registry,
    _fmt_hash,
    _import_options,
    _legend,
    _marker,
    _project_source,
    _resolve_entry,
    _tree_hash_lookup,
)
from pygeodata.enums import ObjectType, Staleness
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.staleness import dependency_staleness
from pygeodata.views.filters import Filter, FilterOperator, FilterTarget, LogicMode, entry_matches_filters
from pygeodata.views.types import spec_info

if TYPE_CHECKING:
    from collections.abc import Callable

    from pygeodata.records import EntryRecord

# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


@click.group()
@click.option('--full-hash', 'full_hash', is_flag=True, default=False, help='Print full hashes.')
@click.pass_context
def entry(ctx: click.Context, full_hash: bool) -> None:
    """Inspect cache entries."""
    ctx.ensure_object(dict)
    ctx.obj['full_hash'] = full_hash


def _stale_indicator(rec: EntryRecord, current: Callable[[str], str | None], vreg: VersionRegistry) -> str:
    """One :data:`~pygeodata.cli.common.LEGEND` code, or a blank when the entry is current."""
    if not rec.is_current_format:
        return 'F'
    staleness = dependency_staleness(rec.class_name, rec.dependency_tree_hash, current, vreg)
    return {Staleness.STALE: 'S', Staleness.UNKNOWN: '?'}.get(staleness, ' ')


def _parse_filter_expr(expr: str) -> tuple[list[Filter], LogicMode] | None:
    """Parse a single --filter expression into its filters and how they combine.

    Syntax:
      [!] [target:]value  (| [!] [target:]value)*

    Examples
    --------
      "RF"                       → ALL contains RF, AND
      "value:RF|value:KMEANS"    → value contains RF OR value contains KMEANS
      "!ElevationLoader"         → ALL not-contains ElevationLoader
      "!RF|KMEANS"               → NOT(RF OR KMEANS)  [! applies to whole OR group]
    """
    expr = expr.strip()
    negate = expr.startswith('!')
    if negate:
        expr = expr[1:].strip()

    parts = [p.strip() for p in expr.split('|') if p.strip()]
    if not parts:
        return None

    filters = []
    for part in parts:
        target = FilterTarget.ALL
        value = part
        if ':' in part:
            raw_target, _, value = part.partition(':')
            try:
                target = FilterTarget(raw_target.strip().lower())
            except ValueError:
                choices = ', '.join(FilterTarget)
                raise click.BadParameter(
                    f'unknown target {raw_target!r}; choose from {choices}',
                    param_hint='--filter',
                ) from None
        filters.append(Filter(target=target, operator=FilterOperator.CONTAINS, value=value.strip()))

    mode = LogicMode.NOT if negate else (LogicMode.OR if len(filters) > 1 else LogicMode.AND)
    return filters, mode


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
        if row.row_type in (ParamRowType.DATA_REF, ParamRowType.CLASS_REF):
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
@click.option(
    '--type',
    'type_filter',
    type=click.Choice(list(ObjectType), case_sensitive=False),
    default=None,
    help='Filter by object type.',
)
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
    type_filter: str | None,
    hide_stale: bool,
    show_params: bool,
    filters: tuple[str, ...],
    full_hash_sub: bool,
    no_pager: bool,
) -> None:
    """List cache entries."""
    current = _tree_hash_lookup(_project_source(do_import, verbose_import))
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub

    parsed_exprs = [parsed for f in filters if (parsed := _parse_filter_expr(f)) is not None]

    reg = _entry_registry()
    if not reg.entries:
        click.echo('No entries found.')
        return

    vreg = VersionRegistry()
    output_rows = []
    for state_hash, entry in sorted(reg.entries.items(), key=lambda kv: (kv[1].class_name, kv[0])):
        if class_name and entry.class_name != class_name:
            continue
        if type_filter and entry.object_type.lower() != type_filter.lower():
            continue
        stale = _stale_indicator(entry.record, current, vreg)
        if hide_stale and stale.strip():
            continue
        spec_record = reg.get_spec(state_hash)
        if not all(entry_matches_filters(entry, spec_record, flts, mode) for flts, mode in parsed_exprs):
            continue
        spec = spec_info(spec_record)
        output_rows.append(
            (
                stale,
                entry.class_name,
                entry.object_type,
                spec.crs or '',
                spec.resolution or '',
                spec.bounds_display or 'None',
                _fmt_hash(state_hash, full_hash),
                entry,
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
    for stale, cls, obj_type, crs, res, bounds, h, entry in output_rows:
        lines.append(
            f'{_marker(stale)} {cls:<{col_class}}  {obj_type:<{col_type}}  {crs:<{col_crs}}  {res:<{col_res}}  {bounds:<{col_bounds}}  {h}',
        )
        if show_params or parsed_exprs:
            lines.extend(_format_param_rows(entry.rows))

    legend = _legend(row[0] for row in output_rows)
    if legend:
        lines.extend(['', legend])
    output = '\n'.join(lines)
    if no_pager or not sys.stdout.isatty():
        click.echo(output)
    else:
        click.echo_via_pager(output + '\n')


@entry.command('show')
@click.argument('hash_prefix')
@_import_options
@click.option('--params/--no-params', 'show_params', default=True, help='Omit params JSON with --no-params.')
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
    current = _tree_hash_lookup(_project_source(do_import, verbose_import))
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub
    reg = _entry_registry()
    entry = _resolve_entry(reg, hash_prefix)
    rec = entry.record
    entry_dir = reg.entry_dir(entry.state_hash)
    resolver = CachePathConstructor(entry_dir)
    spec = spec_info(reg.get_spec(entry.state_hash))

    stale = _stale_indicator(rec, current, VersionRegistry())
    stale_label = LEGEND[stale][1] if stale in LEGEND else 'current'

    click.echo(f'Class        {rec.class_name}')
    click.echo(f'Type         {rec.object_type}')
    click.echo(f'State hash   {_fmt_hash(rec.state_hash, full_hash)}')
    click.echo(f'Instance     {_fmt_hash(rec.instance_hash, full_hash)}')
    click.echo(f'Dep hash     {_fmt_hash(rec.dependency_tree_hash, full_hash)}')
    click.echo(f'Staleness    {stale_label}')

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

    click.echo()
    if entry.primary_file:
        click.echo(f'Output       {entry_dir / entry.primary_file.relpath}')
    for f in entry.sidecar_files:
        click.echo(f'Sidecar      {entry_dir / f.relpath}')
    click.echo(f'Params path  {resolver.params_path}')
    click.echo(f'Hash path    {resolver.state_hash_path}')
    if resolver.spec_path.exists():
        click.echo(f'Spec path    {resolver.spec_path}')

    if show_params:
        try:
            params = json.loads(resolver.params_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            params = None
        if params:
            click.echo()
            click.echo('Params')
            click.echo(json.dumps(params, indent=2))
