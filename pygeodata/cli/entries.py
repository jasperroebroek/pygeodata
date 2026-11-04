import json
import sys
from pathlib import Path

import click
import msgspec

from pygeodata.catalog.types import ParamRowType
from pygeodata.cli.common import _fmt_hash, _import_options, _import_project_modules, _resolve_record
from pygeodata.enums import Staleness
from pygeodata.paths import CachePathConstructor
from pygeodata.records import EntryRecord, SpecRecord
from pygeodata.registries.registry import EntryRegistry
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.staleness import dependency_staleness
from pygeodata.tracked_object import TrackedObject
from pygeodata.views.entries import _enrich_params_path
from pygeodata.views.filters import Filter, FilterOperator, FilterTarget, entry_matches_filters, prescan_text
from pygeodata.views.types import spec_info

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


def _stale_indicator(rec: EntryRecord, vreg: VersionRegistry) -> str:
    """Single-char staleness marker: F=format stale, S=dep stale, N=not loaded, blank=ok."""
    if not rec.is_current_format:
        return 'F'
    if dependency_staleness(rec.class_name, rec.dependency_tree_hash, vreg) is Staleness.STALE:
        return 'S'
    if TrackedObject.find_object_class(rec.class_name) is None:
        return 'N'
    return ' '


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
        paths = CachePathConstructor(reg.entry_dir(state_hash))
        params_path = paths.params_path
        if parsed_exprs:
            try:
                params_blob = params_path.read_text(encoding='utf-8').lower() if params_path.exists() else ''
            except OSError:
                params_blob = ''
            if not prescan_text(rec.class_name or '', params_blob, parsed_exprs):
                continue

        entry_info = None
        if needs_enrich:
            # Deliberately not going through _enrich_with_cache: that would consult the
            # browser's .dashboard_cache.json, whose upfront load is a fixed cost regardless
            # of how many entries are enriched. It only pays off for ~500 enriched entries, which
            # is not common yet. Change as needed.
            try:
                entry_info = _enrich_params_path(params_path, rec)
            except Exception:  # noqa: BLE001
                pass

        if parsed_exprs:
            if entry_info is None:
                continue
            if not _entry_matches_all_exprs(rec.class_name or '', entry_info, parsed_exprs):
                continue

        if entry_info is not None:
            spec = entry_info.spec
        else:
            try:
                spec = spec_info(SpecRecord.from_file(paths.spec_path))
            except (OSError, msgspec.DecodeError):
                spec = spec_info(None)
        crs_str = spec.crs or ''
        res_str = spec.resolution or ''
        bounds_str = spec.bounds_display or 'None'
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
    if do_import:
        n = _import_project_modules(Path.cwd(), verbose=verbose_import)
        click.echo(f'Imported {n} module(s)')
    full_hash: bool = ctx.obj.get('full_hash', False) or full_hash_sub
    reg = EntryRegistry()
    state_hash, rec = _resolve_record(reg, hash_prefix)

    resolver = CachePathConstructor(reg.entry_dir(state_hash))

    entry_info = None
    try:
        entry_info = _enrich_params_path(resolver.params_path, rec)
    except Exception:  # noqa: BLE001
        pass

    stale = _stale_indicator(rec, VersionRegistry())
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

    click.echo()
    if entry_info and entry_info.primary_file:
        click.echo(f'Output       {entry_info.file_path(entry_info.primary_file)}')
    for f in entry_info.sidecar_files if entry_info else []:
        click.echo(f'Sidecar      {entry_info.file_path(f)}')
    click.echo(f'Params path  {resolver.params_path}')
    click.echo(f'Hash path    {resolver.state_hash_path}')
    if resolver.spec_path.exists():
        click.echo(f'Spec path    {resolver.spec_path}')

    if show_params and entry_info and entry_info.params:
        click.echo()
        click.echo('Params')
        click.echo(json.dumps(entry_info.params, indent=2))
