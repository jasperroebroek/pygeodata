from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import click

from pygeodata.analysis import ProjectAnalysis, encode_analysis
from pygeodata.cli._stdout import stdout_to_stderr
from pygeodata.imported import analyse_imported
from pygeodata.static import analyse_project


def _summary(analysis: ProjectAnalysis) -> str:
    counts = Counter(cls.facts.object_type for cls in analysis.classes.values())
    lines = [f'{len(analysis.classes)} class(es), {analysis.mode}: {analysis.root}']
    lines.extend(f'  {object_type}: {n}' for object_type, n in sorted(counts.items()))
    lines.extend(f'Warning: {problem}' for problem in analysis.problems)
    return '\n'.join(lines)


@click.command('analyse')
@click.option(
    '--import-all',
    'do_import',
    is_flag=True,
    default=False,
    help='Import the project instead of reading its source.',
)
@click.option('--json', 'as_json', is_flag=True, default=False, help='Print the result as JSON, and nothing else.')
def analyse_cmd(do_import: bool, as_json: bool) -> None:
    """Analyse the project's classes from source, or by importing it."""
    root = Path.cwd()
    # project modules may print at import; with --json stdout carries the result only
    with stdout_to_stderr() if as_json and do_import else nullcontext():
        analysis = analyse_imported(root) if do_import else analyse_project(root)
    click.echo(encode_analysis(analysis) if as_json else _summary(analysis))
