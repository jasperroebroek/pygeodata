import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from pygeodata.cli import cli
from pygeodata.config import get_config
from pygeodata.static import analyse_project
from tests.fixtures.records import make_record, write_meta

LOADERS = """
from pygeodata import Data, Figure


class SourceLoader(Data):
    def _process(self, spec):
        return 1


class SourceFigure(Figure):
    pass
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_imports: None) -> Path:
    """A project whose classes are read from source; nothing is imported."""
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / '__init__.py').write_text('')
    (tmp_path / 'src' / 'loaders.py').write_text(textwrap.dedent(LOADERS))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_entry(state_hash: str, class_name: str, object_type: str, dependency_tree_hash: str) -> None:
    root = get_config().path_figures if object_type == 'Figure' else get_config().path_cache
    write_meta(
        root / state_hash,
        make_record(state_hash, class_name, object_type, dependency_tree_hash=dependency_tree_hash),
    )


def test_entry_list_marks_staleness_from_source(project: Path) -> None:
    current = analyse_project(project).tree_hashes
    write_entry('aaa111', 'SourceLoader', 'Data', current['SourceLoader'])
    write_entry('bbb222', 'SourceLoader', 'Data', 'an-old-tree')
    write_entry('ccc333', 'GoneLoader', 'Data', 'whatever')

    result = CliRunner().invoke(cli, ['entry', 'list', '--no-pager'])

    assert result.exit_code == 0, result.output
    lines = {line.split()[-1]: line for line in result.output.splitlines() if line[2:].startswith(('Source', 'Gone'))}
    assert lines['aaa111'].startswith('  SourceLoader')
    assert lines['bbb222'].startswith('S SourceLoader')
    assert lines['ccc333'].startswith('? GoneLoader')
    assert 'stale: its dependency tree changed' in result.output


@pytest.mark.usefixtures('project')
def test_entry_list_type_filter() -> None:
    write_entry('aaa111', 'SourceLoader', 'Data', 'h')
    write_entry('fff111', 'SourceFigure', 'Figure', 'h')

    result = CliRunner().invoke(cli, ['entry', 'list', '--no-pager', '--type', 'figure'])

    assert result.exit_code == 0, result.output
    assert 'SourceFigure' in result.output
    assert 'SourceLoader' not in result.output


def test_clean_cache_refuses_when_a_project_module_fails_to_import(project: Path) -> None:
    (project / 'src' / 'broken.py').write_text('raise RuntimeError("missing dependency")\n')
    write_entry('aaa111', 'SourceLoader', 'Data', 'an-old-tree')

    dry = CliRunner().invoke(cli, ['clean-cache'])
    real = CliRunner().invoke(cli, ['clean-cache', '--no-dry-run'])

    assert dry.exit_code == 0
    assert 'missing dependency' in dry.output
    assert real.exit_code == 1
    assert 'Not deleting anything' in real.output
    assert (get_config().path_cache / 'aaa111').exists()


@pytest.mark.usefixtures('project')
def test_catalog_info() -> None:
    write_entry('aaa111', 'SourceLoader', 'Data', 'h')
    (get_config().path_cache / 'not-an-entry').mkdir()

    result = CliRunner().invoke(cli, ['catalog', 'info'])

    assert result.exit_code == 0, result.output
    assert 'writable' in result.output
    assert 'Data: 1' in result.output
    assert 'Not indexed  1 entry dir(s)' in result.output
