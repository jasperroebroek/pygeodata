import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from pygeodata.analysis import AnalysisError, ProjectAnalysis, analyse_in_subprocess, decode_analysis, encode_analysis
from pygeodata.cli import cli
from pygeodata.imported import analyse_imported
from pygeodata.static import analyse_project

REPO = Path(__file__).parent.parent
PROJECT_CLASSES = {'AnaBase', 'AnaChild', 'AnaTable'}


def write(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code), encoding='utf-8')


@pytest.fixture
def own_pygeodata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subprocesses run in a temporary project and must import this checkout of pygeodata."""
    monkeypatch.setenv('PYTHONPATH', str(REPO))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / 'proj'
    write(
        root / 'loaders.py',
        """
        import dataclasses

        from pygeodata import Data, Table

        print('printed at import')


        class AnaBase(Data[int]):
            def _process(self, spec):
                return 1


        @dataclasses.dataclass
        class AnaChild(AnaBase):
            year: int = 2000

            def _process(self, spec):
                return AnaTable()


        class AnaTable(Table):
            pass
        """,
    )
    return root


def project_classes(analysis: ProjectAnalysis) -> dict:
    return {name: cls for name, cls in analysis.classes.items() if name.startswith('Ana')}


@pytest.mark.usefixtures('isolated_imports')
def test_static_and_import_results_agree(project: Path) -> None:
    static = analyse_project(project)
    imported = analyse_imported(project)

    assert static.mode == 'static'
    assert imported.mode == 'import'
    assert set(static.classes) == PROJECT_CLASSES
    assert project_classes(imported) == static.classes
    assert static.problems == []


@pytest.mark.usefixtures('isolated_imports')
def test_import_mode_reports_failed_imports(project: Path) -> None:
    write(project / 'broken.py', 'raise RuntimeError("boom")\n')

    assert 'broken: RuntimeError: boom' in analyse_imported(project).problems


@pytest.mark.usefixtures('isolated_imports')
@pytest.mark.parametrize('mode', ['static', 'import'])
def test_analysis_round_trips_through_json(project: Path, mode: str) -> None:
    analysis = analyse_project(project) if mode == 'static' else analyse_imported(project)
    analysis.problems.append('something')

    assert decode_analysis(encode_analysis(analysis)) == analysis


@pytest.mark.usefixtures('own_pygeodata')
def test_json_stdout_is_only_the_result_despite_prints(project: Path) -> None:
    write(project / 'pygeodata_settings.py', 'print("printed by settings")\n')
    result = subprocess.run(  # noqa: S603 — fixed interpreter and arguments
        [sys.executable, '-m', 'pygeodata', 'analyse', '--import-all', '--json'],
        cwd=project,
        capture_output=True,
        check=True,
    )

    assert set(project_classes(decode_analysis(result.stdout))) == PROJECT_CLASSES
    assert b'printed at import' in result.stderr
    assert b'printed by settings' in result.stderr


def test_summary_without_json(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli, ['analyse'])

    assert result.exit_code == 0, result.output
    assert '3 class(es), static' in result.output
    assert 'Data: 2' in result.output
    assert 'Table: 1' in result.output


def test_static_json(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli, ['analyse', '--json'])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['mode'] == 'static'


@pytest.mark.usefixtures('own_pygeodata')
def test_analyse_in_subprocess(project: Path) -> None:
    analysis = analyse_in_subprocess(project, timeout=120)

    assert analysis.mode == 'import'
    assert Path(analysis.root).resolve() == project.resolve()
    assert set(project_classes(analysis)) == PROJECT_CLASSES


@pytest.mark.usefixtures('own_pygeodata')
def test_subprocess_failure_carries_stderr(tmp_path: Path) -> None:
    write(tmp_path / 'pygeodata_settings.py', 'raise RuntimeError("bad settings")\n')

    with pytest.raises(AnalysisError, match='bad settings'):
        analyse_in_subprocess(tmp_path, timeout=120)


@pytest.mark.usefixtures('own_pygeodata')
def test_subprocess_timeout(tmp_path: Path) -> None:
    write(tmp_path / 'slow.py', 'import time\ntime.sleep(60)\n')

    with pytest.raises(AnalysisError, match='longer than'):
        analyse_in_subprocess(tmp_path, timeout=3)


def test_subprocess_undecodable_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = subprocess.CompletedProcess([], 0, stdout=b'not json', stderr=b'noise')
    monkeypatch.setattr(subprocess, 'run', lambda *_, **__: completed)

    with pytest.raises(AnalysisError, match='undecodable'):
        analyse_in_subprocess(tmp_path, timeout=1)
