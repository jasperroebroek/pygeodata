import importlib
import textwrap
from pathlib import Path

import pytest

from pygeodata.config import set_config
from pygeodata.project import import_project_modules, iter_project_files
from pygeodata.static import analyse_project, compare_with_imports
from pygeodata.tracked_object import TrackedObject

FIXTURES = Path(__file__).parent / 'fixtures'


def write(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(code), encoding='utf-8')


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Base classes in one module, subclasses in a package named like the figures cache root."""
    root = tmp_path / 'proj'
    write(root / 'src' / '__init__.py', '')
    write(
        root / 'src' / 'loaders.py',
        """
        import dataclasses

        import pygeodata as pg
        from pygeodata import Data


        class StaticBase(Data[int]):
            def _process(self, spec):
                return 1


        @dataclasses.dataclass
        class StaticChild(StaticBase):
            year: int = 2000

            def _process(self, spec):
                return StaticHelper()


        class StaticHelper(pg.Data):
            pass


        class NotTracked:
            pass
        """,
    )
    write(root / 'src' / 'figures' / '__init__.py', '')
    write(
        root / 'src' / 'figures' / 'plots.py',
        """
        from pygeodata.figure import Figure

        from ..loaders import StaticChild


        class StaticPlot(Figure):
            def _process(self, spec):
                return StaticChild(year=2001)
        """,
    )
    write(root / 'notes' / 'scratch.py', 'class Ignored: pass\n')  # not a package: never listed
    return root


def test_static_equals_live_for_all_fixture_classes() -> None:
    for module in FIXTURES.glob('*.py'):
        importlib.import_module(f'tests.fixtures.{module.stem}')

    analysis = analyse_project(FIXTURES)

    assert analysis.problems == []
    fixture_classes = {
        name for name, cls in TrackedObject._registry.items() if cls.__module__.startswith('tests.fixtures.')
    }
    assert fixture_classes <= set(analysis.classes)
    for name in fixture_classes:
        live = TrackedObject._registry[name]
        assert analysis.facts[name] == live.get_facts(), name
        assert analysis.tree_hashes[name] == live.get_dependency_tree_hash(), name


@pytest.mark.usefixtures('isolated_imports')
def test_static_equals_live_for_a_project(project: Path) -> None:
    analysis = analyse_project(project)
    report = import_project_modules(project)

    assert report.failed == {}
    assert set(analysis.classes) == {'StaticBase', 'StaticChild', 'StaticHelper', 'StaticPlot'}
    assert analysis.classes['StaticPlot'].facts.object_type == 'Figure'
    assert analysis.classes['StaticChild'].facts.bases == {'StaticBase'}
    assert compare_with_imports(analysis) == [
        f'{name}: imported but not found in source'
        for name in sorted(TrackedObject._registry)
        if not name.startswith('Static')
    ]


def test_edited_source_changes_the_tree_hash(project: Path) -> None:
    before = analyse_project(project).tree_hashes
    loaders = project / 'src' / 'loaders.py'
    loaders.write_text(loaders.read_text().replace('return 1', 'return 2'), encoding='utf-8')

    after = analyse_project(project).tree_hashes

    assert {name for name in before if before[name] != after[name]} == {'StaticBase', 'StaticChild', 'StaticPlot'}


def test_problems_are_reported_and_duplicates_left_out(project: Path) -> None:
    write(project / 'src' / 'broken.py', 'class Broken(:\n')
    write(project / 'src' / 'again.py', 'from pygeodata import Data\n\nclass StaticHelper(Data):\n    pass\n')

    analysis = analyse_project(project)

    assert 'StaticHelper' not in analysis.classes
    assert any('broken.py' in problem for problem in analysis.problems)
    assert any(problem.startswith('StaticHelper') for problem in analysis.problems)


def test_data_dirs_are_skipped_by_path_not_name(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    write(project / 'figures' / '__init__.py', '')
    write(project / 'figures' / 'stray.py', 'x = 1\n')

    with set_config(path_figures=Path('figures')):
        files = {p.relative_to(project).as_posix() for p in iter_project_files(project)}

    assert 'src/figures/plots.py' in files
    assert 'figures/stray.py' not in files
    assert 'notes/scratch.py' not in files
