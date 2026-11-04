from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from pygeodata.cli import cli
from pygeodata.views.types import ClassInfo


def _patch_registries(
    monkeypatch: pytest.MonkeyPatch,
    discover: Callable[[], dict[str, ClassInfo]],
    merge: Callable[..., dict[str, ClassInfo]],
) -> None:
    monkeypatch.setattr('pygeodata.cli.classes._entry_registry', MagicMock(return_value=MagicMock(class_names=[])))
    monkeypatch.setattr('pygeodata.cli.classes.VersionRegistry', MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        'pygeodata.cli.classes.SourceRegistry',
        MagicMock(
            return_value=MagicMock(
                get_latest_state_for_class=MagicMock(return_value=None),
            ),
        ),
    )
    monkeypatch.setattr('pygeodata.cli.classes.source_classes', lambda *_args: discover())
    monkeypatch.setattr('pygeodata.cli.classes.merge_unknown_classes', merge)


@pytest.fixture(autouse=True)
def empty_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reads the project's classes from the working directory."""
    monkeypatch.chdir(tmp_path)


def _identity_merge(classes: dict[str, ClassInfo], *_args: object, **_kwargs: object) -> dict[str, ClassInfo]:
    return classes


def test_classes_no_classes_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no classes in the project or the cache, prints 'No classes found.'."""
    monkeypatch.setattr('pygeodata.tracked_object.TrackedObject._registry', {})

    runner = CliRunner()
    result = runner.invoke(cli, ['classes'])

    assert result.exit_code == 0, result.output
    assert 'No classes found.' in result.output


def test_classes_loaded_class_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    """A current class is printed without a marker."""
    mock_info = ClassInfo(
        class_name='MyLoader',
        object_type='Data',
        known=True,
        call_dependency_names=[],
        inheritance_dependency_names=[],
        source_stale=False,
        deps_stale=False,
    )
    monkeypatch.setattr('pygeodata.views.classes.TrackedObject._registry', {})
    _patch_registries(monkeypatch, lambda: {'MyLoader': mock_info}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes'])

    assert result.exit_code == 0, result.output
    assert '    MyLoader' in result.output
    assert 'Data' in result.output


def test_classes_stale_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source-stale class causes exit code 1."""
    mock_info = ClassInfo(
        class_name='StaleClass',
        object_type='Figure',
        known=True,
        source_stale=True,
        deps_stale=False,
    )
    _patch_registries(monkeypatch, lambda: {'StaleClass': mock_info}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes'])

    assert result.exit_code == 1
    assert 'StaleClass' in result.output


def test_classes_hide_current_omits_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """--hide-current omits classes with no staleness."""
    fresh = ClassInfo(class_name='FreshOne', object_type='Data', known=True)
    stale = ClassInfo(class_name='StaleOne', object_type='Data', known=True, source_stale=True)
    _patch_registries(monkeypatch, lambda: {'FreshOne': fresh, 'StaleOne': stale}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes', '--hide-current'])

    assert 'FreshOne' not in result.output
    assert 'StaleOne' in result.output


def test_classes_type_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """--type DATA filters to Data classes only."""
    data_cls = ClassInfo(class_name='DataClass', object_type='Data', known=True)
    fig_cls = ClassInfo(class_name='FigureClass', object_type='Figure', known=True)
    _patch_registries(monkeypatch, lambda: {'DataClass': data_cls, 'FigureClass': fig_cls}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes', '--type', 'DATA'])

    assert result.exit_code == 0, result.output
    assert 'DataClass' in result.output
    assert 'FigureClass' not in result.output
