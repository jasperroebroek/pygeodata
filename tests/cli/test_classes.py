from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from pygeodata.catalog.types import ClassInfo
from pygeodata.cli import cli


def _patch_registries(
    monkeypatch: pytest.MonkeyPatch,
    discover: Callable[[], dict[str, ClassInfo]],
    merge: Callable[..., dict[str, ClassInfo]],
) -> None:
    monkeypatch.setattr('pygeodata.cli.EntryRegistry', MagicMock(return_value=MagicMock(class_names=[])))
    monkeypatch.setattr('pygeodata.cli.VersionRegistry', MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        'pygeodata.cli.SourceRegistry',
        MagicMock(
            return_value=MagicMock(
                get_latest_state_for_class=MagicMock(return_value=None),
            ),
        ),
    )
    monkeypatch.setattr('pygeodata.cli.discover_loaded_classes', discover)
    monkeypatch.setattr('pygeodata.cli.merge_unloaded_classes', merge)


def _identity_merge(classes: dict[str, ClassInfo], *_args: object, **_kwargs: object) -> dict[str, ClassInfo]:
    return classes


def test_classes_no_classes_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """With an empty registry and no loaded classes, prints 'No classes found.'."""
    monkeypatch.setattr('pygeodata.tracked_object.TrackedObject._registry', {})

    runner = CliRunner()
    result = runner.invoke(cli, ['classes'])

    assert result.exit_code == 0, result.output
    assert 'No classes found.' in result.output


def test_classes_loaded_class_appears(monkeypatch: pytest.MonkeyPatch) -> None:
    """A loaded, current class is printed with a space indicator and 'loaded' label."""
    mock_info = ClassInfo(
        class_name='MyLoader',
        object_type='Data',
        loaded=True,
        call_dependency_names=[],
        inheritance_dependency_names=[],
        source_stale=False,
        deps_stale=False,
    )
    monkeypatch.setattr('pygeodata.catalog.class_catalog.TrackedObject._registry', {})
    _patch_registries(monkeypatch, lambda: {'MyLoader': mock_info}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes'])

    assert result.exit_code == 0, result.output
    assert 'MyLoader' in result.output
    assert 'loaded' in result.output
    assert 'Data' in result.output


def test_classes_stale_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """A source-stale class causes exit code 1."""
    mock_info = ClassInfo(
        class_name='StaleClass',
        object_type='Figure',
        loaded=True,
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
    fresh = ClassInfo(class_name='FreshOne', object_type='Data', loaded=True)
    stale = ClassInfo(class_name='StaleOne', object_type='Data', loaded=True, source_stale=True)
    _patch_registries(monkeypatch, lambda: {'FreshOne': fresh, 'StaleOne': stale}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes', '--hide-current'])

    assert 'FreshOne' not in result.output
    assert 'StaleOne' in result.output


def test_classes_type_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """--type DATA filters to Data classes only."""
    data_cls = ClassInfo(class_name='DataClass', object_type='Data', loaded=True)
    fig_cls = ClassInfo(class_name='FigureClass', object_type='Figure', loaded=True)
    _patch_registries(monkeypatch, lambda: {'DataClass': data_cls, 'FigureClass': fig_cls}, _identity_merge)

    runner = CliRunner()
    result = runner.invoke(cli, ['classes', '--type', 'DATA'])

    assert result.exit_code == 0, result.output
    assert 'DataClass' in result.output
    assert 'FigureClass' not in result.output
