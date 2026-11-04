"""Importing a project's modules, so its classes register themselves."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pygeodata.config import get_config

if TYPE_CHECKING:
    from collections.abc import Iterator

SKIP_DIRS = frozenset({'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test', 'cache'})


@dataclass
class ImportReport:
    imported: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # module name → error


def _data_dirs() -> set[Path]:
    """The cache roots and the registry, resolved: never searched for code, wherever they are."""
    config = get_config()
    return {p.resolve() for p in (*config.cache_roots, config.path_registry)}


def _is_project_file(name: str) -> bool:
    return (
        name.endswith('.py')
        and not name.startswith(('.', 'test_', 'conftest', 'setup', 'manage'))
        and name != 'pygeodata_settings.py'
    )


def iter_project_files(root: Path) -> Iterator[Path]:
    """The project's own ``.py`` files: those at ``root`` and in its packages (dirs with ``__init__.py``).

    Only packages are entered, so data and cache dirs are never listed — on a synced mirror each
    first listing is slow. Settings files are left to :func:`~pygeodata.config.load_project_settings`,
    which honours their priority; importing a second one would override it.
    """
    data_dirs = _data_dirs()
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        for entry in entries:
            if entry.is_file() and _is_project_file(entry.name):
                yield Path(entry.path)
        pending.extend(
            Path(entry.path)
            for entry in reversed(entries)
            if entry.is_dir()
            and not entry.name.startswith(('.', '__pycache__'))
            and entry.name not in SKIP_DIRS
            and Path(entry.path, '__init__.py').is_file()
            and Path(entry.path).resolve() not in data_dirs
        )


def _module_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


def import_project_modules(root: Path) -> ImportReport:
    """Import every project file under ``root`` by its module name, as ``root`` on ``sys.path`` makes it.

    A module that fails to import is reported, not raised; one already imported (also as a side
    effect of an earlier one) is skipped.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    report = ImportReport()
    for py_file in iter_project_files(root):
        module_name = _module_name(py_file, root)
        if module_name in sys.modules:
            continue
        try:
            importlib.import_module(module_name)
            report.imported.append(module_name)
        except Exception as exc:  # noqa: BLE001 — a broken project module must not stop the others
            report.failed[module_name] = f'{type(exc).__name__}: {exc}'
    return report
