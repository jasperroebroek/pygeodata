"""Importing a project's modules, so its classes register themselves."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pygeodata.config import get_config

if TYPE_CHECKING:
    from pathlib import Path

SKIP_DIRS = frozenset({'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test', 'cache'})


@dataclass
class ImportReport:
    imported: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # module name → error


def _skip_dirs() -> set[str]:
    """Fixed tooling dirs plus the top-level dirs of relative cache and registry paths."""
    config = get_config()
    data_dirs = {p.parts[0] for p in (*config.cache_roots, config.path_registry) if p.parts and not p.is_absolute()}
    return SKIP_DIRS | data_dirs


def import_project_modules(root: Path) -> ImportReport:
    """Import every ``.py`` file under ``root``; a module that fails to import is reported, not raised.

    Settings files are left to :func:`~pygeodata.config.load_project_settings`, which honours their
    priority; importing a second one here would override it. Already imported modules are skipped.
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    skip_dirs = _skip_dirs()
    report = ImportReport()
    for py_file in sorted(root.rglob('*.py')):
        parts = py_file.relative_to(root).parts
        if any(p.startswith(('.', '__pycache__')) or p in skip_dirs for p in parts):
            continue
        if parts[-1].startswith(('test_', 'conftest', 'setup', 'manage')) or parts[-1] == 'pygeodata_settings.py':
            continue
        module_name = 'pygeodata_project_' + '.'.join(py_file.relative_to(root).with_suffix('').parts)
        if module_name in sys.modules:
            continue
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                report.imported.append(module_name)
        except Exception as exc:  # noqa: BLE001 — a broken project module must not stop the others
            report.failed[module_name] = str(exc)
    return report
