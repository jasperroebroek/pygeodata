"""The project's tracked classes as imported: every class registered once the project's modules ran."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from pygeodata.analysis import ClassRecord, ProjectAnalysis
from pygeodata.project import import_project_modules
from pygeodata.static import analyse_project, compare_with_imports
from pygeodata.tracked_object import TrackedObject

if TYPE_CHECKING:
    from pathlib import Path


def analyse_imported(root: Path) -> ProjectAnalysis:
    """Import the project's modules, then collect every registered class and where source reading disagrees."""
    source = analyse_project(root)
    report = import_project_modules(root)
    problems = [
        *source.problems,
        *(f'{module}: {error}' for module, error in report.failed.items()),
        *compare_with_imports(source),
    ]

    classes: dict[str, ClassRecord] = {}
    for cls in sorted(TrackedObject.get_registered_objects(), key=lambda c: c.get_class_name()):
        name = cls.get_class_name()
        try:
            path = inspect.getsourcefile(cls)
            lineno = inspect.getsourcelines(cls)[1]
        except (OSError, TypeError) as exc:
            problems.append(f'{name}: source not found: {exc}')
            continue
        classes[name] = ClassRecord(
            facts=cls.get_facts(),
            path=path,
            lineno=lineno,
            tree_hash=cls.get_dependency_tree_hash(),
        )
    return ProjectAnalysis(mode='import', root=str(root), classes=classes, problems=problems)
