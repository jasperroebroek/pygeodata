"""Entry display data: parameter rows, linked entries, and output-file refs.

Computed once at write/backfill time and stored in the catalog, so readers
assemble :class:`~pygeodata.views.types.EntryInfo` from the catalog without
per-entry file reads. The disk files stay the source of truth; this module
merely derives their display form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pygeodata.catalog.params_index import flatten_params
from pygeodata.catalog.types import FileRef, FileRole, LinkedEntry, ParamRow
from pygeodata.enums import ObjectType
from pygeodata.paths import CachePathConstructor, classify_file
from pygeodata.tracked_object import TrackedObject

if TYPE_CHECKING:
    from pathlib import Path


def preferred_ext(class_name: str | None, object_type: str | None) -> str | None:
    """The extension this entry's output is expected to carry, or None if unknown.

    The class's own ``ext`` is the authority, but it is reachable only once the project's classes
    are imported, and ``Data`` leaves it None to be resolved from a driver. The ``object_type``
    in meta.json is always recorded, so the family default stands in — only the concrete
    subclasses register themselves, so the families are named here rather than looked up.
    """
    obj_cls = TrackedObject.find_object_class(class_name) if class_name else None
    ext = getattr(obj_cls, 'ext', None)

    if ext is None and object_type in ObjectType.__members__.values():
        ext = ObjectType(object_type).default_ext

    return f'.{ext}'.lower() if isinstance(ext, str) else None


def find_output_files(
    entry_dir: Path,
    class_name: str | None = None,
    object_type: str | None = None,
) -> list[FileRef]:
    """The entry's output files, relative to ``entry_dir``: the main output first, then sidecars.

    The main output is the file whose suffix the class declares. A ``_process`` may write sidecars
    beside it, a CSV next to a rendered HTML table, say. Taking whatever ``iterdir`` yields first
    picks between them by filesystem order, so one entry is classified 'table' and the next
    'file' for the same class. Sorting breaks the tie deterministically; the declared extension
    breaks it correctly. The other candidates are real outputs too, so they are kept as sidecars.
    """
    candidates = CachePathConstructor(entry_dir).output_paths()
    if not candidates:
        return []

    ext = preferred_ext(class_name, object_type)
    primary = next((p for p in candidates if p.suffix.lower() == ext), candidates[0])
    ordered = [primary, *(p for p in candidates if p != primary)]
    return [
        FileRef(
            relpath=_relpath(p, entry_dir),
            kind=classify_file(p),
            role=FileRole.PRIMARY if p == primary else FileRole.SIDECAR,
        )
        for p in ordered
    ]


def _relpath(path: Path, entry_dir: Path) -> str:
    try:
        # Both sides resolved: entry dirs reached through a symlinked home (e.g. eos_user_home)
        # must relativize against the resolved file paths.
        return str(path.resolve().relative_to(entry_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def param_rows(params_json: dict[str, Any]) -> tuple[list[ParamRow], list[LinkedEntry]]:
    linked: list[LinkedEntry] = []
    rows = flatten_params(params_json, linked_entries=linked)
    return rows, linked
