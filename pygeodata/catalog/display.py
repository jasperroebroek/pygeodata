"""Entry display data: parameter rows, linked entries, and output-file refs.

Computed once at write/backfill time and stored in the catalog, so import
assembles :class:`~pygeodata.catalog.types.EntryInfo` from the DB without
per-entry file reads. The disk files stay the source of truth; this module
merely derives their display form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pygeodata.catalog.params_index import flatten_params
from pygeodata.catalog.types import FileRef
from pygeodata.figure import Figure
from pygeodata.paths import CACHE_DIR_SUFFIXES, CACHE_META_FILES, classify_file
from pygeodata.table import Table
from pygeodata.tracked_object import TrackedObject

_FAMILY_EXT: dict[str, str] = {'Figure': Figure.ext, 'Table': Table.ext}


def is_output_file(path: Path) -> bool:
    if path.name in CACHE_META_FILES:
        return False
    if path.is_dir():
        return path.suffix.lower() in CACHE_DIR_SUFFIXES
    return path.is_file()


def preferred_ext(class_name: str | None, object_type: str | None) -> str | None:
    """The extension this entry's output is expected to carry, or None if unknown.

    The class's own ``ext`` is the authority, but it is reachable only once the project's classes
    are imported, and ``Data`` leaves it None to be resolved from a driver. The ``object_type``
    in meta.json is always recorded, so the family default stands in — only the concrete
    subclasses register themselves, so the families are named here rather than looked up.
    """
    obj_cls = TrackedObject.find_object_class(class_name) if class_name else None
    ext = getattr(obj_cls, 'ext', None)

    if ext is None:
        ext = _FAMILY_EXT.get(object_type)

    return f'.{ext}'.lower() if isinstance(ext, str) else None


def find_primary_and_sidecar_files(
    entry_dir: Path,
    class_name: str | None = None,
    object_type: str | None = None,
) -> tuple[FileRef | None, list[FileRef]]:
    """The entry's main output, preferring the file whose suffix the class declares, plus
    every other output file in the directory as sidecars.

    A ``_process`` may write sidecars beside its declared output — a CSV next to a rendered HTML
    table, say. Taking whatever ``iterdir`` yields first picks between them by filesystem order,
    so one entry is classified 'table' and the next 'file' for the same class. Sorting breaks
    the tie deterministically; the declared extension breaks it correctly. The remaining
    candidates are not primary-file material but are still real outputs, so they are returned
    as sidecars rather than silently dropped.
    """
    candidates = sorted(
        (p for p in entry_dir.iterdir() if is_output_file(p)),
        key=lambda p: p.name,
    )
    if not candidates:
        return None, []

    ext = preferred_ext(class_name, object_type)
    path = next((p for p in candidates if p.suffix.lower() == ext), candidates[0])
    primary = FileRef(label=path.name, path=str(path.resolve()), kind=classify_file(path))
    sidecars = [FileRef(label=p.name, path=str(p.resolve()), kind=classify_file(p)) for p in candidates if p != path]
    return primary, sidecars


@dataclass
class EntryDisplay:
    """The per-entry display data stored in the catalog, all JSON-safe."""

    rows: list[dict[str, Any]]
    linked_entries: list[dict[str, Any]]
    primary: dict[str, str] | None
    sidecars: list[dict[str, str]]


def _rel_file(ref: FileRef | None, source_root: Path) -> dict[str, str] | None:
    if ref is None:
        return None
    try:
        relpath = str(Path(ref.path).relative_to(source_root.parent))
    except ValueError:
        relpath = ref.path
    return {'label': ref.label, 'relpath': relpath, 'kind': ref.kind}


def entry_display(
    params_json: dict[str, Any],
    entry_dir: Path,
    source_root: Path,
    class_name: str | None = None,
    object_type: str | None = None,
) -> EntryDisplay:
    """Derive the stored display data for one entry from its parameters and output files."""
    linked: list = []
    rows = [asdict(r) for r in flatten_params(params_json, linked_entries=linked)]
    primary, sidecar_refs = find_primary_and_sidecar_files(entry_dir, class_name, object_type)
    return EntryDisplay(
        rows=rows,
        linked_entries=[asdict(le) for le in linked],
        primary=_rel_file(primary, source_root),
        sidecars=[_rel_file(r, source_root) for r in sidecar_refs],
    )
