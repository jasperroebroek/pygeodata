"""Entry display data: parameter rows, linked entries, and output-file refs.

Computed once at write/backfill time and stored in the catalog, so readers
display entries without per-entry file reads. The disk files stay the source
of truth; this module merely derives their display form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pygeodata.catalog.params_index import flatten_params
from pygeodata.catalog.types import FileRef, FileRole, LinkedEntry, ParamRow
from pygeodata.paths import CachePathConstructor, classify_file

if TYPE_CHECKING:
    from pathlib import Path


def find_output_files(entry_dir: Path, ext: str | None) -> list[FileRef]:
    """The entry's output files, relative to ``entry_dir``: the main output first, then sidecars.

    The main output is the file with extension ``ext``, the entry's
    :attr:`~pygeodata.records.EntryRecord.output_ext`. A ``_process`` may write sidecars beside it,
    a CSV next to a rendered HTML table, say. Taking whatever ``iterdir`` yields first picks between
    them by filesystem order, so one entry is classified 'table' and the next 'file' for the same
    class. Sorting breaks the tie deterministically; the recorded extension breaks it correctly.
    The other candidates are real outputs too, so they are kept as sidecars.
    """
    candidates = CachePathConstructor(entry_dir).output_paths()
    if not candidates:
        return []

    suffix = f'.{ext}'.lower() if ext else None
    primary = next((p for p in candidates if p.suffix.lower() == suffix), candidates[0])
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
