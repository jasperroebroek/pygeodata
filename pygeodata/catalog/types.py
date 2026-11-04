"""The ``.catalog.json`` contract: an index over the cache entries, decoded with msgspec.

The file is disposable: it is derived from the entries' own files and rebuilt when outdated.
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Self

import msgspec

from pygeodata._version import CATALOG_VERSION
from pygeodata.fileio import write_typed_json_atomic
from pygeodata.records import EntryRecord, SpecRecord  # noqa: TC001 — msgspec resolves field types at runtime

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class FileRole(StrEnum):
    PRIMARY = 'primary'
    SIDECAR = 'sidecar'


class ParamRowType(StrEnum):
    VALUE = 'value'
    LIST_MEMBER = 'list_member'
    DATA_REF = 'data_ref'  # nested artifact — class_name + params
    CLASS_REF = 'class_ref'  # tracked class passed as a value


@dataclass(frozen=True, slots=True)
class FileRef:
    """An output file of an entry, relative to the entry dir."""

    relpath: str
    kind: str
    role: FileRole


@dataclass(slots=True)
class ParamRow:
    path: str
    key_group: str
    final_key: str
    value_text: str
    value_type: str  # Python type name of the leaf value; informational only
    depth: int
    row_type: ParamRowType = ParamRowType.VALUE

    @property
    def search_blob(self) -> str:
        """Search text for the ``all`` filter — derived, never stored.

        Fully determined by the other columns, so the catalog stores none of it
        (it used to cost ~25% of the display JSON). Ref rows keep their
        row-type token; the class-ref dep hash is no longer searchable through
        the ``all`` target (Key/Value filters cover it).
        """
        blob = f'{self.key_group} {self.final_key} {self.value_text}'.lower()
        if self.row_type is ParamRowType.DATA_REF:
            return f'{blob} data_ref'
        if self.row_type is ParamRowType.CLASS_REF:
            return f'{blob} class_ref'
        return blob


@dataclass(slots=True)
class LinkedEntry:
    """A Data/TrackedObject reference embedded in another entry's params.

    The referenced entry's params are already flattened into the parent's
    rows (under this param's scope), so no summary is stored — the display
    layer derives it on demand. This used to duplicate the linked loader's
    full params repr in every referencing entry (~90% of linked_entries bytes).
    """

    param_name: str
    class_name: str
    state_hash: str | None


@dataclass
class CatalogWriter:
    root: str
    host: str
    pygeodata: str
    written_at: float


@dataclass
class CatalogHeader:
    catalog_version: int
    schema_fingerprint: str
    catalog_id: str
    writer: CatalogWriter
    # diagnostics only: readers resolve entries against their own local roots
    roots: dict[str, str]


@dataclass
class CatalogEntry:
    record: EntryRecord
    # relative to the cache root of record.object_type; absolute when outside it
    dir_relpath: str
    rows: list[ParamRow]
    linked_entries: list[LinkedEntry]
    files: list[FileRef]
    created_at: float
    updated_at: float

    @property
    def state_hash(self) -> str:
        return self.record.state_hash

    @property
    def class_name(self) -> str:
        return self.record.class_name

    @property
    def object_type(self) -> str:
        return self.record.object_type

    @property
    def primary_file(self) -> FileRef | None:
        return next((f for f in self.files if f.role is FileRole.PRIMARY), None)

    @property
    def sidecar_files(self) -> list[FileRef]:
        return [f for f in self.files if f.role is FileRole.SIDECAR]


class CatalogOutdatedError(Exception):
    """Written by an older pygeodata, or unreadable; rebuild it from the cache entries.

    ``writer_root`` is the project root it was written from, when its header still says so.
    """

    def __init__(self, message: str, catalog_id: str | None = None, writer_root: str | None = None) -> None:
        super().__init__(message)
        self.catalog_id = catalog_id
        self.writer_root = writer_root


class CatalogNewerError(Exception):
    """Written by a newer pygeodata; never rebuild or overwrite it."""


@dataclass
class _HeaderProbe:
    catalog_version: int
    schema_fingerprint: str
    catalog_id: str
    writer: CatalogWriter


@dataclass
class _Probe:
    header: _HeaderProbe


@dataclass
class CatalogData:
    header: CatalogHeader
    entries: dict[str, CatalogEntry]  # by state_hash
    specs: dict[str, SpecRecord]  # by spec_hash

    @classmethod
    def from_file(cls, path: Path) -> Self:
        """Raises ``OSError``, :class:`CatalogOutdatedError` or :class:`CatalogNewerError`."""
        content = path.read_bytes()
        try:
            header = msgspec.json.decode(content, type=_Probe).header
        except msgspec.DecodeError as exc:
            raise CatalogOutdatedError(f'Catalog has no readable header: {exc}') from exc
        if header.catalog_version > CATALOG_VERSION:
            raise CatalogNewerError(
                f'Catalog version {header.catalog_version} was written by pygeodata {header.writer.pygeodata}; '
                f'this pygeodata reads up to version {CATALOG_VERSION}',
            )
        if header.catalog_version < CATALOG_VERSION or header.schema_fingerprint != schema_fingerprint():
            raise CatalogOutdatedError(
                f'Catalog version {header.catalog_version} is outdated',
                catalog_id=header.catalog_id,
                writer_root=header.writer.root,
            )
        try:
            return msgspec.json.decode(content, type=cls)
        except msgspec.DecodeError as exc:
            raise CatalogOutdatedError(
                f'Catalog unreadable: {exc}',
                catalog_id=header.catalog_id,
                writer_root=header.writer.root,
            ) from exc

    def dump(self, path: Path) -> None:
        write_typed_json_atomic(path, self)

    def upsert(self, entry: CatalogEntry, spec: SpecRecord | None) -> None:
        """``spec`` is None when the entry's ``spec.json`` is unreadable."""
        self.entries[entry.record.state_hash] = entry
        if spec is not None:
            self.specs[entry.record.spec_hash] = spec

    def prune(self, state_hashes: Iterable[str]) -> int:
        """Drop these entries, and the specs no remaining entry refers to."""
        removed = sum(self.entries.pop(h, None) is not None for h in set(state_hashes))
        used = {e.record.spec_hash for e in self.entries.values()}
        for spec_hash in self.specs.keys() - used:
            del self.specs[spec_hash]
        return removed


@functools.cache
def schema_fingerprint() -> str:
    """Any change to the catalog's types changes it; pinned by a test so a change forces a version bump."""
    schema = msgspec.json.encode(msgspec.json.schema(CatalogData), order='sorted')
    return hashlib.sha256(schema).hexdigest()


@dataclass
class Pin:
    state_hash: str
    pinned_at: float
