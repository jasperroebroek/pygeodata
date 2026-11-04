"""On-disk contracts of a cache entry: ``meta.json`` and ``spec.json``.

Plain data only, so reading an entry never loads numpy or pyproj; field names are the JSON keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from affine import Affine  # runtime import: msgspec resolves field types at runtime

from pygeodata._version import FORMAT_VERSION
from pygeodata.enums import ObjectType
from pygeodata.fileio import read_typed_json, write_typed_json_atomic

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class EntryRecord:
    """Identity of one cache entry; ``meta.json`` is the source of truth for its validity."""

    class_name: str
    source_hash: str
    dependency_tree_hash: str
    instance_hash: str
    params_hash: str
    spec_hash: str
    state_hash: str
    object_type: str
    co_output_hashes: list[str]
    # absent in the file: written before format versioning
    format_version: int = 0
    # absent in the file: written before the output's extension was recorded
    ext: str | None = None

    @property
    def is_current_format(self) -> bool:
        return self.format_version == FORMAT_VERSION

    @property
    def output_ext(self) -> str | None:
        if self.ext is not None:
            return self.ext
        try:
            return ObjectType(self.object_type).default_ext
        except ValueError:
            return None

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return read_typed_json(path, cls)

    def dump(self, path: Path) -> None:
        write_typed_json_atomic(path, self, indent=4)


@dataclass(frozen=True)
class SpecRecord:
    """The defining part of a :class:`~pygeodata.spec.SpatialSpec`; ``spec.json`` decodes into it."""

    crs: str | None
    transform: Affine | None = None
    shape: tuple[int, int] | None = None

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return read_typed_json(path, cls)
