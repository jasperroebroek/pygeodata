from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import total_ordering
from typing import TYPE_CHECKING, Self

from pygeodata.fileio import read_typed_json, write_typed_json_atomic

if TYPE_CHECKING:
    from pathlib import Path


class ChangeStatus(str, Enum):
    """Classification of a class's state between two adjacent version groups."""

    ADDED = 'added'
    CHANGED = 'changed'
    REMOVED = 'removed'
    UNCHANGED = 'unchanged'


@dataclass(frozen=True)
class CodeEvent:
    """A class-level code event: first appearance, change, removal, or no-change.

    state_new is None for REMOVED; state_old is None for ADDED.
    """

    state_new: CodeState | None
    state_old: CodeState | None

    def __post_init__(self):
        if self.state_new is None and self.state_old is None:
            raise ValueError('CodeEvent must have at least one state')

    @property
    def class_name(self) -> str:
        return (self.state_new or self.state_old).class_name

    @property
    def mtime(self) -> str:
        return (self.state_new or self.state_old).registered_at

    @property
    def status(self) -> ChangeStatus:
        if self.state_old is None:
            return ChangeStatus.ADDED
        if self.state_new is None:
            return ChangeStatus.REMOVED
        if self.state_old.source_hash == self.state_new.source_hash:
            return ChangeStatus.UNCHANGED
        return ChangeStatus.CHANGED

    def to_dict(self) -> dict:
        return {
            'class_name': self.class_name,
            'status': self.status.value,
            'hash_old': self.state_old.source_hash if self.state_old else None,
            'hash_new': self.state_new.source_hash if self.state_new else None,
        }


@dataclass(frozen=True, slots=True)
class TreeNode:
    hash: str
    object_type: str


@dataclass(slots=True)
class TreeSnapshot:
    dependency_tree_hash: str
    nodes: dict[str, TreeNode]
    call_edges: list[tuple[str, str]]
    inheritance_edges: list[tuple[str, str]]
    root_class: str
    registered_at: str
    # absent in the file: written before format versioning
    format_version: int = 0

    def get_source_hash(self, class_name: str) -> str | None:
        node = self.nodes.get(class_name)
        return node.hash if node is not None else None

    def get_object_type(self, class_name: str) -> str | None:
        node = self.nodes.get(class_name)
        return node.object_type if node is not None else None

    def get_call_dependencies(self) -> list[str]:
        """Direct call dependencies of the root class."""
        return sorted({t for s, t in self.call_edges if s == self.root_class})

    def get_inheritance_dependencies(self) -> list[str]:
        """Direct inheritance dependencies of the root class."""
        return sorted({t for s, t in self.inheritance_edges if s == self.root_class})

    def dump(self, path: Path) -> None:
        write_typed_json_atomic(path, self, indent=4)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return read_typed_json(path, cls)


@dataclass(slots=True)
class CodeState:
    source_hash: str
    class_name: str
    object_type: str
    registered_at: str
    # absent in the file: written before format versioning
    format_version: int = 0

    def dump(self, path: Path) -> None:
        write_typed_json_atomic(path, self, indent=4)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return read_typed_json(path, cls)


@total_ordering
@dataclass
class Version:
    """A version-change group: one or more CodeEvents that occurred together.

    The last entry in VersionRegistry.versions is the Initial group —
    it holds the states that existed before any version change, with mtime set
    to the registered_at of the earliest CodeState across all classes.

    version_id is a stable UUID assigned at build time and is the canonical
    identifier for this group in all API calls and lookups.  Sorting compares
    by mtime (ISO-8601 UTC strings sort correctly as plain strings).

    events contains ALL statuses (ADDED, CHANGED, REMOVED, UNCHANGED) — the
    full change summary versus the predecessor version, computed at build time.
    """

    events: list[CodeEvent]
    mtime: str
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.version_id == other.version_id

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        if self.mtime != other.mtime:
            return self.mtime < other.mtime
        return self.version_id < other.version_id

    def __hash__(self) -> int:
        return hash(self.version_id)

    @property
    def class_names(self) -> list[str]:
        """All class names present in this version (ADDED, CHANGED, REMOVED, UNCHANGED)."""
        return sorted({e.class_name for e in self.events})

    @property
    def changed_class_names(self) -> list[str]:
        """Class names that are ADDED or CHANGED in this version."""
        return sorted({e.class_name for e in self.events if e.status in (ChangeStatus.ADDED, ChangeStatus.CHANGED)})

    @property
    def source_hashes(self) -> list[str]:
        return [e.state_new.source_hash for e in self.events if e.state_new is not None]
