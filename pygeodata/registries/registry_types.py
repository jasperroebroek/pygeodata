from __future__ import annotations

import dataclasses
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Self

from pygeodata.config import FORMAT_VERSION
from pygeodata.paths import CachePathConstructor


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


@dataclass
class EntryRecord:
    """Core identity data for one cache entry.

    Mirrors CodeState / TreeSnapshot — keyed by state_hash in EntryRegistry.
    Browser display fields (rows, spec strings, linked entries, primary file)
    live in EntryInfo in catalog/models.py.

    """

    class_name: str | None = None
    source_hash: str | None = None
    dependency_tree_hash: str | None = None
    instance_hash: str | None = None
    params_hash: str | None = None
    spec_hash: str | None = None
    state_hash: str | None = None
    object_type: str | None = None
    hash_path: str | None = field(default=None, compare=False)
    co_output_hashes: list[str] = field(default_factory=list)
    format_version: int | None = None

    @property
    def params_path(self) -> Path | None:
        if self.hash_path is None:
            return None
        return CachePathConstructor.from_path(Path(self.hash_path)).params_path

    @classmethod
    def from_file(cls, hash_path: Path) -> EntryRecord:
        data = json.loads(hash_path.read_text(encoding='utf-8'))
        return cls(
            class_name=data.get('class_name'),
            hash_path=str(hash_path),
            object_type=data.get('object_type'),
            state_hash=data.get('state_hash'),
            instance_hash=data.get('instance_hash'),
            params_hash=data.get('params_hash'),
            spec_hash=data.get('spec_hash'),
            source_hash=data.get('source_hash'),
            dependency_tree_hash=data.get('dependency_tree_hash'),
            co_output_hashes=data.get('co_output_hashes', []),
            format_version=data.get('format_version'),
        )

    def dump(self, path: Path) -> None:
        d = dataclasses.asdict(self)
        d.pop('hash_path', None)
        path.write_text(json.dumps(d, indent=4), encoding='utf-8')


@dataclass(slots=True)
class TreeSnapshot:
    dependency_tree_hash: str
    nodes: dict[str, dict]
    call_edges: list[list[str]]
    inheritance_edges: list[list[str]]
    root_class: str
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: int = field(default=FORMAT_VERSION)

    def get_source_hash(self, class_name: str) -> str | None:
        node = self.nodes.get(class_name)
        return node.get('hash') if isinstance(node, dict) else None

    def get_object_type(self, class_name: str) -> str | None:
        node = self.nodes.get(class_name)
        return node.get('object_type') if isinstance(node, dict) else None

    def get_call_dependencies(self) -> list[str]:
        """Direct call dependencies of the root class."""
        return sorted({t for s, t in self.call_edges if s == self.root_class})

    def get_inheritance_dependencies(self) -> list[str]:
        """Direct inheritance dependencies of the root class."""
        return sorted({t for s, t in self.inheritance_edges if s == self.root_class})

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=4), encoding='utf-8')

    @classmethod
    def from_file(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(
            dependency_tree_hash=data.get('dependency_tree_hash', ''),
            nodes=data.get('nodes', {}),
            call_edges=data.get('call_edges', []),
            inheritance_edges=data.get('inheritance_edges', []),
            root_class=data.get('root_class', ''),
            registered_at=data.get('registered_at', ''),
            format_version=data.get('format_version', FORMAT_VERSION),
        )


@dataclass(slots=True)
class CodeState:
    source_hash: str
    class_name: str
    object_type: str
    registered_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    format_version: int = field(default=FORMAT_VERSION)

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=4), encoding='utf-8')

    @classmethod
    def from_file(cls, path: Path) -> Self:
        data = json.loads(path.read_text(encoding='utf-8'))
        return cls(
            source_hash=data['source_hash'],
            class_name=data['class_name'],
            object_type=data['object_type'],
            registered_at=data['registered_at'],
            format_version=data.get('format_version', FORMAT_VERSION),
        )


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
