from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from pygeodata.config import FORMAT_VERSION, JSONKeys
from pygeodata.paths import CachePathResolver


@dataclass
class EntryRecord:
    """Core identity data for one cache entry.

    Mirrors CodeState / TreeSnapshot — keyed by state_hash in EntryRegistry.
    Browser display fields (rows, spec strings, linked entries, primary file)
    live in EntryInfo in registry_browser/models.py.

    dep_hash_stale is set during assembly after comparing against the live
    class dependency tree hash.
    """

    class_name: str
    hash_path: str | None = None
    state_hash: str | None = None
    instance_hash: str | None = None
    dep_hash: str | None = None
    co_output_hashes: list[str] = field(default_factory=list)
    object_type: str | None = None
    format_version: int = field(default=FORMAT_VERSION)
    dep_hash_stale: bool | None = None

    @property
    def params_path(self) -> Path | None:
        if self.hash_path is None:
            return None
        return CachePathResolver.from_path(Path(self.hash_path)).params_path

    @classmethod
    def from_hash_path(cls, hash_path: Path) -> EntryRecord | None:
        """Construct from a *.hash.json file. Returns None if missing or unreadable."""
        try:
            state = json.loads(hash_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            return None

        return cls(
            class_name=state.get(JSONKeys.CLASS_NAME),
            hash_path=str(hash_path),
            object_type=state.get(JSONKeys.OBJECT_TYPE),
            state_hash=state.get(JSONKeys.STATE_HASH),
            instance_hash=state.get(JSONKeys.INSTANCE_HASH),
            dep_hash=state.get(JSONKeys.DEPENDENCY_TREE_HASH),
            co_output_hashes=state.get(JSONKeys.CO_OUTPUTS, []),
            format_version=state.get(JSONKeys.FORMAT_VERSION, FORMAT_VERSION),
        )




@dataclass(slots=True)
class GroupRecord:
    """Group of entry records sharing a class_name — keyed by class_name."""

    class_name: str
    object_type: str | None
    state_hashes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TreeSnapshot:
    dep_hash: str
    nodes: dict[str, dict]  # {class_name: {hash, object_type}}
    tree: dict  # full topology
    format_version: int = field(default=FORMAT_VERSION)

    def get_source_hash(self, class_name: str) -> str | None:
        node = self.nodes.get(class_name)
        return node.get('hash') if isinstance(node, dict) else None

    def get_object_type(self, class_name: str) -> str | None:
        node = self.nodes.get(class_name)
        return node.get('object_type') if isinstance(node, dict) else None

    def get_call_deps(self) -> list[str]:
        root_node = next(iter(self.tree.values()), {})
        return sorted(root_node.get(JSONKeys.CALL_DEPENDENCIES, {}).keys())

    def get_inheritance_deps(self) -> list[str]:
        root_node = next(iter(self.tree.values()), {})
        return sorted(root_node.get(JSONKeys.INHERITANCE_DEPENDENCIES, {}).keys())

    def to_dict(self) -> dict[str, Any]:
        return {
            JSONKeys.FORMAT_VERSION: self.format_version,
            JSONKeys.NODES: self.nodes,
            JSONKeys.TREE: self.tree,
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=4), encoding='utf-8')

    @classmethod
    def from_dict(cls, dep_hash: str, data: dict[str, Any]) -> Self:
        return cls(
            dep_hash=dep_hash,
            nodes=data.get(JSONKeys.NODES, data.get('nodes', {})),
            tree=data.get(JSONKeys.TREE, data.get('tree', {})),
            format_version=data.get(JSONKeys.FORMAT_VERSION, FORMAT_VERSION),
        )

    @classmethod
    def from_file(cls, dep_hash: str, path: Path) -> Self:
        return cls.from_dict(dep_hash, json.loads(path.read_text(encoding='utf-8')))


@dataclass(slots=True)
class CodeState:
    source_hash: str
    class_name: str
    object_type: str
    registered_at: str  # ISO-8601
    format_version: int = field(default=FORMAT_VERSION)

    def to_dict(self) -> dict[str, Any]:
        return {
            JSONKeys.FORMAT_VERSION: self.format_version,
            JSONKeys.SOURCE_HASH: self.source_hash,
            JSONKeys.CLASS_NAME: self.class_name,
            JSONKeys.OBJECT_TYPE: self.object_type,
            JSONKeys.REGISTERED_AT: self.registered_at,
        }

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=4), encoding='utf-8')

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            source_hash=data[JSONKeys.SOURCE_HASH],
            class_name=data[JSONKeys.CLASS_NAME],
            object_type=data[JSONKeys.OBJECT_TYPE],
            registered_at=data[JSONKeys.REGISTERED_AT],
            format_version=data.get(JSONKeys.FORMAT_VERSION, FORMAT_VERSION),
        )

    @classmethod
    def from_file(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text(encoding='utf-8')))
