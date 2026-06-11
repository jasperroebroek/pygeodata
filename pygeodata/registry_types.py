from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from pygeodata.config import FORMAT_VERSION, JSONKeys


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
        path.write_text(json.dumps(self.to_dict()), encoding='utf-8')

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
        path.write_text(json.dumps(self.to_dict()), encoding='utf-8')

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


@dataclass(slots=True)
class VersionInfo:
    mtime: str
    class_name: str
