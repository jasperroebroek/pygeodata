from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClassNode:
    cls: type
    name: str
    color: str = '#ffffff'


@dataclass
class DependencyGraph:
    nodes: set[ClassNode] = field(default_factory=set)
    call_edges: set[tuple[ClassNode, ClassNode]] = field(default_factory=set)
    inheritance_edges: set[tuple[ClassNode, ClassNode]] = field(default_factory=set)


@dataclass(frozen=True)
class RuntimeNode:
    node_id: str
    cls: type
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    call_dependencies: tuple[type, ...] = ()
    inheritance_dependencies: tuple[type, ...] = ()


@dataclass(frozen=True)
class RuntimeParamEdge:
    src_id: str
    dst_id: str
    param_name: str


@dataclass
class RuntimeDependencyGraph:
    nodes: dict[str, RuntimeNode] = field(default_factory=dict)
    param_edges: set[RuntimeParamEdge] = field(default_factory=set)


@dataclass(frozen=True)
class SymbolTables:
    imported_objects: dict[str, str]
    module_aliases: dict[str, str]
    local_defs: set[str]