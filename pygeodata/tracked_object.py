from __future__ import annotations

import ast
import functools
from typing import Any, ClassVar

from pygeodata.ast import (
    build_symbol_tables,
    find_names_in_ast_tree,
    get_module_ast_tree,
    get_source_ast_tree,
    get_source_code,
)
from pygeodata.enums import JSONKeys
from pygeodata.fileio import write_bytes_atomic
from pygeodata.graph_types import ClassNode, DependencyGraph
from pygeodata.graphs import plot_class_dependency_graph
from pygeodata.hash import calculate_cls_source_hash, calculate_dict_hash
from pygeodata.paths import CodeRegistryPathConstructor, TreeRegistryPathConstructor
from pygeodata.registries.registry import SourceRegistry
from pygeodata.registries.registry_types import CodeState, TreeSnapshot
from pygeodata.spec import SpatialSpec


class TrackedObject:
    object_type: ClassVar[type[TrackedObject]]
    color: ClassVar[str] = '#f8f9fa'
    _sort_params: ClassVar[tuple[str, ...]] = ()
    _registry: ClassVar[dict[str, type[TrackedObject]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        module = vars(cls).get('__module__', '')
        top_level = module.split('.', maxsplit=1)[0]

        if top_level == 'pygeodata':
            cls.object_type = cls
            return

        parent = cls.__mro__[1]
        cls.object_type = getattr(parent, 'object_type', parent)

        name = cls.get_class_name()
        existing = TrackedObject._registry.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f'Duplicate TrackedObject class name {name!r}: '
                f'{existing.__module__}.{existing.get_class_name()} and '
                f'{cls.__module__}.{cls.get_class_name()}',
            )
        TrackedObject._registry[name] = cls

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.get_class_name()}({", ".join(parts)})'

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    def format_for_display(self) -> str:
        return self.get_class_name()

    def format_as_json(self, spec: SpatialSpec | None = None) -> Any:
        from pygeodata.formatting.json import format_json  # keeps numpy out of the CLI import path

        return {
            JSONKeys.CLASS_NAME: self.get_class_name(),
            JSONKeys.PARAMS: format_json(self.get_params(), spec=spec),
            JSONKeys.INSTANCE_HASH: self.get_instance_hash(),
        }

    def get_params(self) -> dict[str, Any]:
        """
        Return the instance parameters of the object.

        Inspects ``vars(self)`` and filters out underscore-prefixed attributes and
        reserved names (``processor``, ``driver``, ``ext``, ...).

        Returns
        -------
        dict[str, Any]
            A dict of parameter name-value pairs.
        """
        params = {}
        for key, value in vars(self).items():
            if key in ('name', 'class_name', 'processor', 'driver', 'process', 'load', 'ext', '_load', '_process'):
                continue
            if key.startswith('_'):
                continue

            if key in self._sort_params and isinstance(value, (list, tuple)):
                parsed_value = type(value)(sorted(value, key=repr))
            else:
                parsed_value = value

            params.update({key: parsed_value})
        return params

    def get_params_as_json(self, spec: SpatialSpec | None = None) -> dict[str, Any]:
        from pygeodata.formatting.json import format_json  # keeps numpy out of the CLI import path

        return format_json(self.get_params(), spec=spec)  # type: ignore[return-value]

    def _get_params_state(self) -> dict[str, Any]:
        from pygeodata.formatting.json import format_json  # keeps numpy out of the CLI import path

        return {k: format_json(v) for k, v in self.get_params().items()}

    def get_params_hash(self) -> str:
        """Hash of params only — no dep_tree_hash. Groups entries with same params across dep versions."""
        return calculate_dict_hash({JSONKeys.PARAMS: self._get_params_state()})

    def get_instance_state(self) -> dict[str, Any]:
        """
        Build the state dict that identifies this instance, prior to hashing.

        Covers the class name, the dependency tree hash and the instance params. The class name
        disambiguates co-output siblings that share an identical dependency tree. Subclasses that
        carry identity beyond their own source and params override this and extend the result
        — see :meth:`~pygeodata.artifact.Artifact.get_instance_state`, which adds the
        processor source hash.

        Returns
        -------
        dict[str, Any]
            The state hashed by :meth:`get_instance_hash`.
        """
        return {
            JSONKeys.CLASS_NAME: self.get_class_name(),
            JSONKeys.DEPENDENCY_TREE_HASH: self.get_dependency_tree_hash(),
            JSONKeys.PARAMS: self._get_params_state(),
        }

    def get_instance_hash(self) -> str:
        """Hash of class code and params — spec-independent. Stable identifier for this instance."""
        return calculate_dict_hash(self.get_instance_state())

    @classmethod
    def find_object_class(cls, name: str) -> type | None:
        return cls._registry.get(name, None)

    @classmethod
    def get_registered_objects(cls) -> set[type[TrackedObject]]:
        base = TrackedObject if cls is TrackedObject else cls.object_type
        return {r for r in cls._registry.values() if issubclass(r, base)}

    @classmethod
    def get_source_ast_tree(cls) -> ast.AST:
        return get_source_ast_tree(cls)

    @classmethod
    @functools.cache
    def get_call_dependencies(cls) -> set[type[TrackedObject]]:
        cls_module_tree = get_module_ast_tree(cls)
        tables = build_symbol_tables(cls_module_tree)

        cls_tree = cls.get_source_ast_tree()
        names = find_names_in_ast_tree(
            tree=cls_tree,
            tables=tables,
            valid_names=set(cls._registry),
            exclude_bases=True,
        )

        return {
            dep_cls for name in names if (dep_cls := cls.find_object_class(name)) is not None and dep_cls is not cls
        }

    @classmethod
    @functools.cache
    def get_inheritance_dependencies(cls) -> set[type[TrackedObject]]:
        return {base for base in cls.__mro__ if base != cls and base in cls._registry.values()}

    @classmethod
    @functools.cache
    def get_all_dependencies(cls) -> set[type[TrackedObject]]:
        return cls.get_call_dependencies() | cls.get_inheritance_dependencies()

    @classmethod
    def has_dependencies(cls) -> bool:
        return bool(cls.get_all_dependencies())

    @classmethod
    def _traverse(cls, make_node):
        """DFS over the dependency graph, returning (nodes, call_edges, inheritance_edges).

        ``make_node`` converts a TrackedObject class to whatever node representation
        the caller needs — e.g. a ClassNode for graphs, a str name for snapshots.
        """
        nodes = set()
        call_edges = set()
        inheritance_edges = set()

        def visit(current_cls):
            node = make_node(current_cls)
            if node in nodes:
                return
            nodes.add(node)
            for dep_cls in current_cls.get_call_dependencies():
                dep_node = make_node(dep_cls)
                call_edges.add((node, dep_node))
                visit(dep_cls)
            for dep_cls in current_cls.get_inheritance_dependencies():
                dep_node = make_node(dep_cls)
                inheritance_edges.add((node, dep_node))
                visit(dep_cls)

        visit(cls)
        return nodes, call_edges, inheritance_edges

    @classmethod
    @functools.cache
    def _dependency_tree_dict(cls) -> dict:
        """Raw tree data dict — shared by get_dependency_tree_hash and get_dependency_tree."""
        names, call_edges, inheritance_edges = cls._traverse(lambda c: c.get_class_name())
        return {
            JSONKeys.NODES: {
                name: {
                    'hash': calculate_cls_source_hash(cls._registry[name]),
                    JSONKeys.OBJECT_TYPE: cls._registry[name].object_type.get_class_name(),
                }
                for name in names
            },
            JSONKeys.CALL_EDGES: sorted([s, t] for s, t in call_edges),
            JSONKeys.INHERITANCE_EDGES: sorted([s, t] for s, t in inheritance_edges),
        }

    @classmethod
    @functools.cache
    def get_dependency_tree_hash(cls) -> str:
        """SHA-256 digest of the full dependency tree — changes when any transitive dep changes."""
        return calculate_dict_hash(cls._dependency_tree_dict())

    @classmethod
    @functools.cache
    def get_dependency_tree(cls) -> TreeSnapshot:
        """Build the serializable dependency snapshot for this class."""
        d = cls._dependency_tree_dict()
        return TreeSnapshot(
            dependency_tree_hash=cls.get_dependency_tree_hash(),
            nodes=d[JSONKeys.NODES],
            call_edges=d[JSONKeys.CALL_EDGES],
            inheritance_edges=d[JSONKeys.INHERITANCE_EDGES],
            root_class=cls.get_class_name(),
        )

    @classmethod
    def get_dependency_graph(cls) -> DependencyGraph:
        """Build a typed graph of all :class:`TrackedObject` dependencies from this class."""
        nodes, call_edges, inheritance_edges = cls._traverse(
            lambda c: ClassNode(cls=c, name=c.get_class_name(), color=c.color),
        )
        return DependencyGraph(nodes=nodes, call_edges=call_edges, inheritance_edges=inheritance_edges)

    @classmethod
    def to_code_state(cls) -> CodeState:
        return CodeState(
            source_hash=calculate_cls_source_hash(cls),
            class_name=cls.get_class_name(),
            object_type=cls.object_type.get_class_name(),
        )

    @classmethod
    def _is_code_registry_current(cls, registry: SourceRegistry) -> bool:
        """``source.py`` exists and this source hash is the latest registered state of the class.

        The second condition also catches a revert: an older hash that exists on disk but is not
        the latest registration.
        """
        source_hash = calculate_cls_source_hash(cls)
        latest = registry.get_latest_state_for_class(cls.get_class_name())
        return (
            CodeRegistryPathConstructor.from_source_hash(source_hash).source_path.exists()
            and latest is not None
            and latest.source_hash == source_hash
        )

    @classmethod
    def _is_tree_registry_complete(cls) -> bool:
        resolver = TreeRegistryPathConstructor.from_dep_tree_hash(cls.get_dependency_tree_hash())
        return resolver.tree_path.exists() and (not cls.has_dependencies() or resolver.graph_path.exists())

    @classmethod
    def _write_code_registry(cls, registry: SourceRegistry) -> None:
        """Write ``source.py`` once and (re-)register ``source.json`` when this hash is not the latest."""
        if cls._is_code_registry_current(registry):
            return
        state = cls.to_code_state()
        resolver = CodeRegistryPathConstructor.from_source_hash(state.source_hash)
        resolver.directory.mkdir(parents=True, exist_ok=True)
        if not resolver.source_path.exists():
            write_bytes_atomic(resolver.source_path, get_source_code(cls).encode('utf-8'))
        state.dump(resolver.meta_path)

    @classmethod
    def _write_tree_registry(cls) -> None:
        if cls._is_tree_registry_complete():
            return
        resolver = TreeRegistryPathConstructor.from_dep_tree_hash(cls.get_dependency_tree_hash())
        resolver.directory.mkdir(parents=True, exist_ok=True)
        cls.get_dependency_tree().dump(resolver.tree_path)
        if cls.has_dependencies():
            plot_class_dependency_graph(
                cls_name=cls.get_class_name(),
                graph_data=cls.get_dependency_graph(),
                path=resolver.graph_path,
                view=False,
            )

    @classmethod
    def write_registry(cls, registry: SourceRegistry | None = None) -> None:
        cls._write_code_registry(registry if registry is not None else SourceRegistry())
        cls._write_tree_registry()

    @classmethod
    def is_registry_valid(cls, registry: SourceRegistry | None = None) -> bool:
        registry = registry if registry is not None else SourceRegistry()
        return cls._is_code_registry_current(registry) and cls._is_tree_registry_complete()

    @classmethod
    def update_registry(cls) -> None:
        """Ensure this class and all its dependencies have a current registry on disk."""
        registry = SourceRegistry()
        done: set[type[TrackedObject]] = set()
        pending: list[type[TrackedObject]] = [cls]
        while pending:
            tracked_cls = pending.pop()
            if tracked_cls in done:
                continue
            done.add(tracked_cls)
            tracked_cls.write_registry(registry)
            pending.extend(tracked_cls.get_all_dependencies())

    @classmethod
    def clear_function_caches(cls) -> None:
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if callable(attr) and hasattr(attr, 'cache_clear'):
                attr.cache_clear()
