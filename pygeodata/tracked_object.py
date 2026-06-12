import ast
import functools
import json
from datetime import datetime, timezone
from typing import Any, ClassVar

from pygeodata.ast import (
    build_symbol_tables,
    find_names_in_ast_tree,
    get_module_ast_tree,
    get_source_ast_tree,
    get_source_code,
)
from pygeodata.config import get_config
from pygeodata.graph_types import ClassNode, DependencyGraph
from pygeodata.graphs import plot_class_dependency_graph
from pygeodata.hash import calculate_cls_source_hash, calculate_dict_hash
from pygeodata.paths import CodeRegistryResolver, TreeRegistryResolver
from pygeodata.registry_types import CodeState, TreeSnapshot


class TrackedObject:
    object_type: ClassVar[type['TrackedObject']]
    color: ClassVar[str] = '#f8f9fa'
    _registry: ClassVar[dict[str, type['TrackedObject']]] = {}

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

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def find_object_class(cls, name: str) -> type | None:
        return cls._registry.get(name, None)

    @classmethod
    def get_registered_objects(cls) -> set[type['TrackedObject']]:
        base = TrackedObject if cls is TrackedObject else cls.object_type
        return {r for r in cls._registry.values() if issubclass(r, base)}

    @classmethod
    def get_source_ast_tree(cls) -> ast.AST:
        return get_source_ast_tree(cls)

    @classmethod
    @functools.cache
    def get_call_dependencies(cls) -> set[type['TrackedObject']]:
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
    def get_inheritance_dependencies(cls) -> set[type['TrackedObject']]:
        return {base for base in cls.__mro__ if base != cls and base in cls._registry.values()}

    @classmethod
    @functools.cache
    def get_all_dependencies(cls) -> set[type['TrackedObject']]:
        return cls.get_call_dependencies() | cls.get_inheritance_dependencies()

    @classmethod
    def has_dependencies(cls) -> bool:
        return bool(cls.get_all_dependencies())

    @classmethod
    def _build_topology_subtree(
        cls,
        visited: frozenset[type] | None = None,
    ) -> tuple[dict, dict]:
        """Recursively build the topology subtree and node metadata for this class.

        Returns
        -------
        tuple[dict, dict]
            A ``(nodes, subtree)`` pair where ``nodes`` is a flat dict mapping
            ``class_name`` to ``{"hash": ..., "object_type": ...}`` for every class
            reachable from this one, and ``subtree`` is the pure-topology dict with
            ``call_dependencies`` and ``inheritance_dependencies`` keys (no metadata).
            When a cycle is detected the subtree for the repeated class is returned
            with empty dependency dicts to terminate the recursion.
        """
        if visited is None:
            visited = frozenset()

        nodes = {
            cls.get_class_name(): {
                'hash': calculate_cls_source_hash(cls),
                'object_type': cls.object_type.get_class_name(),
            },
        }

        if cls in visited:
            return nodes, {'call_dependencies': {}, 'inheritance_dependencies': {}}

        next_visited = visited | {cls}

        call_deps = {}
        for dep in cls.get_call_dependencies():
            dep_nodes, dep_subtree = dep._build_topology_subtree(next_visited)
            nodes.update(dep_nodes)
            call_deps[dep.get_class_name()] = dep_subtree

        inh_deps = {}
        for dep in cls.get_inheritance_dependencies():
            dep_nodes, dep_subtree = dep._build_topology_subtree(next_visited)
            nodes.update(dep_nodes)
            inh_deps[dep.get_class_name()] = dep_subtree

        return nodes, {'call_dependencies': call_deps, 'inheritance_dependencies': inh_deps}

    @classmethod
    @functools.cache
    def get_dependency_tree(cls) -> dict:
        """Build the full dependency tree for this class in ``{nodes, tree}`` format.

        Separates node metadata from topology: ``nodes`` is a flat dict with one entry
        per reachable class (no duplication), while ``tree`` is a fully-expanded nested
        topology where shared dependencies appear at every occurrence.

        Cached via :func:`functools.cache`. Any change in this class or any transitive
        dependency will produce a different :meth:`get_dependency_tree_hash`.

        Returns
        -------
        dict
            ``{"nodes": {class_name: {"hash": ..., "object_type": ...}, ...},
            "tree": {class_name: {"call_dependencies": {...}, "inheritance_dependencies": {...}}}}``
        """
        nodes, subtree = cls._build_topology_subtree()
        return {'nodes': nodes, 'tree': {cls.get_class_name(): subtree}}

    @classmethod
    def get_dependency_graph(cls) -> DependencyGraph:
        """
        Build a flat graph of all :class:`TrackedObject` dependencies from this class.

        Returns
        -------
        dict with keys:

        - ``nodes`` *(dict[type, type])*: All reachable classes.
        - ``call_edges`` *(set[tuple[type, type]])*: Edges from call dependencies.
        - ``inheritance_edges`` *(set[tuple[type, type]])*: Edges from inheritance.
        """
        nodes: set[ClassNode] = set()
        call_edges: set[tuple[ClassNode, ClassNode]] = set()
        inheritance_edges: set[tuple[ClassNode, ClassNode]] = set()

        def construct_class_node(cls: type['TrackedObject']) -> ClassNode:
            return ClassNode(cls=cls, name=cls.get_class_name(), color=cls.color)

        def visit(current_cls: type['TrackedObject']) -> None:
            cls_node = construct_class_node(current_cls)

            if cls_node in nodes:
                return

            nodes.add(cls_node)

            for dep_cls in current_cls.get_call_dependencies():
                dep_cls_node = construct_class_node(dep_cls)
                call_edges.add((cls_node, dep_cls_node))
                visit(dep_cls)

            for dep_cls in current_cls.get_inheritance_dependencies():
                dep_cls_node = construct_class_node(dep_cls)
                inheritance_edges.add((cls_node, dep_cls_node))
                visit(dep_cls)

        visit(cls)

        return DependencyGraph(
            nodes=nodes,
            call_edges=call_edges,
            inheritance_edges=inheritance_edges,
        )

    @classmethod
    @functools.cache
    def get_dependency_tree_hash(cls) -> str:
        """
        Compute a hash over the full dependency tree of this class.

        Hashes the JSON-serialized :meth:`get_dependency_tree` result, so any change
        in this class or any of its transitive dependencies will produce a different hash.

        Returns
        -------
        str
            A SHA-256 hex digest of the full dependency tree.
        """
        tree = cls.get_dependency_tree()
        return calculate_dict_hash(tree)

    @classmethod
    def _previously_active_source_hash(cls) -> str | None:
        """Return the source_hash from the most-recent ``source.json`` for this class."""
        from pygeodata.registry import SourceRegistry
        state = SourceRegistry.instance().latest_for_class(cls.get_class_name())
        return state.source_hash if state else None

    @classmethod
    def _write_code_registry(cls) -> None:
        """Write source code and metadata to ``.source/code/{source_hash}/``.

        ``source.py`` is content-addressed and written once.  ``source.json`` is written
        on first use and its mtime is refreshed whenever this hash is re-activated after
        a different hash was active (i.e. a revert).  This keeps the mtime accurate for
        version resolution in the Code browser.
        """
        source_hash = calculate_cls_source_hash(cls)
        resolver = CodeRegistryResolver.from_source_hash(source_hash)
        resolver.directory.mkdir(parents=True, exist_ok=True)

        if not resolver.source_path.exists():
            tmp = resolver.source_path.with_suffix('.tmp')
            tmp.write_text(get_source_code(cls), encoding='utf-8')
            tmp.replace(resolver.source_path)

        state = CodeState(
            source_hash=source_hash,
            class_name=cls.get_class_name(),
            object_type=cls.object_type.get_class_name(),
            registered_at=datetime.now(timezone.utc).isoformat(),
        )

        previously_active = cls._previously_active_source_hash()
        if not resolver.meta_path.exists() or (previously_active is not None and previously_active != source_hash):
            tmp = resolver.meta_path.with_suffix('.tmp')
            state.dump(tmp)
            tmp.replace(resolver.meta_path)

    @classmethod
    def _write_tree_registry(cls) -> None:
        """Write the dependency tree and graph to ``.source/snapshots/{dep_tree_hash}/``.

        Skips writing only when all expected files are present — ``tree.json``
        always, and ``graph.pdf`` when the class has dependencies.
        """
        dep_tree_hash = cls.get_dependency_tree_hash()
        resolver = TreeRegistryResolver.from_dep_tree_hash(dep_tree_hash)
        has_deps = cls.has_dependencies()
        resolver.directory.mkdir(parents=True, exist_ok=True)
        complete = resolver.tree_path.exists() and (not has_deps or resolver.graph_path.exists())
        if not complete:
            dep_tree = cls.get_dependency_tree()
            tree = TreeSnapshot(
                dep_hash=dep_tree_hash,
                nodes=dep_tree['nodes'],
                tree=dep_tree['tree'],
            )
            tmp = resolver.tree_path.with_suffix('.tmp')
            tree.dump(tmp)
            tmp.replace(resolver.tree_path)
            if has_deps:
                plot_class_dependency_graph(
                    cls_name=cls.get_class_name(),
                    graph_data=cls.get_dependency_graph(),
                    path=resolver.graph_path,
                    view=False,
                )

    @classmethod
    def write_registry(cls) -> None:
        """Write both the source code snapshot and the dependency tree snapshot to disk.

        Delegates to :meth:`_write_code_registry` and :meth:`_write_tree_registry`. Both
        stores are content-addressed and append-only — existing entries are never overwritten.
        """
        cls._write_code_registry()
        cls._write_tree_registry()

    @classmethod
    def read_registry(cls) -> dict[str, Any]:
        """Read the dependency tree snapshot for this class from disk.

        Returns
        -------
        dict
            The ``{nodes, tree}`` dict stored in ``.source/snapshots/{dep_tree_hash}/tree.json``.

        Raises
        ------
        FileNotFoundError
            If the snapshot has not been written yet.
        """
        tree_path = TreeRegistryResolver.from_dep_tree_hash(cls.get_dependency_tree_hash()).tree_path
        with tree_path.open(encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def is_registry_valid(cls) -> bool:
        """Check whether the on-disk registry is complete for the current class state.

        Returns ``True`` when both ``_write_code_registry`` and ``_write_tree_registry``
        would be no-ops — i.e. all expected files exist for the current source hash and
        dependency tree hash.

        Returns
        -------
        bool
        """
        source_hash = calculate_cls_source_hash(cls)
        code_resolver = CodeRegistryResolver.from_source_hash(source_hash)
        if not (code_resolver.source_path.exists() and code_resolver.meta_path.exists()):
            return False
        tree_resolver = TreeRegistryResolver.from_dep_tree_hash(cls.get_dependency_tree_hash())
        has_deps = cls.has_dependencies()
        return tree_resolver.tree_path.exists() and (not has_deps or tree_resolver.graph_path.exists())

    @classmethod
    def update_registry(cls, visited: frozenset[type] | None = None) -> None:
        """
        Ensure this class and all its dependencies have a valid source registry on disk.

        Calls :meth:`save_source_code_and_hash` if the registry is stale, then
        recursively initializes registries for all dependencies.
        """
        if visited is None:
            visited = frozenset()

        if cls in visited:
            return

        next_visited = visited | frozenset([cls])

        if not cls.is_registry_valid():
            cls.write_registry()

        for dep_class in cls.get_all_dependencies():
            dep_class.update_registry(next_visited)

    @classmethod
    def clear_function_caches(cls) -> None:
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if callable(attr) and hasattr(attr, 'cache_clear'):
                attr.cache_clear()
