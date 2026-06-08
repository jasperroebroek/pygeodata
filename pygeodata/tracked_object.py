import ast
import functools
import json
from pathlib import Path
from typing import Any, ClassVar

from filelock import FileLock

from pygeodata.ast import (
    build_symbol_tables,
    find_names_in_ast_tree,
    get_module_ast_tree,
    get_source_ast_tree,
    get_source_code,
)
from pygeodata.config import JSONKeys, get_config
from pygeodata.graphs import plot_class_dependency_graph
from pygeodata.hash import calculate_cls_source_hash, calculate_dict_hash
from pygeodata.paths import RegistryPathResolver
from pygeodata.types import ClassNode, DependencyGraph


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
    def get_registry_dir(cls) -> Path:
        """Centralized storage: .source/module/path/ClassName/."""
        module_path_parts = cls.__module__.split('.')
        return Path(get_config().path_registry, *module_path_parts, cls.get_class_name())

    @classmethod
    def resolve_registry_paths(cls) -> RegistryPathResolver:
        return RegistryPathResolver.from_directory(cls.get_registry_dir())

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
    def _get_dependency_tree_recursive(cls, visited: frozenset[type] | None = None) -> dict[JSONKeys, Any] | str:
        if visited is None:
            visited = frozenset()

        if cls in visited:
            return 'circular'

        next_visited = visited | frozenset([cls])

        tree = {
            JSONKeys.CLASS_NAME: cls.get_class_name(),
            JSONKeys.OBJECT_TYPE: cls.object_type.get_class_name(),
            JSONKeys.SOURCE_HASH: calculate_cls_source_hash(cls),
            JSONKeys.CALL_DEPENDENCIES: {},
            JSONKeys.INHERITANCE_DEPENDENCIES: {},
        }

        for dep_cls in cls.get_call_dependencies():
            tree[JSONKeys.CALL_DEPENDENCIES][dep_cls.get_class_name()] = dep_cls._get_dependency_tree_recursive(
                next_visited,
            )

        for dep_cls in cls.get_inheritance_dependencies():
            tree[JSONKeys.INHERITANCE_DEPENDENCIES][dep_cls.get_class_name()] = dep_cls._get_dependency_tree_recursive(
                next_visited,
            )

        return tree

    @classmethod
    @functools.cache
    def get_dependency_tree(cls) -> dict[JSONKeys, Any] | str:
        """
        Build a nested dict representing the full dependency tree of this class.

        Cached via :func:`functools.cache`. Handles circular dependencies by substituting
        ``"circular"`` for repeated nodes.

        Returns
        -------
        dict
            Nested structure with keys ``class_name``, ``object_type``, ``source_hash``,
            ``call_dependencies``, and ``inheritance_dependencies``.
        """
        return cls._get_dependency_tree_recursive()

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
    def write_registry(cls) -> None:
        """Saves the current AST hash and writes the source code for inspection."""
        registry_paths = cls.resolve_registry_paths()
        registry_paths.mkdir()

        json_path = registry_paths.registry_path
        code_lock_path = registry_paths.lock_path

        with FileLock(code_lock_path, timeout=60):
            with Path.open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        JSONKeys.CLASS_NAME: cls.get_class_name(),
                        JSONKeys.OBJECT_TYPE: cls.object_type.get_class_name(),
                        JSONKeys.SOURCE_HASH: calculate_cls_source_hash(cls),
                        JSONKeys.DEPENDENCY_TREE_HASH: cls.get_dependency_tree_hash(),
                        JSONKeys.TREE: cls.get_dependency_tree(),
                    },
                    f,
                    indent=4,
                )

            code_path = registry_paths.code_path
            with Path.open(code_path, 'w', encoding='utf-8') as f:
                f.write(get_source_code(cls))

            if cls.has_dependencies():
                plot_class_dependency_graph(
                    cls_name=cls.get_class_name(),
                    graph_data=cls.get_dependency_graph(),
                    path=registry_paths.graph_path,
                    view=False,
                )

    @classmethod
    def read_registry(cls) -> dict[str, Any]:
        registry_file = cls.resolve_registry_paths().registry_path
        with Path.open(registry_file, encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def is_registry_valid(cls) -> bool:
        """
        Check whether the on-disk source registry matches the current class.

        Compares the ``dependency_tree_hash`` stored in ``.source/.../source.json``
        against :meth:`get_dependency_tree_hash`.

        Returns
        -------
        bool
        """
        registry_file = cls.resolve_registry_paths().registry_path
        if not registry_file.exists():
            return False
        registry = cls.read_registry()
        return registry.get(JSONKeys.DEPENDENCY_TREE_HASH) == cls.get_dependency_tree_hash()

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
