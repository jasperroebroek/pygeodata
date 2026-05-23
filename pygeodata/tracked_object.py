import ast
import functools
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any, ClassVar

from filelock import FileLock

from pygeodata.config import get_config


class TrackedObject:
    object_type: ClassVar[type['TrackedObject']]
    _registry: ClassVar[dict[str, type['TrackedObject']]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        module: str = vars(cls).get('__module__', '')
        top_level = module.split('.', maxsplit=1)[0]

        if top_level == 'pygeodata':
            cls.object_type = cls
        else:
            parent = cls.__mro__[1]
            cls.object_type = getattr(parent, 'object_type', parent)

        if top_level != 'pygeodata':
            TrackedObject._registry[cls.get_class_name()] = cls

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def find_object_class(cls, name: str) -> type | None:
        return cls._registry.get(name, None)

    @classmethod
    def get_source_registry_dir(cls) -> Path:
        """Centralized storage: .source/module/path/ClassName/."""
        module_path = cls.__module__.replace('.', '/')
        return get_config().path_source_registry / module_path / cls.get_class_name()

    @classmethod
    def get_source_registry_path(cls) -> Path:
        return cls.get_source_registry_dir() / 'source.json'

    @classmethod
    def get_source_registry_code_path(cls) -> Path:
        return cls.get_source_registry_dir() / 'source.py'

    @classmethod
    @functools.cache
    def get_source_ast_tree(cls) -> ast.AST:
        try:
            source = textwrap.dedent(inspect.getsource(cls))
            return ast.parse(source)
        except (TypeError, OSError) as err:
            raise OSError(
                'AST Parsing failed. Caching is disabled. You are likely in a REPL/Notebook environment. Use standard .py files.',
            ) from err

    @classmethod
    @functools.cache
    def get_source_code(cls) -> str:
        return ast.unparse(cls.get_source_ast_tree())

    @classmethod
    @functools.cache
    def get_source_hash(cls) -> str:
        tree = cls.get_source_ast_tree()
        return hashlib.sha256(ast.dump(tree).encode()).hexdigest()

    @classmethod
    @functools.cache
    def get_call_dependencies(cls) -> set[type['TrackedObject']]:
        tree = cls.get_source_ast_tree()

        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)

        call_dependencies = set()
        for name in called_names:
            if name == cls.get_class_name():
                continue
            dep_cls = cls.find_object_class(name)
            if dep_cls is not None:
                call_dependencies.add(dep_cls)

        return call_dependencies

    @classmethod
    @functools.cache
    def get_inheritance_dependencies(cls) -> set[type['TrackedObject']]:
        return {base for base in cls.__mro__ if base != cls and base in cls._registry.values()}

    @classmethod
    @functools.cache
    def get_all_dependencies(cls) -> set[type['TrackedObject']]:
        return cls.get_call_dependencies() | cls.get_inheritance_dependencies()

    @classmethod
    def _get_dependency_tree_recursive(cls, visited: frozenset[type] | None = None) -> dict[str, Any] | str:
        if visited is None:
            visited = frozenset()

        if cls in visited:
            return 'circular'

        next_visited = visited | frozenset([cls])

        tree = {
            'class_name': cls.get_class_name(),
            'class_type': cls.object_type.get_class_name(),
            'source_hash': cls.get_source_hash(),
            'call_dependencies': {},
            'inheritance_dependencies': {},
        }

        for dep_cls in cls.get_call_dependencies():
            tree['call_dependencies'][dep_cls.get_class_name()] = dep_cls._get_dependency_tree_recursive(next_visited)

        for dep_cls in cls.get_inheritance_dependencies():
            tree['inheritance_dependencies'][dep_cls.get_class_name()] = dep_cls._get_dependency_tree_recursive(
                next_visited,
            )

        return tree

    @classmethod
    @functools.cache
    def get_dependency_tree(cls) -> dict[str, Any] | str:
        """
        Build a nested dict representing the full dependency tree of this class.

        Cached via :func:`functools.cache`. Handles circular dependencies by substituting
        ``"circular"`` for repeated nodes.

        Returns
        -------
        dict
            Nested structure with keys ``class_name``, ``source_hash``,
            ``call_dependencies``, and ``inheritance_dependencies``.
        """
        return cls._get_dependency_tree_recursive()

    @classmethod
    def get_dependency_graph(cls) -> dict[str, Any]:
        """
        Build a flat graph of all :class:`TrackedObject` dependencies from this class.

        Returns
        -------
        dict with keys:

        - ``nodes`` *(dict[type, type])*: All reachable classes.
        - ``call_edges`` *(set[tuple[type, type]])*: Edges from call dependencies.
        - ``inheritance_edges`` *(set[tuple[type, type]])*: Edges from inheritance.
        """
        nodes: dict[type, type] = {}
        call_edges: set[tuple[type, type]] = set()
        inheritance_edges: set[tuple[type, type]] = set()

        def visit(current_cls: type['TrackedObject']) -> None:
            if current_cls in nodes:
                return

            nodes[current_cls] = current_cls

            for dep_cls in current_cls.get_call_dependencies():
                call_edges.add((current_cls, dep_cls))
                visit(dep_cls)

            for dep_cls in current_cls.get_inheritance_dependencies():
                inheritance_edges.add((current_cls, dep_cls))
                visit(dep_cls)

        visit(cls)

        return {
            'nodes': nodes,
            'call_edges': call_edges,
            'inheritance_edges': inheritance_edges,
        }

    @classmethod
    @functools.cache
    def get_source_hierarchy_hash(cls) -> str:
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
        return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()

    @classmethod
    def save_source_code_and_hash(cls) -> None:
        """Saves the current AST hash and writes the source code for inspection."""
        from pygeodata.visualisations import plot_class_dependency_graph

        tree = cls.get_dependency_tree()

        json_file = cls.get_source_registry_path()
        json_file.parent.mkdir(parents=True, exist_ok=True)

        code_lock_path = cls.get_source_registry_dir() / 'source.lock'
        with FileLock(code_lock_path, timeout=60):
            with Path.open(json_file, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'source_hash': cls.get_source_hash(),
                        'source_hierarchy_hash': cls.get_source_hierarchy_hash(),
                        'tree': tree,
                    },
                    f,
                    indent=4,
                )

            py_file = cls.get_source_registry_code_path()
            with Path.open(py_file, 'w', encoding='utf-8') as f:
                f.write(cls.get_source_code())

            if len(tree['inheritance_dependencies']) + len(tree['call_dependencies']) > 0:
                plot_class_dependency_graph(cls, view=False)

    @classmethod
    def is_source_registry_valid(cls) -> bool:
        """
        Check whether the on-disk source registry matches the current class.

        Compares the ``source_hierarchy_hash`` stored in ``.source/.../source.json``
        against :meth:`get_source_hierarchy_hash`.

        Returns
        -------
        bool
        """
        registry_file = cls.get_source_registry_path()
        if not registry_file.exists():
            return False

        with Path.open(registry_file, encoding='utf-8') as f:
            saved_state = json.load(f)

        return saved_state.get('source_hierarchy_hash') == cls.get_source_hierarchy_hash()

    @classmethod
    def init_source_registry(cls, visited: frozenset[type] | None = None) -> None:
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

        if not cls.is_source_registry_valid():
            cls.save_source_code_and_hash()

        for dep_class in cls.get_all_dependencies():
            dep_class.init_source_registry(next_visited)
