from __future__ import annotations

import functools
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pygeodata._version import FORMAT_VERSION
from pygeodata.ast import build_symbol_tables, get_module_ast_tree, get_source_ast_tree
from pygeodata.dependencies import (
    ClassFacts,
    call_dependencies,
    dependency_tree,
    dependency_tree_dict,
    inheritance_dependencies,
)
from pygeodata.enums import JSONKeys
from pygeodata.graph_types import ClassNode, DependencyGraph
from pygeodata.hash import calculate_ast_hash, calculate_dict_hash
from pygeodata.registries.registry_types import CodeState, TreeNode, TreeSnapshot

if TYPE_CHECKING:
    import ast

    from pygeodata.dependencies import FactsLookup
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
    def is_registered(cls) -> bool:
        return TrackedObject._registry.get(cls.get_class_name()) is cls

    @classmethod
    @functools.cache
    def get_facts(cls) -> ClassFacts:
        """The class as :mod:`pygeodata.dependencies` sees it; the tracked names are those registered now."""
        name = cls.get_class_name()
        class_tree = cls.get_source_ast_tree()
        return ClassFacts(
            name=name,
            object_type=cls.object_type.get_class_name(),
            source_hash=calculate_ast_hash(class_tree),
            call_deps=call_dependencies(
                class_tree,
                build_symbol_tables(get_module_ast_tree(cls)),
                set(cls._registry),
                name,
            ),
            bases=frozenset(
                base.get_class_name()
                for base in cls.__bases__
                if isinstance(base, type) and issubclass(base, TrackedObject) and base.is_registered()
            ),
        )

    @classmethod
    def _facts(cls) -> FactsLookup:
        """Facts of this class and of the registered classes, computed on first use."""
        own = cls.get_class_name()
        return lambda name: cls.get_facts() if name == own else cls._registry[name].get_facts()

    @classmethod
    def get_call_dependencies(cls) -> set[type[TrackedObject]]:
        return {cls._registry[name] for name in cls.get_facts().call_deps}

    @classmethod
    def get_inheritance_dependencies(cls) -> set[type[TrackedObject]]:
        return {cls._registry[name] for name in inheritance_dependencies(cls._facts(), cls.get_class_name())}

    @classmethod
    def get_all_dependencies(cls) -> set[type[TrackedObject]]:
        return cls.get_call_dependencies() | cls.get_inheritance_dependencies()

    @classmethod
    def has_dependencies(cls) -> bool:
        facts = cls.get_facts()
        return bool(facts.call_deps or facts.bases)

    @classmethod
    @functools.cache
    def _dependency_tree_dict(cls) -> dict:
        return dependency_tree_dict(cls._facts(), cls.get_class_name())

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
            nodes={name: TreeNode(**node) for name, node in d[JSONKeys.NODES].items()},
            call_edges=[(s, t) for s, t in d[JSONKeys.CALL_EDGES]],
            inheritance_edges=[(s, t) for s, t in d[JSONKeys.INHERITANCE_EDGES]],
            root_class=cls.get_class_name(),
            registered_at=datetime.now(UTC).isoformat(),
            format_version=FORMAT_VERSION,
        )

    @classmethod
    def get_dependency_graph(cls) -> DependencyGraph:
        """Build a typed graph of all :class:`TrackedObject` dependencies from this class."""
        tree = dependency_tree(cls._facts(), cls.get_class_name())

        def node(name: str) -> ClassNode:
            node_cls = cls if name == cls.get_class_name() else cls._registry[name]
            return ClassNode(cls=node_cls, name=name, color=node_cls.color)

        return DependencyGraph(
            nodes={node(name) for name in tree.nodes},
            call_edges={(node(s), node(t)) for s, t in tree.call_edges},
            inheritance_edges={(node(s), node(t)) for s, t in tree.inheritance_edges},
        )

    @classmethod
    def to_code_state(cls) -> CodeState:
        return CodeState(
            source_hash=cls.get_facts().source_hash,
            class_name=cls.get_class_name(),
            object_type=cls.object_type.get_class_name(),
            registered_at=datetime.now(UTC).isoformat(),
            format_version=FORMAT_VERSION,
        )

    @classmethod
    def clear_function_caches(cls) -> None:
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name, None)
            if callable(attr) and hasattr(attr, 'cache_clear'):
                attr.cache_clear()


def live_tree_hash(class_name: str) -> str | None:
    """The current dependency tree hash of a registered (imported) class; None for any other."""
    cls = TrackedObject.find_object_class(class_name)
    return cls.get_dependency_tree_hash() if cls is not None else None
