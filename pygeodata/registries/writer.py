from __future__ import annotations

from typing import TYPE_CHECKING

from pygeodata.ast import get_source_code
from pygeodata.fileio import write_bytes_atomic
from pygeodata.graphs import plot_class_dependency_graph
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryPathConstructor, TreeRegistryPathConstructor
from pygeodata.registries.registry import SourceRegistry

if TYPE_CHECKING:
    from pygeodata.tracked_object import TrackedObject


def _is_code_registry_current(cls: type[TrackedObject], registry: SourceRegistry) -> bool:
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


def _is_tree_registry_complete(cls: type[TrackedObject]) -> bool:
    resolver = TreeRegistryPathConstructor.from_dep_tree_hash(cls.get_dependency_tree_hash())
    return resolver.tree_path.exists() and (not cls.has_dependencies() or resolver.graph_path.exists())


def _write_code_registry(cls: type[TrackedObject], registry: SourceRegistry) -> None:
    if _is_code_registry_current(cls, registry):
        return
    state = cls.to_code_state()
    resolver = CodeRegistryPathConstructor.from_source_hash(state.source_hash)
    resolver.directory.mkdir(parents=True, exist_ok=True)
    if not resolver.source_path.exists():
        write_bytes_atomic(resolver.source_path, get_source_code(cls).encode('utf-8'))
    state.dump(resolver.meta_path)


def _write_tree_registry(cls: type[TrackedObject]) -> None:
    if _is_tree_registry_complete(cls):
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


def write_registry(cls: type[TrackedObject], registry: SourceRegistry | None = None) -> None:
    _write_code_registry(cls, registry if registry is not None else SourceRegistry())
    _write_tree_registry(cls)


def is_registry_valid(cls: type[TrackedObject], registry: SourceRegistry | None = None) -> bool:
    registry = registry if registry is not None else SourceRegistry()
    return _is_code_registry_current(cls, registry) and _is_tree_registry_complete(cls)


def update_registry(cls: type[TrackedObject]) -> None:
    """Ensure ``cls`` and all its dependencies have a current registry on disk."""
    registry = SourceRegistry()
    done: set[type[TrackedObject]] = set()
    pending: list[type[TrackedObject]] = [cls]
    while pending:
        tracked_cls = pending.pop()
        if tracked_cls in done:
            continue
        done.add(tracked_cls)
        write_registry(tracked_cls, registry)
        pending.extend(tracked_cls.get_all_dependencies())
