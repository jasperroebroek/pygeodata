from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from pygeodata.enums import Staleness
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryPathConstructor, TreeRegistryPathConstructor
from pygeodata.registries.registry import EntryRegistry, SourceRegistry, TreeRegistry
from pygeodata.staleness import class_staleness
from pygeodata.tracked_object import TrackedObject
from pygeodata.views.types import ClassInfo

if TYPE_CHECKING:
    from pathlib import Path

    from pygeodata.registries.versioning import VersionRegistry


def _existing(path: Path, resolve: bool = False) -> str | None:
    if not path.exists():
        return None
    return str(path.resolve() if resolve else path)


def source_info_from_disk(
    class_name: str,
    src: SourceRegistry | None = None,
    trees: TreeRegistry | None = None,
) -> ClassInfo:
    """Read class metadata from the content-addressed code registry.

    Uses :class:`SourceRegistry` to find the most recent snapshot for
    ``class_name``, then searches ``snapshots/*/tree.json`` for a dependency
    tree that has this class as root.

    Returns a default :class:`ClassInfo` if nothing is found.
    """
    registry = src if src is not None else SourceRegistry()
    latest = registry.get_latest_state_for_class(class_name)
    if latest is None:
        return ClassInfo(class_name=class_name, object_type=None, loaded=False)

    source_hash = latest.source_hash
    code_resolver = CodeRegistryPathConstructor.from_source_hash(source_hash)

    trees = trees if trees is not None else TreeRegistry()
    snapshot = trees.get_snapshot_for_source_hash(class_name, source_hash)
    info = ClassInfo(
        class_name=class_name,
        object_type=latest.object_type,
        loaded=False,
        class_source_path=_existing(code_resolver.source_path, resolve=True),
        class_registry_path=_existing(code_resolver.meta_path, resolve=True),
    )
    if snapshot is None:
        return info

    tree_path = trees.get_tree_path(snapshot.dependency_tree_hash)
    return dataclasses.replace(
        info,
        call_dependency_names=trees.get_call_dependencies(snapshot.dependency_tree_hash),
        inheritance_dependency_names=trees.get_inheritance_dependencies(snapshot.dependency_tree_hash),
        class_graph_path=_existing(TreeRegistryPathConstructor(tree_path.parent).graph_path, resolve=True),
        class_tree_path=str(tree_path.resolve()),
    )


def discover_loaded_classes() -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}

    for class_name, cls in TrackedObject._registry.items():
        call_dependency_names = sorted(dep.get_class_name() for dep in cls.get_call_dependencies())
        inheritance_dependency_names = sorted(dep.get_class_name() for dep in cls.get_inheritance_dependencies())

        live_source_hash = calculate_cls_source_hash(cls)
        code_resolver = CodeRegistryPathConstructor.from_source_hash(live_source_hash)
        tree_resolver = TreeRegistryPathConstructor.from_dep_tree_hash(cls.get_dependency_tree_hash())

        # Source is stale when the current source hash has no code snapshot yet
        source_stale = not code_resolver.exists()
        # Deps are stale when the current dep tree hash has no snapshot yet
        deps_stale = not tree_resolver.exists()

        classes[class_name] = ClassInfo(
            class_name=class_name,
            object_type=cls.object_type.get_class_name(),
            loaded=True,
            call_dependency_names=call_dependency_names,
            inheritance_dependency_names=inheritance_dependency_names,
            class_source_path=_existing(code_resolver.source_path),
            class_graph_path=_existing(tree_resolver.graph_path),
            class_registry_path=_existing(code_resolver.meta_path),
            class_tree_path=_existing(tree_resolver.tree_path),
            source_stale=source_stale,
            deps_stale=deps_stale,
        )

    return classes


def merge_unloaded_classes(
    classes: dict[str, ClassInfo],
    entry_registry: EntryRegistry,
    version_registry: VersionRegistry | None = None,
    src: SourceRegistry | None = None,
    trees: TreeRegistry | None = None,
) -> dict[str, ClassInfo]:
    """Add the classes that have cache entries but are not loaded, from their registered source."""
    merged = dict(classes)
    for class_name in entry_registry.class_names:
        if class_name in merged:
            continue
        info = source_info_from_disk(class_name, src=src, trees=trees)
        deps_stale = False
        if version_registry is not None:
            dep_hashes = (
                entry_registry.entries[h].record.dependency_tree_hash
                for h in entry_registry.get_state_hashes(class_name)
            )
            deps_stale = class_staleness(class_name, dep_hashes, version_registry) is Staleness.STALE
        merged[class_name] = dataclasses.replace(
            info,
            object_type=info.object_type or entry_registry.get_object_type(class_name),
            deps_stale=deps_stale,
        )
    return merged
