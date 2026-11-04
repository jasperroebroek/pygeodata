from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from pygeodata.dependencies import inheritance_dependencies
from pygeodata.enums import Staleness
from pygeodata.paths import CodeRegistryPathConstructor, TreeRegistryPathConstructor
from pygeodata.registries.registry import EntryRegistry, SourceRegistry, TreeRegistry
from pygeodata.staleness import class_staleness
from pygeodata.tracked_object import TrackedObject
from pygeodata.views.types import ClassInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from pygeodata.analysis import ProjectAnalysis
    from pygeodata.dependencies import ClassFacts
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
        return ClassInfo(class_name=class_name, object_type=None, known=False)

    source_hash = latest.source_hash
    code_resolver = CodeRegistryPathConstructor.from_source_hash(source_hash)

    trees = trees if trees is not None else TreeRegistry()
    snapshot = trees.get_snapshot_for_source_hash(class_name, source_hash)
    info = ClassInfo(
        class_name=class_name,
        object_type=latest.object_type,
        known=False,
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


def current_classes(
    facts: Mapping[str, ClassFacts],
    tree_hashes: Mapping[str, str],
    entry_registry: EntryRegistry,
) -> dict[str, ClassInfo]:
    """The classes whose current code is known, and whether their cache entries were made by it.

    A class is stale only when it has entries and none of them carries its current hash.
    """
    classes: dict[str, ClassInfo] = {}
    for name, class_facts in facts.items():
        code = CodeRegistryPathConstructor.from_source_hash(class_facts.source_hash)
        tree = TreeRegistryPathConstructor.from_dep_tree_hash(tree_hashes[name])
        records = [entry_registry.entries[h].record for h in entry_registry.get_state_hashes(name)]
        classes[name] = ClassInfo(
            class_name=name,
            object_type=class_facts.object_type,
            known=True,
            call_dependency_names=sorted(class_facts.call_deps),
            inheritance_dependency_names=sorted(inheritance_dependencies(facts.__getitem__, name)),
            class_source_path=_existing(code.source_path),
            class_graph_path=_existing(tree.graph_path),
            class_registry_path=_existing(code.meta_path),
            class_tree_path=_existing(tree.tree_path),
            source_stale=bool(records) and all(r.source_hash != class_facts.source_hash for r in records),
            deps_stale=bool(records) and all(r.dependency_tree_hash != tree_hashes[name] for r in records),
        )
    return classes


def imported_classes(entry_registry: EntryRegistry) -> dict[str, ClassInfo]:
    registry = TrackedObject._registry
    return current_classes(
        {name: cls.get_facts() for name, cls in registry.items()},
        {name: cls.get_dependency_tree_hash() for name, cls in registry.items()},
        entry_registry,
    )


def source_classes(analysis: ProjectAnalysis, entry_registry: EntryRegistry) -> dict[str, ClassInfo]:
    return current_classes(analysis.facts, analysis.tree_hashes, entry_registry)


def merge_unknown_classes(
    classes: dict[str, ClassInfo],
    entry_registry: EntryRegistry,
    current: Callable[[str], str | None],
    version_registry: VersionRegistry | None = None,
    src: SourceRegistry | None = None,
    trees: TreeRegistry | None = None,
) -> dict[str, ClassInfo]:
    """Add the classes that have cache entries but whose current code is unknown, from their snapshots."""
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
            deps_stale = class_staleness(class_name, dep_hashes, current, version_registry) is Staleness.STALE
        merged[class_name] = dataclasses.replace(
            info,
            object_type=info.object_type or entry_registry.get_object_type(class_name),
            deps_stale=deps_stale,
        )
    return merged
