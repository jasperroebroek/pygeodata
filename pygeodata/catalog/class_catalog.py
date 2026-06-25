from pygeodata.catalog.types import ClassInfo, EntryInfo, RegistryClassInfo
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryPathConstructor, TreeRegistryPathConstructor
from pygeodata.registries.registry import EntryRegistry, SourceRegistry, TreeRegistry
from pygeodata.catalog.io_utils import existing_path_str
from pygeodata.tracked_object import TrackedObject
from pygeodata.registries.versioning import VersionRegistry


def source_info_from_disk(
    class_name: str,
    src: SourceRegistry | None = None,
    trees: TreeRegistry | None = None,
) -> RegistryClassInfo:
    """Read class metadata from the content-addressed code registry.

    Uses :class:`SourceRegistry` to find the most recent snapshot for
    ``class_name``, then searches ``snapshots/*/tree.json`` for a dependency
    tree that has this class as root.

    Returns a default :class:`RegistryClassInfo` if nothing is found.
    """
    registry = src if src is not None else SourceRegistry()
    latest = registry.get_latest_state_for_class(class_name)
    if latest is None:
        return RegistryClassInfo()

    source_hash = latest.source_hash
    code_resolver = CodeRegistryPathConstructor.from_source_hash(source_hash) if source_hash else None
    meta_path = code_resolver.meta_path if code_resolver else None

    call_dep_names: list[str] = []
    inh_dep_names: list[str] = []
    graph_path_str: str | None = None
    tree_path_str: str | None = None

    trees = trees if trees is not None else TreeRegistry()
    snapshot = trees.get_snapshot_for_source_hash(class_name, source_hash) if source_hash else None
    if snapshot is not None:
        call_dep_names = trees.get_call_dependencies(snapshot.dependency_tree_hash)
        inh_dep_names = trees.get_inheritance_dependencies(snapshot.dependency_tree_hash)
        tree_path = trees.get_tree_path(snapshot.dependency_tree_hash)
        graph_path_candidate = tree_path.parent / 'graph.pdf'
        graph_path_str = str(graph_path_candidate.resolve()) if graph_path_candidate.exists() else None
        tree_path_str = str(tree_path.resolve())

    state = registry.get_state_by_hash(source_hash) if source_hash else None

    return RegistryClassInfo(
        object_type=state.object_type if state else None,
        call_dependency_names=call_dep_names,
        inheritance_dependency_names=inh_dep_names,
        stored_source_hash=source_hash,
        stored_dependency_tree_hash=None,
        source_path=str(code_resolver.source_path.resolve())
        if code_resolver and code_resolver.source_path.exists()
        else None,
        graph_path=graph_path_str,
        registry_path=str(meta_path.resolve()) if meta_path and meta_path.exists() else None,
        tree_path=tree_path_str,
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
            class_source_path=existing_path_str(code_resolver.source_path),
            class_graph_path=existing_path_str(tree_resolver.graph_path),
            class_registry_path=existing_path_str(code_resolver.meta_path),
            class_tree_path=existing_path_str(tree_resolver.tree_path),
            source_stale=source_stale,
            deps_stale=deps_stale,
        )

    return classes


def merge_unloaded_classes(
    classes: dict[str, ClassInfo],
    entry_registry: EntryRegistry,
    entries: dict[str, EntryInfo] | None = None,
    version_registry: VersionRegistry | None = None,
    src: SourceRegistry | None = None,
    trees: TreeRegistry | None = None,
) -> dict[str, ClassInfo]:
    merged = dict(classes)

    for class_name in entry_registry.class_names:
        if class_name in merged:
            continue

        info = source_info_from_disk(class_name, src=src, trees=trees)
        object_type = info.object_type or entry_registry.get_object_type(class_name)

        deps_stale = False
        if entries is not None and version_registry is not None:
            # Use the most recently versioned entry's dep_hash to represent the class.
            # If the newest entry is current, the class dot stays clear even if older
            # entries are stale — the user has already rerun at the current version.
            best_dep_hash: str | None = None
            best_version = None
            for rid in entry_registry.get_state_hashes(class_name):
                entry = entries.get(rid)
                if entry is None or not entry.dep_hash:
                    continue
                v = version_registry.version_for_dep_hash(entry.dep_hash)
                if v is not None and (best_version is None or v > best_version):
                    best_version = v
                    best_dep_hash = entry.dep_hash
            if best_dep_hash is not None:
                deps_stale = version_registry.is_dependency_hash_stale(best_dep_hash)

        merged[class_name] = ClassInfo(
            class_name=class_name,
            object_type=object_type,
            loaded=False,
            call_dependency_names=info.call_dependency_names,
            inheritance_dependency_names=info.inheritance_dependency_names,
            class_source_path=info.source_path,
            class_graph_path=info.graph_path,
            class_registry_path=info.registry_path,
            class_tree_path=info.tree_path,
            deps_stale=deps_stale,
        )

    return merged
