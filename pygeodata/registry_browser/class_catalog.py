import logging
from typing import Any

from pygeodata.config import JSONKeys, get_config
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.registry_browser.io_utils import existing_path_str, read_json_dict
from pygeodata.registry_browser.models import ClassInfo, RegistryClassInfo
from pygeodata.tracked_object import TrackedObject

logger = logging.getLogger(__name__)


def _extract_dep_names(tree_data: dict, dep_type: str) -> list[str]:
    """Extract direct child names under ``dep_type`` from the tree root.

    Parameters
    ----------
    tree_data:
        The full ``{nodes, tree}`` dict as stored in ``source.json``.
    dep_type:
        Either ``"call_dependencies"`` or ``"inheritance_dependencies"``.

    Returns
    -------
    list[str]
        Sorted list of class names that are direct dependencies of the root node.
    """
    tree = tree_data.get(JSONKeys.TREE, {})
    if not tree:
        return []
    root_node = next(iter(tree.values()), {})
    return sorted(root_node.get(dep_type, {}).keys())


def source_info_from_disk(class_name: str) -> RegistryClassInfo:
    """Read class metadata from source.json in the registry.

    Used for classes that are known from cache but not loaded into the Python registry.
    Returns a default RegistryClassInfo if nothing is found.
    """
    source_root = get_config().path_registry
    candidate = next(source_root.rglob(f'{class_name}/source.json'), None)
    if candidate is None:
        return RegistryClassInfo()
    data = read_json_dict(candidate)
    code_path = candidate.parent / 'source.py'
    graph_path = candidate.parent / 'source.pdf'
    return RegistryClassInfo(
        object_type=str(data[JSONKeys.OBJECT_TYPE]) if JSONKeys.OBJECT_TYPE in data else None,
        call_dependency_names=_extract_dep_names(data, 'call_dependencies'),
        inheritance_dependency_names=_extract_dep_names(data, 'inheritance_dependencies'),
        stored_source_hash=data.get(JSONKeys.SOURCE_HASH),
        stored_dependency_tree_hash=data.get(JSONKeys.DEPENDENCY_TREE_HASH),
        source_path=str(code_path.resolve()) if code_path.exists() else None,
        graph_path=str(graph_path.resolve()) if graph_path.exists() else None,
        registry_path=str(candidate.resolve()),
    )


def discover_loaded_classes() -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}

    for class_name, cls in sorted(TrackedObject._registry.items()):
        resolver = None
        call_dependency_names: list[str] = []
        inheritance_dependency_names: list[str] = []

        resolver = cls.resolve_registry_paths()

        call_dependency_names = sorted(dep.get_class_name() for dep in cls.get_call_dependencies())
        inheritance_dependency_names = sorted(dep.get_class_name() for dep in cls.get_inheritance_dependencies())

        source_stale = False
        deps_stale = False

        live_source_hash = calculate_cls_source_hash(cls)
        registry_info = source_info_from_disk(class_name)
        if registry_info.stored_source_hash is not None and live_source_hash != registry_info.stored_source_hash:
            source_stale = True
        if registry_info.stored_dependency_tree_hash is not None:
            live_dep_hash = cls.get_dependency_tree_hash()
            if live_dep_hash != registry_info.stored_dependency_tree_hash:
                deps_stale = True

        classes[class_name] = ClassInfo(
            class_name=class_name,
            object_type=cls.object_type.get_class_name(),
            loaded=True,
            call_dependency_names=call_dependency_names,
            inheritance_dependency_names=inheritance_dependency_names,
            class_source_path=existing_path_str(resolver.code_path) if resolver else None,
            class_graph_path=existing_path_str(resolver.graph_path) if resolver else None,
            class_registry_path=existing_path_str(resolver.registry_path) if resolver else None,
            source_stale=source_stale,
            deps_stale=deps_stale,
        )

    return classes


def merge_unloaded_classes(
    classes: dict[str, ClassInfo],
    groups: dict[str, Any],
) -> dict[str, ClassInfo]:
    merged = dict(classes)

    for class_name, group in groups.items():
        if class_name in merged:
            continue

        info = source_info_from_disk(class_name)
        object_type = (
            info.object_type or (str(getattr(group, 'object_type')) if getattr(group, 'object_type', None) else None)
        )

        merged[class_name] = ClassInfo(
            class_name=class_name,
            object_type=object_type,
            loaded=False,
            call_dependency_names=info.call_dependency_names,
            inheritance_dependency_names=info.inheritance_dependency_names,
            class_source_path=info.source_path,
            class_graph_path=info.graph_path,
            class_registry_path=info.registry_path,
        )

    return merged
