import json
import logging
from pathlib import Path
from typing import Any

from pygeodata.config import JSONKeys, get_config
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryResolver, TreeRegistryResolver
from pygeodata.registry_browser.io_utils import existing_path_str, read_json_dict
from pygeodata.registry_browser.models import ClassInfo, RegistryClassInfo
from pygeodata.tracked_object import TrackedObject

logger = logging.getLogger(__name__)


def _extract_dep_names(tree_data: dict, dep_type: str) -> list[str]:
    """Extract direct child names under ``dep_type`` from the tree root.

    Parameters
    ----------
    tree_data:
        The full ``{nodes, tree}`` dict as stored in ``tree.json``.
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
    """Read class metadata from the content-addressed code registry.

    Scans ``.source/code/*/source.json`` for a snapshot whose ``class_name`` field
    matches, then finds the most recent one by ``mtime``. Also searches
    ``.source/snapshots/*/tree.json`` for a dependency tree that has this class as root.

    Used for classes that are known from cache but not loaded into the Python registry.
    Returns a default :class:`RegistryClassInfo` if nothing is found.
    """
    code_root = Path(get_config().path_registry) / 'code'
    if not code_root.exists():
        return RegistryClassInfo()

    candidates = []
    for meta_path in code_root.glob('*/source.json'):
        try:
            data = json.loads(meta_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get(JSONKeys.CLASS_NAME) == class_name:
            candidates.append((data.get('mtime', ''), meta_path, data))

    if not candidates:
        return RegistryClassInfo()

    _, meta_path, data = max(candidates, key=lambda x: x[0])
    source_hash = data.get(JSONKeys.SOURCE_HASH)
    code_resolver = CodeRegistryResolver.from_source_hash(source_hash) if source_hash else None

    call_dep_names: list[str] = []
    inh_dep_names: list[str] = []
    graph_path_str: str | None = None

    snapshots_dir = Path(get_config().path_registry) / 'snapshots'
    if snapshots_dir.exists():
        for tree_path in snapshots_dir.glob('*/tree.json'):
            try:
                tree_data = json.loads(tree_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            tree = tree_data.get(JSONKeys.TREE, {})
            if class_name in tree:
                call_dep_names = _extract_dep_names(tree_data, 'call_dependencies')
                inh_dep_names = _extract_dep_names(tree_data, 'inheritance_dependencies')
                graph_path_candidate = tree_path.parent / 'graph.pdf'
                graph_path_str = str(graph_path_candidate.resolve()) if graph_path_candidate.exists() else None
                break

    return RegistryClassInfo(
        object_type=str(data[JSONKeys.OBJECT_TYPE]) if JSONKeys.OBJECT_TYPE in data else None,
        call_dependency_names=call_dep_names,
        inheritance_dependency_names=inh_dep_names,
        stored_source_hash=source_hash,
        stored_dependency_tree_hash=None,
        source_path=str(code_resolver.source_path.resolve()) if code_resolver and code_resolver.source_path.exists() else None,
        graph_path=graph_path_str,
        registry_path=str(meta_path.resolve()),
    )


def discover_loaded_classes() -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}

    for class_name, cls in sorted(TrackedObject._registry.items()):
        call_dependency_names = sorted(dep.get_class_name() for dep in cls.get_call_dependencies())
        inheritance_dependency_names = sorted(dep.get_class_name() for dep in cls.get_inheritance_dependencies())

        live_source_hash = calculate_cls_source_hash(cls)
        code_resolver = CodeRegistryResolver.from_source_hash(live_source_hash)
        tree_resolver = TreeRegistryResolver.from_dep_tree_hash(cls.get_dependency_tree_hash())

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
