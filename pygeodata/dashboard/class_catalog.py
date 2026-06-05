import logging
from pathlib import Path
from typing import Any

from pygeodata.config import JSONKeys, get_config
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.tracked_object import TrackedObject

from .io_utils import existing_path_str, read_json_dict
from .models import ClassInfo

logger = logging.getLogger(__name__)


def source_info_from_disk(class_name: str) -> tuple[str, list[str], str | None, str | None, str | None]:
    """Read object_type, dependency_names, source_path, graph_path, stored_source_hash from source.json.

    Used for classes that are known from cache but not loaded into the Python registry.
    Returns ('unknown', [], None, None, None) if nothing is found.
    """
    source_root = get_config().path_registry
    for candidate in source_root.rglob(f'{class_name}/source.json'):
        try:
            data = read_json_dict(candidate)
        except Exception:
            continue
        object_type = str(data.get(JSONKeys.OBJECT_TYPE, 'unknown')).lower()
        tree = data.get(JSONKeys.TREE, {})
        dep_names: list[str] = sorted(
            set(tree.get(JSONKeys.CALL_DEPENDENCIES, {}).keys()) |
            set(tree.get(JSONKeys.INHERITANCE_DEPENDENCIES, {}).keys()),
        )
        code_path = candidate.parent / 'code.py'
        graph_path = candidate.parent / 'graph.pdf'
        stored_hash = data.get(JSONKeys.SOURCE_HASH)
        return (
            object_type,
            dep_names,
            str(code_path.resolve()) if code_path.exists() else None,
            str(graph_path.resolve()) if graph_path.exists() else None,
            stored_hash,
        )
    return 'unknown', [], None, None, None


def _stored_hashes(class_name: str) -> tuple[str | None, str | None]:
    """Return (stored_source_hash, stored_dependency_tree_hash) from source.json."""
    source_root = get_config().path_registry
    for candidate in source_root.rglob(f'{class_name}/source.json'):
        try:
            data = read_json_dict(candidate)
            return data.get(JSONKeys.SOURCE_HASH), data.get(JSONKeys.DEPENDENCY_TREE_HASH)
        except Exception:
            return None, None
    return None, None


def discover_loaded_classes() -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}

    for class_name, cls in sorted(TrackedObject._registry.items()):
        resolver = None
        dependency_names: list[str] = []

        try:
            resolver = cls.resolve_registry_paths()
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.warning('Failed to resolve registry paths for %s: %s', class_name, exc)

        try:
            dependency_names = sorted(dep.get_class_name() for dep in cls.get_all_dependencies())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning('Failed to resolve dependencies for %s: %s', class_name, exc)

        source_stale = False
        deps_stale = False
        try:
            live_source_hash = calculate_cls_source_hash(cls)
            stored_source_hash, stored_dep_hash = _stored_hashes(class_name)
            if stored_source_hash is not None and live_source_hash != stored_source_hash:
                source_stale = True
            if stored_dep_hash is not None:
                live_dep_hash = cls.get_dependency_tree_hash()
                if live_dep_hash != stored_dep_hash:
                    deps_stale = True
        except Exception as exc:
            logger.warning('Failed to compute source hash for %s: %s', class_name, exc)

        classes[class_name] = ClassInfo(
            class_name=class_name,
            object_type=cls.object_type.get_class_name(),
            loaded=True,
            dependency_names=dependency_names,
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

        object_type, dep_names, source_path, graph_path, _ = source_info_from_disk(class_name)
        if object_type == 'unknown':
            object_type = str(getattr(group, 'object_type', 'unknown'))

        merged[class_name] = ClassInfo(
            class_name=class_name,
            object_type=object_type,
            loaded=False,
            dependency_names=dep_names,
            class_source_path=source_path,
            class_graph_path=graph_path,
        )

    return merged