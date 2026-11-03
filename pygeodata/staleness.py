from __future__ import annotations

from typing import TYPE_CHECKING

from pygeodata.enums import Staleness
from pygeodata.tracked_object import TrackedObject

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pygeodata.registries.versioning import VersionRegistry


def dependency_staleness(
    class_name: str | None,
    dependency_tree_hash: str | None,
    versions: VersionRegistry | None = None,
) -> Staleness:
    """Compare against the live class when importable, otherwise against ``versions`` when given."""
    if not dependency_tree_hash:
        return Staleness.UNKNOWN
    live = TrackedObject.find_object_class(class_name) if class_name else None
    if live is not None:
        return Staleness.CURRENT if live.get_dependency_tree_hash() == dependency_tree_hash else Staleness.STALE
    if versions is None:
        return Staleness.UNKNOWN
    return versions.dependency_staleness(dependency_tree_hash)


def class_staleness(
    class_name: str,
    dependency_tree_hashes: Iterable[str | None],
    versions: VersionRegistry,
) -> Staleness:
    """Staleness of the class's newest versioned dependency hash; older entries don't count."""
    versioned = [(version, h) for h in dependency_tree_hashes if h and (version := versions.version_for_dep_hash(h))]
    if not versioned:
        return Staleness.UNKNOWN
    _, newest = max(versioned, key=lambda pair: pair[0])
    return dependency_staleness(class_name, newest, versions)
