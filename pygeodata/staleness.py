"""Staleness of stored dependency tree hashes against the current ones.

``current`` gives a class's current dependency tree hash, or None when it does not know the class:
:func:`~pygeodata.tracked_object.live_tree_hash` for the imported classes, or
``StaticAnalysis.tree_hashes.get`` for those read from source. An unknown class is never current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygeodata.enums import Staleness

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pygeodata.registries.versioning import VersionRegistry


def dependency_staleness(
    class_name: str,
    dependency_tree_hash: str,
    current: Callable[[str], str | None],
    versions: VersionRegistry | None = None,
) -> Staleness:
    """Compare with the class's current hash; for a class ``current`` does not know, ask ``versions`` when given."""
    if not dependency_tree_hash:
        return Staleness.UNKNOWN
    if (current_hash := current(class_name)) is not None:
        return Staleness.CURRENT if current_hash == dependency_tree_hash else Staleness.STALE
    if versions is None:
        return Staleness.UNKNOWN
    return versions.dependency_staleness(dependency_tree_hash)


def class_staleness(
    class_name: str,
    dependency_tree_hashes: Iterable[str],
    current: Callable[[str], str | None],
    versions: VersionRegistry,
) -> Staleness:
    """Staleness of the class's newest versioned dependency hash; older entries don't count."""
    versioned = [(version, h) for h in dependency_tree_hashes if h and (version := versions.version_for_dep_hash(h))]
    if not versioned:
        return Staleness.UNKNOWN
    _, newest = max(versioned, key=lambda pair: pair[0])
    return dependency_staleness(class_name, newest, current, versions)
