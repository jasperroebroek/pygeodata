"""Version-group logic for the .source/ code registry.

Owns all algorithms that map snapshots or source hashes to version-group
identities:

    merge_version_groups   — greedy merge of consecutive single-class events
    version_group_label    — human-readable label for a group
    build_version_groups   — canonical newest-first list consumed by the API
    snapshot_version_identity(registry, dep_hash) -> str
    source_hash_version_identity(registry, source_hash, class_name) -> str | None
"""

from __future__ import annotations

from datetime import datetime

from pygeodata.registry import SourceRegistry, TreeRegistry
from pygeodata.registry_types import VersionInfo


def merge_version_groups(versions: list) -> list[list]:
    """Greedily merge consecutive VersionInfo events touching different classes.

    Input sorted oldest-first.  Returns groups oldest-first.
    """
    sorted_versions = sorted(versions, key=lambda v: v.mtime)
    groups: list[list] = []
    for v in sorted_versions:
        if groups and v.class_name not in {g.class_name for g in groups[-1]}:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def version_group_label(
    group: list,
    *,
    is_oldest: bool = False,
    display_mtime: str | None = None,
) -> str:
    prefix = ', '.join(sorted({v.class_name for v in group}))
    mtime_str = display_mtime if display_mtime is not None else max(v.mtime for v in group)
    try:
        dt = datetime.fromisoformat(mtime_str)
        return f'{prefix} · {dt.strftime("%b %-d, %H:%M")}'
    except (ValueError, AttributeError):
        return prefix


def build_version_groups(versions: list, code_groups: dict[str, list[dict]]) -> list[dict]:
    """Canonical version-group list shared by code-tab API and snapshot filter.

    Returns groups newest-first.  Each entry has mtime, all_mtimes, class_names,
    label, exclusive, cutoff_mtime, cutoff_exclusive.  Final 'Initial' entry
    has mtime='initial'.
    """
    groups = merge_version_groups(versions)
    if not groups:
        return []

    all_class_names = sorted(code_groups.keys())
    oldest_change_mtime = min(v.mtime for v in groups[0])

    oldest_registration_mtime = (
        min(e['mtime'] for entries in code_groups.values() for e in entries if entries)
        if code_groups
        else oldest_change_mtime
    )

    result = []
    groups_newest_first = list(reversed(groups))
    for i, group in enumerate(groups_newest_first):
        rep_mtime = max(v.mtime for v in group)
        if i == 0:
            cutoff_mtime = 'now'
            cutoff_exclusive = False
            group_class_names = all_class_names
        else:
            cutoff_mtime = max(v.mtime for v in groups_newest_first[i - 1])
            cutoff_exclusive = True
            group_class_names = sorted({v.class_name for v in group})
        result.append(
            {
                'mtime': rep_mtime,
                'all_mtimes': [v.mtime for v in group],
                'class_names': group_class_names,
                'label': version_group_label(group),
                'exclusive': False,
                'cutoff_mtime': cutoff_mtime,
                'cutoff_exclusive': cutoff_exclusive,
            },
        )

    try:
        dt = datetime.fromisoformat(oldest_registration_mtime)
        initial_label = f'Initial · {dt.strftime("%b %-d, %H:%M")}'
    except (ValueError, AttributeError):
        initial_label = 'Initial'

    initial_class_names = sorted(
        cn for cn, entries in code_groups.items() if any(e['mtime'] < oldest_change_mtime for e in entries)
    )
    result.append(
        {
            'mtime': 'initial',
            'all_mtimes': [oldest_change_mtime],
            'class_names': initial_class_names,
            'label': initial_label,
            'exclusive': True,
            'cutoff_mtime': oldest_change_mtime,
            'cutoff_exclusive': True,
        },
    )
    return result


def version_infos(registry: SourceRegistry) -> list:
    """Return all code-change events from ``registry`` as VersionInfo, newest-first."""
    events = []
    for class_name in registry.class_names:
        for s in registry.get_states(class_name):
            if registry.is_version_change(s):
                events.append(VersionInfo(mtime=s.registered_at, class_name=class_name))
    events.sort(key=lambda v: v.mtime, reverse=True)
    return events


def _window_mtime(mtime: str, version_opts: list[dict]) -> str:
    """Walk version_opts newest-first and return the identity for a given mtime scalar."""
    identity = version_opts[-1]['mtime']  # default: Initial
    for opt in version_opts:
        cutoff = opt['cutoff_mtime']
        excl = opt['cutoff_exclusive']
        if cutoff == 'now' or (excl and mtime < cutoff) or (not excl and mtime <= cutoff):
            identity = opt['mtime']
    return identity


def snapshot_version_identity(registry, dep_hash: str, tree_registry=None) -> str:
    """Map dep_hash → the version-group identity mtime it belongs to.

    Returns 'initial' for snapshots predating the first code change, the
    rep_mtime of the newest group when the snapshot is current, or the
    rep_mtime of the oldest group whose window covers the snapshot's max
    node mtime.

    ``registry`` is a :class:`~pygeodata.registry.SourceRegistry`.
    ``tree_registry`` is an optional :class:`~pygeodata.registry.TreeRegistry`;
    if omitted one is constructed from the same root as ``registry``.
    """
    trees = tree_registry or TreeRegistry(registry._root)
    tree = trees.get_snapshot(dep_hash)
    if tree is None:
        return 'initial'

    node_mtimes = [
        mtime
        for n in tree.nodes.values()
        if isinstance(n, dict) and n.get('hash') and (mtime := registry.hash_to_mtime(n['hash'])) is not None
    ]
    if not node_mtimes:
        return 'initial'

    tree_max_mtime = max(node_mtimes)

    version_opts = build_version_groups(version_infos(registry), registry.code_groups_dict())
    if not version_opts:
        return 'initial'

    return _window_mtime(tree_max_mtime, version_opts)


def source_hash_version_identity(
    registry: SourceRegistry,
    source_hash: str,
    class_name: str,
) -> str | None:
    """Map a source_hash + class_name → the version-group identity mtime.

    Returns ``None`` if ``source_hash`` is not found for ``class_name``.
    """
    states = registry.get_states(class_name)
    match = next((s for s in snaps if s.source_hash == source_hash), None)
    if match is None:
        return None

    version_opts = build_version_groups(version_infos(registry), registry.code_groups_dict())
    if not version_opts:
        return 'now'

    return _window_mtime(match.registered_at, version_opts)
