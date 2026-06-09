import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pygeodata.config import JSONKeys, get_config
from pygeodata.registry_browser.class_catalog import (
    discover_loaded_classes,
    merge_unloaded_classes,
    scan_code_snapshots,
)
from pygeodata.registry_browser.entry_catalog import discover_entries
from pygeodata.registry_browser.models import ClassInfo, EntryInfo, GroupInfo, VersionInfo
from pygeodata.types import SpecKeys


@dataclass(slots=True)
class AppState:
    classes: dict[str, ClassInfo]
    entries: dict[str, EntryInfo]
    groups: dict[str, GroupInfo]
    diagnostics: dict
    spec_options: dict[str, list[str]]
    versions: list[VersionInfo]
    snapshots: dict[str, str]
    code_groups: dict[str, list[dict]]


class AppContext:
    def __init__(self) -> None:
        self.state: AppState | None = None
        self.ready = threading.Event()
        self.progress: dict = {}

    def start_load(self) -> None:
        def _load() -> None:
            self.state = build_state(progress=self.progress)
            self.ready.set()

        threading.Thread(target=_load, daemon=True).start()

    def start_reload(self) -> None:
        self.ready.clear()
        self.progress.clear()
        self.start_load()

    def is_loading(self) -> bool:
        return not self.ready.is_set()


def merge_version_groups(versions: list[VersionInfo]) -> list[list[VersionInfo]]:
    """Greedily merge consecutive VersionInfo events that touch different classes.

    Sorted oldest-first input. Two adjacent events merge iff they don't share a
    class_name. Returns groups oldest-first.
    """
    sorted_versions = sorted(versions, key=lambda v: v.mtime)
    groups: list[list[VersionInfo]] = []
    for v in sorted_versions:
        if groups and v.class_name not in {g.class_name for g in groups[-1]}:
            groups[-1].append(v)
        else:
            groups.append([v])
    return groups


def version_group_label(group: list[VersionInfo], *, is_oldest: bool = False, display_mtime: str | None = None) -> str:
    prefix = ', '.join(sorted({v.class_name for v in group}))
    mtime_str = display_mtime if display_mtime is not None else max(v.mtime for v in group)
    try:
        dt = datetime.fromisoformat(mtime_str)
        return f'{prefix} · {dt.strftime("%b %-d, %H:%M")}'
    except (ValueError, AttributeError):
        return prefix


def build_version_groups(versions: list[VersionInfo], code_groups: dict[str, list[dict]]) -> list[dict]:
    """Canonical version-group list shared by the code-tab API and snapshot filter.

    Returns groups newest-first.  Each entry has mtime, all_mtimes, class_names,
    label, exclusive, cutoff_mtime, cutoff_exclusive.  The final 'Initial' entry
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


def _build_versions(groups: dict[str, list[dict]]) -> list[VersionInfo]:
    """All code-change events from scan_code_snapshots, sorted newest first."""
    versions = []
    for class_name, entries in groups.items():
        for e in entries:
            if e['is_version_change']:
                versions.append(VersionInfo(mtime=e['mtime'], class_name=class_name))
    versions.sort(key=lambda v: v.mtime, reverse=True)
    return versions


def _build_snapshots(
    entries: dict,
    versions: list[VersionInfo],
    code_groups: dict[str, list[dict]],
) -> dict[str, str]:
    """Map each distinct dep_hash to the version-group identity mtime it belongs to.

    The identity is 'initial' for snapshots created before the first code change, or
    the rep_mtime of the oldest version group whose window contains the snapshot.
    Groups are assigned newest-first; the snapshot gets the oldest group that still
    covers it (i.e. snapshot_max_mtime < group's cutoff).
    """
    dep_hashes = {e.dep_hash for e in entries.values() if e.dep_hash}
    if not dep_hashes:
        return {}

    # Build source_hash → registered_at from the already-scanned code groups
    hash_to_mtime: dict[str, str] = {
        e['source_hash']: e['mtime']
        for class_entries in code_groups.values()
        for e in class_entries
        if e['source_hash']
    }

    version_opts = build_version_groups(versions, code_groups)
    if not version_opts:
        return {}

    snapshots_dir = Path(get_config().path_registry) / 'snapshots'
    result: dict[str, str] = {}

    for dep_hash in dep_hashes:
        tree_path = snapshots_dir / dep_hash / 'tree.json'
        try:
            tree_data = json.loads(tree_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue

        nodes = tree_data.get(JSONKeys.NODES, tree_data.get('nodes', {}))
        node_mtimes = [
            hash_to_mtime[n['hash']] for n in nodes.values() if isinstance(n, dict) and n.get('hash') in hash_to_mtime
        ]
        if not node_mtimes:
            continue

        snapshot_max_mtime = max(node_mtimes)

        # Assign to the oldest group whose window covers the snapshot.
        # Groups are newest-first; the 'initial' group (last) has cutoff_exclusive=True.
        identity = version_opts[-1]['mtime']  # default to Initial
        for opt in version_opts:
            cutoff = opt['cutoff_mtime']
            excl = opt['cutoff_exclusive']
            if cutoff == 'now' or (excl and snapshot_max_mtime < cutoff) or (not excl and snapshot_max_mtime <= cutoff):
                identity = opt['mtime']

        result[dep_hash] = identity

    return result


def build_state(progress: dict | None = None) -> AppState:
    entries, groups, diagnostics = discover_entries(progress=progress)
    classes = merge_unloaded_classes(discover_loaded_classes(), groups)
    spec_options = {
        SpecKeys.CRS: sorted({entry.spec.crs for entry in entries.values() if entry.spec.crs}),
        SpecKeys.RESOLUTION: sorted({entry.spec.resolution for entry in entries.values() if entry.spec.resolution}),
        SpecKeys.SHAPE: sorted({entry.spec.shape for entry in entries.values() if entry.spec.shape}),
        SpecKeys.BOUNDS: sorted(
            {str(list(entry.spec.bounds_latlon)) for entry in entries.values() if entry.spec.bounds_latlon},
        ),
    }
    code_groups = scan_code_snapshots()
    versions = _build_versions(code_groups)
    snapshots = _build_snapshots(entries, versions, code_groups)
    return AppState(
        classes=classes,
        entries=entries,
        groups=groups,
        diagnostics=diagnostics,
        spec_options=spec_options,
        versions=versions,
        snapshots=snapshots,
        code_groups=code_groups,
    )
