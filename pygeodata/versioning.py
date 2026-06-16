"""Version-group logic and VersionRegistry for the .source/ code registry.

VersionRegistry is the single authoritative source for:

    .version_groups          — list[VersionInfo], newest-first, last entry is Initial
    .version_mtime_for_source_hash(source_hash) -> str | None
    .version_mtime_for_dep_hash(dep_hash) -> str | None

Construct a fresh instance to scan from disk.  Call .reload() after
write_registry to refresh without invalidating held references.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pygeodata.registry import SourceRegistry, TreeRegistry


@dataclass(frozen=True)
class CodeChangeEvent:
    """A single class-level code change recorded in the source registry."""

    mtime: str
    class_name: str
    source_hash: str
    prev_source_hash: str


@dataclass
class VersionInfo:
    """A version-change group: one or more CodeChangeEvents that occurred together.

    The last entry in VersionRegistry.version_groups is the Initial group —
    it holds the states that existed before any version change, with mtime set
    to the registered_at of the earliest CodeState across all classes.
    """

    events: list[CodeChangeEvent]
    mtime: str

    @property
    def class_names(self) -> list[str]:
        return sorted({e.class_name for e in self.events})

    @property
    def source_hashes(self) -> list[str]:
        return [e.source_hash for e in self.events]

    version_number: int = 0  # set by VersionRegistry after construction

    @property
    def label(self) -> str:
        v = f'v{self.version_number}' if self.version_number else 'Initial'
        try:
            dt = datetime.fromisoformat(self.mtime)
            return f'{v} · {dt.strftime("%b %-d, %H:%M")}'
        except (ValueError, AttributeError):
            return v


class VersionRegistry:
    """In-memory index mapping source/dep hashes to their version-group mtime.

    Constructed from a SourceRegistry + TreeRegistry scan.  Call reload()
    after write_registry to refresh without invalidating held references.
    """

    def __init__(self, registry_root: Path | None = None) -> None:
        self._registry_root = registry_root
        self._src = SourceRegistry(registry_root)
        self._trees = TreeRegistry(registry_root)
        self._build()

    @staticmethod
    def _collect_events(src: SourceRegistry) -> list[CodeChangeEvent]:
        """Return all version-change events, oldest-first."""
        events: list[CodeChangeEvent] = []
        for class_name in src.class_names:
            states = src.get_states(class_name)
            for prev, curr in itertools.pairwise(states):
                events.append(CodeChangeEvent(curr.registered_at, class_name, curr.source_hash, prev.source_hash))
        events.sort(key=lambda e: e.mtime)
        return events

    @staticmethod
    def _build_hash_to_snapshots(trees: TreeRegistry) -> dict[str, set[str]]:
        """Index source_hash → set of dep_hashes whose snapshot contains that hash."""
        index: dict[str, set[str]] = {}
        for dep_hash in trees.dependency_hashes:
            snapshot = trees.get_snapshot_from_hash(dep_hash)
            if snapshot is None:
                continue
            for node in snapshot.nodes.values():
                if isinstance(node, dict) and (h := node.get('hash')):
                    index.setdefault(h, set()).add(dep_hash)
        return index

    @staticmethod
    def _merge_events(
        events: list[CodeChangeEvent],
        hash_to_snapshots: dict[str, set[str]],
    ) -> list[list[CodeChangeEvent]]:
        """Group change events into version batches using snapshot co-occurrence.

        Greedy merge: consecutive events touching different classes join the same
        group — unless a snapshot exists that contains a current-group hash alongside
        the incoming event's prev_source_hash.  Such a snapshot proves a real
        computation ran between the two changes, so they belong in separate groups.
        """
        raw: list[list[CodeChangeEvent]] = []
        for event in events:
            if raw and event.class_name not in {e.class_name for e in raw[-1]}:
                group_snap_sets = [hash_to_snapshots.get(e.source_hash, set()) for e in raw[-1]]
                prev_snaps = hash_to_snapshots.get(event.prev_source_hash, set())
                intermediate_exists = any(g & prev_snaps for g in group_snap_sets)
                if not intermediate_exists:
                    raw[-1].append(event)
                    continue
            raw.append([event])
        return raw

    @staticmethod
    def _build_initial_group(src: SourceRegistry, oldest_change_mtime: str | None) -> VersionInfo | None:
        """Build the Initial VersionInfo from CodeStates that predate any version change."""
        earliest_mtime: str | None = None
        initial_events: list[CodeChangeEvent] = []
        for class_name in src.class_names:
            states = src.get_states(class_name)
            if not states:
                continue
            oldest = states[0]
            if earliest_mtime is None or oldest.registered_at < earliest_mtime:
                earliest_mtime = oldest.registered_at
            if oldest_change_mtime is None or oldest.registered_at < oldest_change_mtime:
                initial_events.append(CodeChangeEvent(oldest.registered_at, class_name, oldest.source_hash, ''))
        if earliest_mtime is None:
            return None
        return VersionInfo(events=initial_events, mtime=earliest_mtime)

    @staticmethod
    def _assign_dep_hash(
        node_version_times: list[str],
        initial_mtime: str,
    ) -> str:
        """Return the version-group mtime for a snapshot.

        Each node hash is looked up in _source_hash_to_mtime to find which version
        group introduced it.  The snapshot belongs to the newest such group — i.e.
        the max version time across all its nodes.  Falls back to initial_mtime when
        no node maps to a change event (snapshot uses only initial-state hashes).
        """
        if not node_version_times:
            return initial_mtime
        return max(node_version_times)

    def _build(self) -> None:
        src = self._src
        trees = self._trees

        events = self._collect_events(src)
        hash_to_snapshots = self._build_hash_to_snapshots(trees)
        raw_groups = self._merge_events(events, hash_to_snapshots)

        version_groups: list[VersionInfo] = [
            VersionInfo(events=group, mtime=max(e.mtime for e in group)) for group in reversed(raw_groups)
        ]

        oldest_change_mtime = min(e.mtime for e in raw_groups[0]) if raw_groups else None
        initial = self._build_initial_group(src, oldest_change_mtime)
        if initial is not None:
            version_groups.append(initial)

        # Assign version numbers: Initial=0 (last entry), v1=oldest change, counting up.
        # version_groups is newest-first; the last entry is Initial.
        non_initial = version_groups[:-1] if version_groups else []
        for i, vi in enumerate(reversed(non_initial), start=1):
            vi.version_number = i
        # Initial group stays at version_number=0 (default)

        self.version_groups: list[VersionInfo] = version_groups

        # source_hash → version-group mtime (O(1) lookup)
        self._source_hash_to_mtime: dict[str, str] = {
            e.source_hash: vi.mtime for vi in version_groups for e in vi.events
        }

        # dep_hash → version-group mtime (O(1) lookup)
        initial_mtime = version_groups[-1].mtime if version_groups else ''
        non_initial = version_groups[:-1]  # newest-first, excludes Initial
        self._dep_hash_to_mtime: dict[str, str] = {}
        for dep_hash in trees.dependency_hashes:
            snapshot = trees.get_snapshot_from_hash(dep_hash)
            if snapshot is None:
                continue
            nodes = [n for n in snapshot.nodes.values() if isinstance(n, dict) and n.get('hash')]
            node_version_times = [vm for n in nodes if (vm := self._source_hash_to_mtime.get(n['hash'])) is not None]
            if node_version_times:
                self._dep_hash_to_mtime[dep_hash] = self._assign_dep_hash(node_version_times, initial_mtime)
            else:
                # No node maps to a change event: all nodes are from classes that have
                # never changed.  Find the oldest version group whose mtime is <=
                # max(registered_at) across the snapshot's nodes — that group was already
                # in effect when the newest dependency in this snapshot was registered.
                node_reg_times = [t for n in nodes if (t := src.get_mtime_from_hash(n['hash'])) is not None]
                max_reg = max(node_reg_times) if node_reg_times else None
                assigned = initial_mtime
                if max_reg is not None:
                    for vi in reversed(non_initial):  # oldest-first
                        if vi.mtime <= max_reg:
                            assigned = vi.mtime
                            break
                self._dep_hash_to_mtime[dep_hash] = assigned

    def reload(self) -> None:
        """Reload sub-registries then recompute all indexes in-place."""
        self._src.reload()
        self._trees.reload()
        self._build()

    def version_mtime_for_source_hash(self, source_hash: str) -> str | None:
        """Return the version-group mtime for a source hash, or None if unknown."""
        return self._source_hash_to_mtime.get(source_hash)

    def version_mtime_for_dep_hash(self, dep_hash: str) -> str | None:
        """Return the version-group mtime for a dep hash, or None if unknown."""
        return self._dep_hash_to_mtime.get(dep_hash)

    def is_dep_hash_stale(self, dep_hash: str) -> bool:
        """Return True if any class in the dep tree has a newer source_hash than stored.

        A dep_hash recorded at v1 is NOT stale at v2 if v2 only changed unrelated classes.
        Comparing stored vs latest source_hash per node is the only correct check.
        Returns False if the dep tree snapshot is not found.
        """
        snapshot = self._trees.get_snapshot_from_hash(dep_hash)
        if snapshot is None:
            return False
        for class_name in snapshot.nodes:
            stored_hash = snapshot.get_source_hash(class_name)
            latest_state = self._src.get_latest_state_for_class(class_name)
            latest_hash = latest_state.source_hash if latest_state else None
            if stored_hash != latest_hash:
                return True
        return False

    @property
    def source_registry(self) -> SourceRegistry:
        return self._src

    @property
    def tree_registry(self) -> TreeRegistry:
        return self._trees

    @property
    def dep_hash_to_mtime(self) -> dict[str, str]:
        """Full dep_hash → version_mtime mapping (read-only view)."""
        return self._dep_hash_to_mtime
