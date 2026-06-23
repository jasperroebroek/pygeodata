"""Version-group logic and VersionRegistry for the .source/ code registry.

VersionRegistry is the single authoritative source for:

    .versions                           — list[Version], newest-first, last entry is Initial
    .version_for_source_hash(hash)      -> Version | None
    .version_for_dep_hash(hash)         -> Version | None
    .version_number(version)            -> int   (Initial=0, oldest change=1, ...)
    .label(version)                     -> str
    .class_snapshot_at_version(version) -> dict[class_name, source_hash]
    .version_change_summary(version)    -> list[CodeEvent]

Construct a fresh instance to scan from disk.  Call .reload() after
write_registry to refresh without invalidating held references.
"""

from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import total_ordering
from pathlib import Path

from pygeodata.registry import SourceRegistry, TreeRegistry
from pygeodata.registry_types import ChangeStatus, CodeEvent, CodeState


@total_ordering
@dataclass
class Version:
    """A version-change group: one or more CodeEvents that occurred together.

    The last entry in VersionRegistry.versions is the Initial group —
    it holds the states that existed before any version change, with mtime set
    to the registered_at of the earliest CodeState across all classes.

    version_id is a stable UUID assigned at build time and is the canonical
    identifier for this group in all API calls and lookups.  Sorting compares
    by mtime (ISO-8601 UTC strings sort correctly as plain strings).

    events contains ALL statuses (ADDED, CHANGED, REMOVED, UNCHANGED) — the
    full change summary versus the predecessor version, computed at build time.
    """

    events: list[CodeEvent]
    mtime: str
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.version_id == other.version_id

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self.mtime < other.mtime

    def __hash__(self) -> int:
        return hash(self.version_id)

    @property
    def class_names(self) -> list[str]:
        """Classes that are ADDED or CHANGED in this version (not REMOVED/UNCHANGED)."""
        return sorted({e.class_name for e in self.events if e.status in (ChangeStatus.ADDED, ChangeStatus.CHANGED)})

    @property
    def source_hashes(self) -> list[str]:
        return [e.state_new.source_hash for e in self.events if e.state_new is not None]


class VersionRegistry:
    """In-memory index mapping source/dep hashes to their Version group.

    Constructed from a SourceRegistry + TreeRegistry scan.  Call reload()
    after write_registry to refresh without invalidating held references.
    """

    def __init__(self, registry_root: Path | None = None) -> None:
        self._registry_root = registry_root
        self._src: SourceRegistry = SourceRegistry(registry_root)
        self._trees: TreeRegistry = TreeRegistry(registry_root)

        # Populated by _build(); declared here so type checkers see them.
        self.versions: list[Version] = []
        self._id_to_version: dict[str, Version] = {}
        self._version_numbers: dict[str, int] = {}
        self._source_hash_to_version: dict[str, Version] = {}
        self._dep_hash_to_version: dict[str, Version] = {}
        self._version_to_dep_hashes: dict[str, set[str]] = {}

        self._build()

    @staticmethod
    def _collect_raw_events(src: SourceRegistry) -> list[CodeEvent]:
        """Return all ADDED/CHANGED events from the source registry, oldest-first."""
        events: list[CodeEvent] = []
        for class_name in src.class_names:
            states = src.get_states(class_name)
            if states:
                events.append(CodeEvent(state_new=states[0], state_old=None))
            for prev, curr in itertools.pairwise(states):
                events.append(CodeEvent(state_new=curr, state_old=prev))
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
        events: list[CodeEvent],
        hash_to_snapshots: dict[str, set[str]],
    ) -> list[list[CodeEvent]]:
        """Group raw ADDED/CHANGED events into version batches using snapshot co-occurrence.

        Greedy merge: consecutive events touching different classes join the same
        group — unless a snapshot exists that contains a current-group hash alongside
        the incoming event's prev hash.  Such a snapshot proves a real computation
        ran between the two changes, so they belong in separate groups.
        """
        raw: list[list[CodeEvent]] = []
        for event in events:
            prev_hash = event.state_old.source_hash if event.state_old else None
            if raw and event.class_name not in {e.class_name for e in raw[-1]}:
                group_snap_sets = [
                    hash_to_snapshots.get(e.state_new.source_hash, set()) for e in raw[-1] if e.state_new
                ]
                prev_snaps = hash_to_snapshots.get(prev_hash, set()) if prev_hash else set()
                intermediate_exists = any(g & prev_snaps for g in group_snap_sets)
                if not intermediate_exists:
                    raw[-1].append(event)
                    continue
            raw.append([event])
        return raw

    @staticmethod
    def _inject_untracked_classes(
        src: SourceRegistry,
        all_groups: list[list[CodeEvent]],
    ) -> None:
        """Inject a first-appearance ADDED event for each class into the right group.

        all_groups is oldest-first: all_groups[0] is Initial (open lower bound),
        all_groups[1..] are version-change groups ordered oldest→newest.

        For each class, its first state is injected into exactly one group:
          - Initial  if the first registration predates all version changes
                     (or if there are no version groups at all).
          - Otherwise the newest version group whose lower-bound mtime ≤ first_reg.

        Classes that already have a change event in a non-Initial group are skipped
        for that group (their first-state already flows into Initial via the rule
        above if it predates oldest_change_mtime).  Mutates all_groups in place.
        """
        # Classes with change events in version groups (all_groups[1..])
        in_version_groups = {e.class_name for g in all_groups[1:] for e in g}
        group_lower = [min(e.mtime for e in g) if g else '' for g in all_groups]

        for class_name in src.class_names:
            states = src.get_states(class_name)
            if not states:
                continue
            first_reg = states[0].registered_at
            oldest_change_mtime = group_lower[1] if len(group_lower) > 1 else ''
            if oldest_change_mtime and first_reg >= oldest_change_mtime:
                # Debut is at or after the first version change — inject into the
                # newest version group whose lower bound ≤ first_reg, unless already there.
                if class_name in in_version_groups:
                    continue
                target_idx = max(
                    (i for i in range(1, len(all_groups)) if not group_lower[i] or group_lower[i] <= first_reg),
                    default=0,
                )
                all_groups[target_idx].append(CodeEvent(state_new=states[0], state_old=None))
            else:
                # Debut predates any version change → Initial
                all_groups[0].append(CodeEvent(state_new=states[0], state_old=None))

    @staticmethod
    def _raw_source_hash_to_version(
        all_groups: list[list[CodeEvent]],
        all_versions: list[Version],
    ) -> dict[str, Version]:
        """Build source_hash → Version from raw group events before full events are assigned.

        all_groups and all_versions are parallel, oldest-first.
        """
        index: dict[str, Version] = {}
        for group, vi in zip(all_groups, all_versions, strict=False):
            for e in group:
                if e.state_new:
                    index[e.state_new.source_hash] = vi
        return index

    @staticmethod
    def _assign_full_events(
        src: SourceRegistry,
        versions: list[Version],
        initial: Version | None,
        raw_index: dict[str, Version],
    ) -> None:
        """Compute and assign full ADDED/CHANGED/REMOVED/UNCHANGED events to each Version."""
        ordered = ([initial] if initial else []) + list(reversed(versions))  # oldest-first

        def snapshot_at(vi: Version) -> dict[str, CodeState]:
            snap: dict[str, CodeState] = {}
            for class_name in src.class_names:
                for state in reversed(src.get_states(class_name)):
                    sv = raw_index.get(state.source_hash)
                    if sv is None or sv <= vi:
                        snap[class_name] = state
                        break
            return snap

        for idx, vi in enumerate(ordered):
            snap_new = snapshot_at(vi)
            snap_old = snapshot_at(ordered[idx - 1]) if idx > 0 else {}
            vi.events = [
                CodeEvent(state_new=snap_new.get(cn), state_old=snap_old.get(cn))
                for cn in sorted(set(snap_new) | set(snap_old))
            ]

    @staticmethod
    def _build_version_groups(
        src: SourceRegistry,
        raw_groups: list[list[CodeEvent]],
    ) -> tuple[list[Version], Version | None]:
        """Build Version objects (newest-first) and the Initial group.

        Returns (non_initial_versions_newest_first, initial_version_or_None).
        """
        # Initial is an open-lower-bound bucket; version groups are oldest-first.
        # _inject_untracked_classes assigns each untracked class to the right bucket.
        initial_raw: list[CodeEvent] = []
        all_groups_asc = [initial_raw, *raw_groups]  # oldest-first
        VersionRegistry._inject_untracked_classes(src, all_groups_asc)

        initial = (
            Version(events=initial_raw, mtime=min(e.mtime for e in initial_raw))
            if initial_raw else None
        )
        versions: list[Version] = [
            Version(events=[], mtime=max(e.mtime for e in group)) for group in reversed(raw_groups)
        ]

        all_versions_asc = ([initial] if initial else [None]) + list(reversed(versions))
        paired = [(g, v) for g, v in zip(all_groups_asc, all_versions_asc, strict=True) if v is not None]
        raw_index = VersionRegistry._raw_source_hash_to_version(
            [g for g, _ in paired],
            [v for _, v in paired],
        )
        VersionRegistry._assign_full_events(src, versions, initial, raw_index)
        return versions, initial

    @staticmethod
    def _build_dep_hash_index(
        trees: TreeRegistry,
        src: SourceRegistry,
        source_hash_to_version: dict[str, Version],
        versions: list[Version],
    ) -> dict[str, Version]:
        """Return dep_hash → Version for every snapshot in trees."""
        initial_version = versions[-1] if versions else None
        non_initial_asc = list(reversed(versions[:-1]))  # oldest-first, excludes Initial
        result: dict[str, Version] = {}
        for dep_hash in trees.dependency_hashes:
            snapshot = trees.get_snapshot_from_hash(dep_hash)
            if snapshot is None:
                continue
            nodes = [n for n in snapshot.nodes.values() if isinstance(n, dict) and n.get('hash')]
            node_versions = [nv for n in nodes if (nv := source_hash_to_version.get(n['hash'])) is not None]
            if node_versions:
                result[dep_hash] = max(node_versions)
            elif initial_version is not None:
                node_reg_times = [t for n in nodes if (t := src.get_mtime_from_hash(n['hash'])) is not None]
                max_reg = max(node_reg_times) if node_reg_times else None
                assigned = initial_version
                if max_reg is not None:
                    for v in non_initial_asc:
                        if v.mtime <= max_reg:
                            assigned = v
                            break
                result[dep_hash] = assigned
        return result

    def _build(self) -> None:
        src = self._src
        trees = self._trees

        raw_events = self._collect_raw_events(src)
        # Split into ADDED (first appearance) and CHANGED (subsequent) for merge logic
        changed_events = [e for e in raw_events if e.status == ChangeStatus.CHANGED]
        hash_to_snapshots = self._build_hash_to_snapshots(trees)
        raw_groups = self._merge_events(changed_events, hash_to_snapshots)

        versions, initial = self._build_version_groups(src, raw_groups)
        if initial is not None:
            versions.append(initial)

        # versions is newest-first; last entry is Initial (version_number=0).
        non_initial = versions[:-1] if versions else []
        version_numbers: dict[str, int] = {v.version_id: 0 for v in versions}
        for i, v in enumerate(reversed(non_initial), start=1):
            version_numbers[v.version_id] = i

        source_hash_to_version: dict[str, Version] = {}
        for v in versions:
            for e in v.events:
                if e.state_new and e.status in (ChangeStatus.ADDED, ChangeStatus.CHANGED):
                    source_hash_to_version[e.state_new.source_hash] = v

        dep_hash_to_version = self._build_dep_hash_index(trees, src, source_hash_to_version, versions)

        # Version → set[dep_hash] reverse index
        version_to_dep_hashes: dict[str, set[str]] = {v.version_id: set() for v in versions}
        for dh, v in dep_hash_to_version.items():
            version_to_dep_hashes[v.version_id].add(dh)

        self.versions = versions
        self._id_to_version = {v.version_id: v for v in versions}
        self._version_numbers = version_numbers
        self._source_hash_to_version = source_hash_to_version
        self._dep_hash_to_version = dep_hash_to_version
        self._version_to_dep_hashes = version_to_dep_hashes

    def reload(self) -> None:
        """Reload sub-registries then recompute all indexes in-place."""
        self._src.reload()
        self._trees.reload()
        self._build()

    def version_for_source_hash(self, source_hash: str) -> Version | None:
        """Return the Version group for a source hash, or None if unknown."""
        return self._source_hash_to_version.get(source_hash)

    def version_for_dep_hash(self, dep_hash: str) -> Version | None:
        """Return the Version group for a dep hash, or None if unknown."""
        return self._dep_hash_to_version.get(dep_hash)

    def version_by_id(self, version_id: str) -> Version | None:
        """Return the Version with the given UUID, or None if not found."""
        return self._id_to_version.get(version_id)

    def version_number(self, version: Version) -> int:
        """Return the version number for a Version (Initial=0, oldest change=1, ...)."""
        return self._version_numbers.get(version.version_id, 0)

    def label(self, version: Version) -> str:
        """Return the display label for a Version, e.g. 'v3 · Jun 12, 19:39'."""
        n = self._version_numbers.get(version.version_id, 0)
        v = f'v{n}' if n else 'Initial'
        try:
            dt = datetime.fromisoformat(version.mtime)
            return f'{v} · {dt.strftime("%b %-d, %H:%M")}'
        except (ValueError, AttributeError):
            return v

    def class_snapshot_at_version(self, version: Version) -> dict[str, str]:
        """Return {class_name: source_hash} for every class as of the given version.

        Uses _source_hash_to_version (event-chain index) rather than timestamp
        windowing, so it is immune to intra-group timestamp spread.
        Returns an empty dict if the version is not in this registry.
        """
        if version not in self._id_to_version.values():
            return {}
        src = self._src
        snapshot: dict[str, str] = {}
        for class_name in src.class_names:
            for state in reversed(src.get_states(class_name)):
                sv = self._source_hash_to_version.get(state.source_hash)
                if sv is None or sv <= version:
                    snapshot[class_name] = state.source_hash
                    break
        return snapshot

    def version_change_summary(self, version: Version) -> list[CodeEvent] | None:
        """Return the full per-class change summary for the given version.

        This is the pre-computed Version.events list — a direct lookup.
        Returns None if version is not in this registry.
        """
        if version not in self._id_to_version.values():
            return None
        return version.events

    def version_change_summary_from_id(self, version_id: str) -> list[CodeEvent] | None:
        """Look up by version_id and return version_change_summary, or None if not found."""
        version = self.version_by_id(version_id)
        if version is None:
            return None
        return self.version_change_summary(version)

    @property
    def source_registry(self) -> SourceRegistry:
        return self._src

    @property
    def tree_registry(self) -> TreeRegistry:
        return self._trees

    def dep_hashes_for_version(self, version: Version) -> set[str]:
        """Return the set of dep_hashes whose snapshot belongs to the given version."""
        return set(self._version_to_dep_hashes.get(version.version_id, set()))

    @property
    def dep_hash_to_version(self) -> dict[str, Version]:
        """Full dep_hash → Version mapping (read-only view)."""
        return self._dep_hash_to_version
