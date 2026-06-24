"""Version-group logic and VersionRegistry for the .source/ code registry.

VersionRegistry is the single authoritative source for:

    .versions                                  — list[Version], newest-first, last entry is Initial
    .version_for_source_hash(hash)             -> Version | None
    .version_for_dep_hash(hash)                -> Version | None
    .dep_hashes_for_version(version)           -> set[str]
    .dep_hash_to_version                       -> dict[str, Version]  (read-only)
    .version_number(version)                   -> int   (Initial=1, oldest change=2, ...)
    .version_by_id(version_id)                 -> Version | None
    .label(version)                            -> str
    .class_snapshot_at_version(version)        -> dict[class_name, source_hash]
    .is_dependency_hash_stale(dependency_hash) -> bool  (True if any class in snapshot differs from current)
    .version_change_summary(version)           -> list[CodeEvent] | None
    .version_change_summary_from_id(vid)       -> list[CodeEvent] | None
    .live_snapshot()                           -> dict[class_name, source_hash]  (live in-memory state)
    .has_live_stale()                          -> bool  (True when any loaded class diverges from disk)
    .snapshot_for_version(version_id)          -> dict[class_name, source_hash] | None
    .compare_versions(base, target)            -> list[CodeEvent]
    .source_registry                           -> SourceRegistry
    .tree_registry                             -> TreeRegistry

Version groups are built from the .source/ tree by grouping CHANGED events via
snapshot co-occurrence (a snapshot containing both old and new hashes means a
computation ran between them, separating the events into different groups).
Each Version.events carries the full ADDED/CHANGED/REMOVED/UNCHANGED summary
versus its predecessor, pre-computed at build time.

Construct a fresh instance to scan from disk.  Call .reload() after
write_registry to refresh without invalidating held references.
"""

from __future__ import annotations

import itertools
from datetime import datetime
from pathlib import Path

from pygeodata.registries.registry import SourceRegistry, TreeRegistry
from pygeodata.registries.registry_types import ChangeStatus, CodeEvent, CodeState, Version
from pygeodata.tracked_object import TrackedObject


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
                    (i for i in range(1, len(all_groups)) if group_lower[i] <= first_reg),
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
        for group, vi in zip(all_groups, all_versions, strict=True):
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

        running: dict[str, CodeState] = {}
        snapshots: list[dict[str, CodeState]] = []
        for vi in ordered:
            for class_name in src.class_names:
                for state in reversed(src.get_states(class_name)):
                    if raw_index.get(state.source_hash) is vi:
                        running[class_name] = state
                        break
            snapshots.append(dict(running))

        for idx, vi in enumerate(ordered):
            snap_new = snapshots[idx]
            snap_old = snapshots[idx - 1] if idx > 0 else {}
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

        initial = Version(events=initial_raw, mtime=min(e.mtime for e in initial_raw)) if initial_raw else None
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
                    candidates = [v for v in non_initial_asc if v.mtime <= max_reg]
                    if candidates:
                        assigned = max(candidates)
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

        # versions is newest-first; last entry is Initial (version_number=1).
        # Numbering: Initial=1, oldest change=2, ..., newest change=N.
        non_initial = versions[:-1] if versions else []
        has_initial = len(versions) > len(non_initial)
        version_numbers: dict[str, int] = {v.version_id: 1 for v in versions}
        for i, v in enumerate(reversed(non_initial), start=2 if has_initial else 1):
            version_numbers[v.version_id] = i

        # Final index — derived from full assigned events (ADDED + CHANGED across all
        # classes). Replaces the provisional raw_index built in _build_version_groups,
        # which only covered CHANGED state_new hashes from raw groups and couldn't
        # include ADDED events (those are injected by _inject_untracked_classes).
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

    def version_number(self, version: Version) -> int | None:
        """Return the version number for a Version (Initial=1, oldest change=2, ...).

        Returns None if the version is not found in this registry.
        """
        return self._version_numbers.get(version.version_id)

    def label(self, version: Version) -> str:
        """Return the display label for a Version, e.g. 'v3 · Jun 12, 19:39'."""
        n = self._version_numbers.get(version.version_id, 0)
        v = f'v{n}'
        try:
            dt = datetime.fromisoformat(version.mtime)
            return f'{v} · {dt.strftime("%b")} {dt.day}, {dt.strftime("%H:%M")}'
        except (ValueError, AttributeError):
            return v

    def class_snapshot_at_version(self, version: Version) -> dict[str, str]:
        """Return {class_name: source_hash} for every class as of the given version.

        Reads directly from version.events (pre-computed at build time).
        Returns an empty dict if the version is not in this registry.
        """
        if version.version_id not in self._id_to_version:
            return {}
        return {e.state_new.class_name: e.state_new.source_hash for e in version.events if e.state_new}

    def is_dependency_hash_stale(self, dependency_hash: str) -> bool:
        """Return True if any class in the dependency snapshot differs from the current source.

        Compares each class's source_hash in the stored tree snapshot against
        the newest version's events. Returns False if the dependency_hash is
        unknown, has no snapshot, or there are no versions yet.
        """
        if not self.versions:
            return False
        current = self.class_snapshot_at_version(self.versions[0])
        snapshot = self._trees.get_snapshot_from_hash(dependency_hash)
        if snapshot is None:
            return False
        for class_name, node in snapshot.nodes.items():
            if not isinstance(node, dict):
                continue
            stored_hash = node.get('hash')
            if stored_hash is None:
                continue
            current_hash = current.get(class_name)
            if current_hash is not None and stored_hash != current_hash:
                return True
        return False

    def version_change_summary(self, version: Version) -> list[CodeEvent] | None:
        """Return the full per-class change summary for the given version.

        This is the pre-computed Version.events list — a direct lookup.
        Returns None if version is not in this registry.
        """
        if version.version_id not in self._id_to_version:
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
        return dict(self._dep_hash_to_version)

    def live_snapshot(self) -> dict[str, str]:
        """Return {class_name: source_hash} for the current live state.

        Prefers the in-memory hash from a loaded TrackedObject over the newest
        on-disk hash so that unsaved edits are reflected.
        """
        nodes: dict[str, str] = {}
        for class_name in self._src.class_names:
            state = self._src.get_latest_state_for_class(class_name)
            if state is None:
                continue
            cls = TrackedObject.find_object_class(class_name)
            nodes[class_name] = cls.to_code_state().source_hash if cls is not None else state.source_hash
        return nodes

    def has_live_stale(self) -> bool:
        """True when any loaded class's in-memory hash diverges from its newest on-disk hash."""
        for class_name in self._src.class_names:
            state = self._src.get_latest_state_for_class(class_name)
            if state is None:
                continue
            cls = TrackedObject.find_object_class(class_name)
            if cls is not None and cls.to_code_state().source_hash != state.source_hash:
                return True
        return False

    def snapshot_for_version(self, version_id: str) -> dict[str, str] | None:
        """Return {class_name: source_hash} for version_id, or None if not found."""
        vi = self.version_by_id(version_id)
        if vi is None:
            return None
        return self.class_snapshot_at_version(vi)

    def compare_versions(
        self,
        base: dict[str, str],
        target: dict[str, str],
    ) -> list[CodeEvent]:
        """Compare two {class_name: source_hash} snapshots; return CodeEvents sorted by class_name.

        For hashes not found on disk (e.g. an unregistered in-memory live hash),
        a CodeState is constructed via TrackedObject.to_code_state().
        """

        def _resolve(class_name: str, source_hash: str) -> CodeState | None:
            state = self._src.get_state_by_hash(source_hash)
            if state is not None:
                return state
            cls = TrackedObject.find_object_class(class_name)
            if cls is not None and cls.to_code_state().source_hash == source_hash:
                return cls.to_code_state()
            return None

        events: list[CodeEvent] = []
        for class_name in sorted(set(base) | set(target)):
            h_old = base.get(class_name)
            h_new = target.get(class_name)
            state_old = _resolve(class_name, h_old) if h_old else None
            state_new = _resolve(class_name, h_new) if h_new else None
            if state_old is None and state_new is None:
                continue
            events.append(CodeEvent(state_new=state_new, state_old=state_old))
        return events
