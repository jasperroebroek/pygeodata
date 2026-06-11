import json
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys
from pygeodata.registry import SourceRegistry
from pygeodata.versioning import build_version_groups, merge_version_groups, snapshot_version_identity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_snapshot(
    registry: Path,
    source_hash: str,
    class_name: str,
    source_text: str = 'class Foo: pass',
    mtime: str = '2026-01-01T00:00:00+00:00',
    object_type: str = 'Data',
) -> None:
    code_dir = registry / 'code' / source_hash
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / 'source.py').write_text(source_text, encoding='utf-8')
    (code_dir / 'source.json').write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: class_name,
                JSONKeys.OBJECT_TYPE: object_type,
                JSONKeys.SOURCE_HASH: source_hash,
                JSONKeys.REGISTERED_AT: mtime,
            },
        ),
        encoding='utf-8',
    )


def _write_tree(registry: Path, dep_hash: str, nodes: dict) -> None:
    snapshot_dir = registry / 'snapshots' / dep_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / 'tree.json').write_text(
        json.dumps({JSONKeys.NODES: nodes, JSONKeys.TREE: {}}),
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# scan / CodeState basics
# ---------------------------------------------------------------------------


def test_empty_registry(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path / '.source')
    assert reg.class_names == []
    assert reg.latest_for_class('Anything') is None
    assert reg.hash_to_mtime('anything') is None


def test_single_snapshot_not_version_change(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'MyClass', mtime='2026-01-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    snaps = reg.get_states('MyClass')
    assert len(snaps) == 1
    assert reg.is_version_change(snaps[0]) is False


def test_second_snapshot_is_version_change(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'MyClass', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'MyClass', mtime='2026-06-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    snaps = sorted(reg.get_states('MyClass'), key=lambda s: s.registered_at)
    assert reg.is_version_change(snaps[0]) is False  # oldest: baseline
    assert reg.is_version_change(snaps[1]) is True  # second: real change


def test_latest_for_class_returns_newest(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'MyClass', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'MyClass', mtime='2026-06-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    latest = reg.latest_for_class('MyClass')
    assert latest is not None
    assert latest.source_hash == 'h2'


def test_hash_to_mtime_covers_all_hashes(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'B', mtime='2026-03-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    assert reg.hash_to_mtime('h1') == '2026-01-01T00:00:00+00:00'
    assert reg.hash_to_mtime('h2') == '2026-03-01T00:00:00+00:00'
    assert reg.hash_to_mtime('unknown') is None


def test_missing_class_name_skipped(tmp_path: Path) -> None:
    """source.json without class_name must be silently skipped."""
    registry = tmp_path / '.source'
    bad_dir = registry / 'code' / 'badhash'
    bad_dir.mkdir(parents=True)
    (bad_dir / 'source.json').write_text(json.dumps({'source_hash': 'badhash'}), encoding='utf-8')
    reg = SourceRegistry(registry)
    assert reg.class_names == []


def test_is_version_change_with_tied_mtimes(tmp_path: Path) -> None:
    """Two entries with identical mtime: oldest_mtime == both, so neither is a change."""
    registry = tmp_path / '.source'
    same_mtime = '2026-01-01T00:00:00+00:00'
    _write_snapshot(registry, 'h1', 'MyClass', mtime=same_mtime)
    _write_snapshot(registry, 'h2', 'MyClass', mtime=same_mtime)
    reg = SourceRegistry(registry)
    for state in reg.get_states('MyClass'):
        assert reg.is_version_change(state) is False


# ---------------------------------------------------------------------------
# get_source / get_snapshot_by_hash
# ---------------------------------------------------------------------------


def test_get_source_returns_text(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', source_text='class A: pass\n', mtime='2026-01-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    assert reg.get_source('h1') == 'class A: pass\n'


def test_get_source_missing_returns_none(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path / '.source')
    assert reg.get_source('nonexistent') is None


def test_get_snapshot_by_hash_returns_snapshot(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    state = reg.get_state_by_hash('h1')
    assert state is not None
    assert state.source_hash == 'h1'
    assert state.class_name == 'A'


def test_get_snapshot_by_hash_missing_returns_none(tmp_path: Path) -> None:
    reg = SourceRegistry(tmp_path / '.source')
    assert reg.get_state_by_hash('nonexistent') is None


# ---------------------------------------------------------------------------
# snapshot_version_identity — windowing algorithm
# ---------------------------------------------------------------------------


@pytest.fixture
def two_version_registry(tmp_path: Path):
    """
    MyLoader  h1  2026-01-01  (first registration — NOT a version-change)
    MyLoader  h2  2026-06-01  (version change)
    MyDep     d1  2026-03-01  (single registration, dep only)

    snapshot_pre  nodes={MyLoader:h1, MyDep:d1}  → max mtime 2026-03-01 → initial
    snapshot_post nodes={MyLoader:h2, MyDep:d1}  → max mtime 2026-06-01 → newest group
    """
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'MyLoader', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'MyLoader', mtime='2026-06-01T00:00:00+00:00')
    _write_snapshot(registry, 'd1', 'MyDep', mtime='2026-03-01T00:00:00+00:00')
    _write_tree(
        registry,
        'snapshot_pre',
        {
            'MyLoader': {'hash': 'h1', 'object_type': 'Data'},
            'MyDep': {'hash': 'd1', 'object_type': 'Data'},
        },
    )
    _write_tree(
        registry,
        'snapshot_post',
        {
            'MyLoader': {'hash': 'h2', 'object_type': 'Data'},
            'MyDep': {'hash': 'd1', 'object_type': 'Data'},
        },
    )
    return registry


def test_snapshot_identity_pre_change_is_initial(two_version_registry: Path) -> None:
    """snapshot_pre max mtime = 2026-03-01, before June change → 'initial'."""
    reg = SourceRegistry(two_version_registry)
    assert snapshot_version_identity(reg, 'snapshot_pre') == 'initial'


def test_snapshot_identity_post_change_is_newest_group(two_version_registry: Path) -> None:
    """snapshot_post max mtime = 2026-06-01, equal to cutoff of newest group (≤ inclusive) → newest group."""
    reg = SourceRegistry(two_version_registry)
    identity = snapshot_version_identity(reg, 'snapshot_post')
    assert identity == '2026-06-01T00:00:00+00:00'


def test_snapshot_identity_missing_dep_hash_returns_initial(two_version_registry: Path) -> None:
    """dep_hash with no tree.json → falls through to 'initial'."""
    reg = SourceRegistry(two_version_registry)
    assert snapshot_version_identity(reg, 'nonexistent') == 'initial'


def test_snapshot_identity_no_matching_nodes_returns_initial(tmp_path: Path) -> None:
    """Nodes whose hashes aren't in the registry index → node_mtimes empty → 'initial'."""
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'known', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'known2', 'A', mtime='2026-06-01T00:00:00+00:00')
    _write_tree(registry, 'snapshot1', {'A': {'hash': 'UNKNOWN_HASH'}})
    reg = SourceRegistry(registry)
    assert snapshot_version_identity(reg, 'snapshot1') == 'initial'


def test_snapshot_identity_no_version_groups_returns_initial(tmp_path: Path) -> None:
    """Single registration (no changes) → no version groups → identity is 'initial'."""
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_tree(registry, 'snapshot1', {'A': {'hash': 'h1'}})
    reg = SourceRegistry(registry)
    # No is_version_change=True entries → version_infos() empty → build_version_groups returns []
    assert snapshot_version_identity(reg, 'snapshot1') == 'initial'


def test_snapshot_identity_exactly_at_cutoff_inclusive(tmp_path: Path) -> None:
    """Snapshot max mtime == cutoff_mtime of newest group; cutoff_exclusive=False → newest group."""
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'A', mtime='2026-06-01T00:00:00+00:00')  # change
    _write_tree(registry, 's1', {'A': {'hash': 'h2'}})  # max mtime == cutoff 'now'-group's change
    reg = SourceRegistry(registry)
    identity = snapshot_version_identity(reg, 's1')
    assert identity == '2026-06-01T00:00:00+00:00'


# ---------------------------------------------------------------------------
# merge_version_groups / build_version_groups
# ---------------------------------------------------------------------------


def test_merge_version_groups_single_class(tmp_path: Path) -> None:
    from pygeodata.registry_types import VersionInfo

    versions = [
        VersionInfo(mtime='2026-01-01T00:00:00+00:00', class_name='A'),
        VersionInfo(mtime='2026-06-01T00:00:00+00:00', class_name='A'),
    ]
    groups = merge_version_groups(versions)
    # Same class → never merged
    assert len(groups) == 2


def test_merge_version_groups_different_classes_merge(tmp_path: Path) -> None:
    from pygeodata.registry_types import VersionInfo

    versions = [
        VersionInfo(mtime='2026-01-01T00:00:00+00:00', class_name='A'),
        VersionInfo(mtime='2026-01-01T00:00:01+00:00', class_name='B'),
    ]
    groups = merge_version_groups(versions)
    # Different classes → merged into one group
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_build_version_groups_empty_returns_empty() -> None:
    assert build_version_groups([], {}) == []


def test_build_version_groups_structure(tmp_path: Path) -> None:
    from pygeodata.registry_types import VersionInfo

    versions = [VersionInfo(mtime='2026-06-01T00:00:00+00:00', class_name='MyLoader')]
    code_groups = {
        'MyLoader': [
            {
                'source_hash': 'h1',
                'mtime': '2026-01-01T00:00:00+00:00',
                'object_type': 'Data',
                'is_version_change': False,
            },
            {
                'source_hash': 'h2',
                'mtime': '2026-06-01T00:00:00+00:00',
                'object_type': 'Data',
                'is_version_change': True,
            },
        ],
    }
    result = build_version_groups(versions, code_groups)
    # Newest-first: change group + Initial
    assert len(result) == 2
    assert result[0]['mtime'] == '2026-06-01T00:00:00+00:00'
    assert result[0]['cutoff_mtime'] == 'now'
    assert result[0]['cutoff_exclusive'] is False
    assert result[1]['mtime'] == 'initial'
    assert result[1]['exclusive'] is True
    assert result[1]['cutoff_exclusive'] is True
