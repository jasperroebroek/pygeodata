import json
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys
from pygeodata.registry import SourceRegistry
from pygeodata.versioning import VersionRegistry

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
    assert reg.get_latest_state_for_class('Anything') is None
    assert reg.get_mtime_from_hash('anything') is None


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
    latest = reg.get_latest_state_for_class('MyClass')
    assert latest is not None
    assert latest.source_hash == 'h2'


def test_hash_to_mtime_covers_all_hashes(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'B', mtime='2026-03-01T00:00:00+00:00')
    reg = SourceRegistry(registry)
    assert reg.get_mtime_from_hash('h1') == '2026-01-01T00:00:00+00:00'
    assert reg.get_mtime_from_hash('h2') == '2026-03-01T00:00:00+00:00'
    assert reg.get_mtime_from_hash('unknown') is None


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
# VersionRegistry — version group and dep_hash assignment
# ---------------------------------------------------------------------------


@pytest.fixture
def two_version_registry(tmp_path: Path):
    """
    MyLoader  h1  2026-01-01  (initial registration)
    MyLoader  h2  2026-06-01  (version change)
    MyDep     d1  2026-03-01  (single registration, dep only)

    snapshot_pre  nodes={MyLoader:h1, MyDep:d1}  → uses h1 (Initial hash) → Initial
    snapshot_post nodes={MyLoader:h2, MyDep:d1}  → uses h2 (change hash)  → v1
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


def test_version_groups_single_change(two_version_registry: Path) -> None:
    """One class with two states → one change group (v2) + first group (v1)."""
    vr = VersionRegistry(two_version_registry)
    assert len(vr.versions) == 2
    v2, v1 = vr.versions
    assert vr.version_number(v2) == 2
    assert v2.changed_class_names == ['MyLoader']
    assert vr.version_number(v1) == 1
    assert 'MyLoader' in v1.class_names


def test_snapshot_pre_change_assigned_to_initial(two_version_registry: Path) -> None:
    """snapshot_pre uses h1 (Initial hash, not a change event) → assigned to Initial."""
    vr = VersionRegistry(two_version_registry)
    initial = vr.versions[-1]
    assert vr.version_for_dep_hash('snapshot_pre') == initial


def test_snapshot_post_change_assigned_to_v1(two_version_registry: Path) -> None:
    """snapshot_post uses h2 (the change hash) → assigned to v1."""
    vr = VersionRegistry(two_version_registry)
    v1 = vr.versions[0]
    assert vr.version_for_dep_hash('snapshot_post') == v1


def test_missing_dep_hash_returns_none(two_version_registry: Path) -> None:
    vr = VersionRegistry(two_version_registry)
    assert vr.version_for_dep_hash('nonexistent') is None


def test_snapshot_with_unknown_node_hash_assigned_to_initial(tmp_path: Path) -> None:
    """Nodes whose hashes aren't in the registry → falls back to Initial."""
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'A', mtime='2026-06-01T00:00:00+00:00')
    _write_tree(registry, 'snap1', {'A': {'hash': 'UNKNOWN_HASH'}})
    vr = VersionRegistry(registry)
    initial = vr.versions[-1]
    assert vr.version_for_dep_hash('snap1') == initial


def test_no_changes_produces_only_initial(tmp_path: Path) -> None:
    """Single registration per class → no change events → only one group (v1)."""
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_tree(registry, 'snap1', {'A': {'hash': 'h1'}})
    vr = VersionRegistry(registry)
    assert len(vr.versions) == 1
    assert vr.version_number(vr.versions[0]) == 1


def test_snapshot_using_change_hash_assigned_to_its_group(tmp_path: Path) -> None:
    """Snapshot containing the change hash maps to that version group."""
    registry = tmp_path / '.source'
    _write_snapshot(registry, 'h1', 'A', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(registry, 'h2', 'A', mtime='2026-06-01T00:00:00+00:00')
    _write_tree(registry, 'snap1', {'A': {'hash': 'h2'}})
    vr = VersionRegistry(registry)
    v1 = vr.versions[0]
    assert vr.version_for_dep_hash('snap1') == v1
