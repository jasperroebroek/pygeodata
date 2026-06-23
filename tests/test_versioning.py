"""Tests for versioning.py: Version event chains, change summaries, snapshots."""

import json
from pathlib import Path

import pytest

from pygeodata.config import JSONKeys
from pygeodata.registry_browser.code_service import _build_structured_hunks, _word_segments
from pygeodata.registry_types import ChangeStatus
from pygeodata.versioning import VersionRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_snapshot(
    registry: Path,
    source_hash: str,
    class_name: str,
    source_text: str = 'class Foo: pass\n',
    mtime: str = '2026-01-01T00:00:00+00:00',
    object_type: str = 'Data',
) -> None:
    code_dir = registry / 'code' / source_hash
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / 'source.py').write_text(source_text, encoding='utf-8')
    (code_dir / 'source.json').write_text(
        json.dumps({
            JSONKeys.CLASS_NAME: class_name,
            JSONKeys.OBJECT_TYPE: object_type,
            JSONKeys.SOURCE_HASH: source_hash,
            JSONKeys.REGISTERED_AT: mtime,
        }),
        encoding='utf-8',
    )


def _write_tree(registry: Path, dep_hash: str, nodes: dict) -> None:
    d = registry / 'snapshots' / dep_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / 'tree.json').write_text(
        json.dumps({JSONKeys.NODES: nodes, JSONKeys.TREE: {}}),
        encoding='utf-8',
    )


# ---------------------------------------------------------------------------
# Fixture: two classes, two versions each
#
#   MyLoader  h1  Jan-01  (Initial)
#   MyLoader  h2  Jun-01  (v1)
#   MyDep     d1  Mar-01  (Initial — single registration, dependency)
# ---------------------------------------------------------------------------


@pytest.fixture
def two_class_registry(tmp_path: Path):
    r = tmp_path / '.source'
    _write_snapshot(r, 'h1', 'MyLoader', 'class MyLoader:\n    x = 1\n', mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(r, 'h2', 'MyLoader', 'class MyLoader:\n    x = 2\n', mtime='2026-06-01T00:00:00+00:00')
    _write_snapshot(r, 'd1', 'MyDep',    'class MyDep: pass\n',          mtime='2026-03-01T00:00:00+00:00')
    _write_tree(r, 'snap_pre',  {'MyLoader': {'hash': 'h1'}, 'MyDep': {'hash': 'd1'}})
    _write_tree(r, 'snap_post', {'MyLoader': {'hash': 'h2'}, 'MyDep': {'hash': 'd1'}})
    return r


# ===========================================================================
# Version group structure
# ===========================================================================


def test_two_versions_produced(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    assert len(vr.versions) == 2


def test_initial_is_last(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    assert vr.version_number(vr.versions[-1]) == 0


def test_v1_is_first(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    assert vr.version_number(vr.versions[0]) == 1


def test_initial_class_names_includes_both(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    initial = vr.versions[-1]
    assert set(initial.class_names) == {'MyLoader', 'MyDep'}


def test_v1_class_names_includes_only_changed(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    assert v1.class_names == ['MyLoader']


# ===========================================================================
# Version.events — full change summary (ADDED/CHANGED/REMOVED/UNCHANGED)
# ===========================================================================


def test_initial_events_are_all_added(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    initial = vr.versions[-1]
    statuses = {e.class_name: e.status for e in initial.events}
    assert statuses['MyLoader'] == ChangeStatus.ADDED
    assert statuses['MyDep'] == ChangeStatus.ADDED


def test_v1_events_contain_changed_loader_and_unchanged_dep(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    statuses = {e.class_name: e.status for e in v1.events}
    assert statuses['MyLoader'] == ChangeStatus.CHANGED
    assert statuses['MyDep'] == ChangeStatus.UNCHANGED


def test_v1_changed_event_has_correct_hashes(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    loader_event = next(e for e in v1.events if e.class_name == 'MyLoader')
    assert loader_event.state_old.source_hash == 'h1'
    assert loader_event.state_new.source_hash == 'h2'


def test_initial_added_event_has_no_state_old(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    initial = vr.versions[-1]
    loader_event = next(e for e in initial.events if e.class_name == 'MyLoader')
    assert loader_event.state_old is None
    assert loader_event.state_new.source_hash == 'h1'


# ===========================================================================
# version_change_summary
# ===========================================================================


def test_version_change_summary_returns_events(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    summary = vr.version_change_summary(v1)
    assert summary is v1.events


def test_version_change_summary_from_id(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    summary = vr.version_change_summary_from_id(v1.version_id)
    assert summary is v1.events


def test_version_change_summary_unknown_id_returns_none(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    assert vr.version_change_summary_from_id('no-such-id') is None


def test_version_change_summary_to_dict_shape(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    dicts = [e.to_dict() for e in v1.events]
    changed = next(d for d in dicts if d['class_name'] == 'MyLoader')
    assert set(changed.keys()) == {'class_name', 'status', 'hash_old', 'hash_new'}
    assert changed['status'] == 'changed'
    assert changed['hash_old'] == 'h1'
    assert changed['hash_new'] == 'h2'


# ===========================================================================
# class_snapshot_at_version
# ===========================================================================


def test_snapshot_at_initial_uses_first_hashes(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    initial = vr.versions[-1]
    snap = vr.class_snapshot_at_version(initial)
    assert snap['MyLoader'] == 'h1'
    assert snap['MyDep'] == 'd1'


def test_snapshot_at_v1_uses_latest_hashes(two_class_registry):
    vr = VersionRegistry(two_class_registry)
    v1 = vr.versions[0]
    snap = vr.class_snapshot_at_version(v1)
    assert snap['MyLoader'] == 'h2'
    assert snap['MyDep'] == 'd1'


def test_snapshot_at_unknown_version_returns_empty(two_class_registry):
    from pygeodata.versioning import Version
    vr = VersionRegistry(two_class_registry)
    fake = Version(events=[], mtime='2026-01-01T00:00:00+00:00')
    assert vr.class_snapshot_at_version(fake) == {}


# ===========================================================================
# Timestamp-spread regression
#
# Three classes all changed in a single batch (no intermediate snapshot).
# Their mtimes differ slightly. class_snapshot_at_version must return the
# correct hash for each class at each version despite the spread.
# ===========================================================================


@pytest.fixture
def spread_registry(tmp_path: Path):
    r = tmp_path / '.source'
    # v1 group: A, B, C all changed together (no intermediate snapshot proves separation)
    _write_snapshot(r, 'a1', 'A', 'class A: pass\n',  mtime='2026-01-01T00:00:00+00:00')
    _write_snapshot(r, 'b1', 'B', 'class B: pass\n',  mtime='2026-01-02T00:00:00+00:00')
    _write_snapshot(r, 'c1', 'C', 'class C: pass\n',  mtime='2026-01-03T00:00:00+00:00')
    _write_snapshot(r, 'a2', 'A', 'class A: v2\n',    mtime='2026-06-01T00:00:00+00:00')
    _write_snapshot(r, 'b2', 'B', 'class B: v2\n',    mtime='2026-06-02T00:00:00+00:00')
    _write_snapshot(r, 'c2', 'C', 'class C: v2\n',    mtime='2026-06-03T00:00:00+00:00')
    return r


def test_spread_produces_two_versions(spread_registry):
    vr = VersionRegistry(spread_registry)
    assert len(vr.versions) == 2


def test_spread_v1_class_names_all_three(spread_registry):
    vr = VersionRegistry(spread_registry)
    v1 = vr.versions[0]
    assert set(v1.class_names) == {'A', 'B', 'C'}


def test_spread_snapshot_at_initial_uses_v1_hashes(spread_registry):
    vr = VersionRegistry(spread_registry)
    initial = vr.versions[-1]
    snap = vr.class_snapshot_at_version(initial)
    assert snap == {'A': 'a1', 'B': 'b1', 'C': 'c1'}


def test_spread_snapshot_at_v1_uses_v2_hashes(spread_registry):
    vr = VersionRegistry(spread_registry)
    v1 = vr.versions[0]
    snap = vr.class_snapshot_at_version(v1)
    assert snap == {'A': 'a2', 'B': 'b2', 'C': 'c2'}


def test_spread_v1_events_are_all_changed(spread_registry):
    vr = VersionRegistry(spread_registry)
    v1 = vr.versions[0]
    statuses = {e.class_name: e.status for e in v1.events}
    assert all(s == ChangeStatus.CHANGED for s in statuses.values())


# ===========================================================================
# Structured diff: _word_segments
# ===========================================================================


def test_word_segments_identical_lines():
    segs_old, segs_new = _word_segments('hello world', 'hello world')
    assert all(s['type'] == 'eq' for s in segs_old)
    assert all(s['type'] == 'eq' for s in segs_new)
    assert ''.join(s['text'] for s in segs_old) == 'hello world'


def test_word_segments_single_word_change():
    segs_old, segs_new = _word_segments('x = 1', 'x = 2')
    old_text = ''.join(s['text'] for s in segs_old)
    new_text = ''.join(s['text'] for s in segs_new)
    assert old_text == 'x = 1'
    assert new_text == 'x = 2'
    assert any(s['type'] == 'del' and '1' in s['text'] for s in segs_old)
    assert any(s['type'] == 'ins' and '2' in s['text'] for s in segs_new)


def test_word_segments_no_del_in_new(tmp_path):
    _, segs_new = _word_segments('old value', 'new value')
    assert all(s['type'] != 'del' for s in segs_new)


def test_word_segments_no_ins_in_old(tmp_path):
    segs_old, _ = _word_segments('old value', 'new value')
    assert all(s['type'] != 'ins' for s in segs_old)


# ===========================================================================
# Structured diff: _build_structured_hunks
# ===========================================================================


def test_structured_hunks_identical_files():
    hunks = _build_structured_hunks('class A: pass\n', 'class A: pass\n')
    assert hunks == []


def test_structured_hunks_basic_shape():
    hunks = _build_structured_hunks('x = 1\n', 'x = 2\n')
    assert len(hunks) == 1
    hunk = hunks[0]
    assert 'header' in hunk
    assert 'start_old' in hunk
    assert 'start_new' in hunk
    assert 'lines' in hunk


def test_structured_hunks_line_types():
    old = 'a\nb\nc\n'
    new = 'a\nB\nc\n'
    hunks = _build_structured_hunks(old, new)
    all_lines = [l for h in hunks for l in h['lines']]
    types = {l['type'] for l in all_lines}
    assert 'del' in types
    assert 'add' in types


def test_structured_hunks_line_numbers_present():
    hunks = _build_structured_hunks('x = 1\ny = 2\n', 'x = 1\ny = 3\n')
    all_lines = [l for h in hunks for l in h['lines']]
    ctx_lines = [l for l in all_lines if l['type'] == 'ctx']
    del_lines = [l for l in all_lines if l['type'] == 'del']
    add_lines = [l for l in all_lines if l['type'] == 'add']
    assert all(l['line_old'] is not None and l['line_new'] is not None for l in ctx_lines)
    assert all(l['line_old'] is not None and l['line_new'] is None for l in del_lines)
    assert all(l['line_old'] is None and l['line_new'] is not None for l in add_lines)


def test_structured_hunks_paired_lines_get_segments():
    hunks = _build_structured_hunks('x = 1\n', 'x = 2\n')
    all_lines = [l for h in hunks for l in h['lines']]
    del_line = next(l for l in all_lines if l['type'] == 'del')
    add_line = next(l for l in all_lines if l['type'] == 'add')
    assert del_line.get('segments') is not None
    assert add_line.get('segments') is not None


def test_structured_hunks_ctx_lines_have_no_segments():
    old = 'a\nb\nc\n'
    new = 'a\nB\nc\n'
    hunks = _build_structured_hunks(old, new)
    all_lines = [l for h in hunks for l in h['lines']]
    ctx_lines = [l for l in all_lines if l['type'] == 'ctx']
    assert all('segments' not in l for l in ctx_lines)


def test_structured_hunks_full_reconstruction():
    old = 'line1\nline2\nline3\n'
    new = 'line1\nLINE2\nline3\n'
    hunks = _build_structured_hunks(old, new)
    all_lines = [l for h in hunks for l in h['lines']]
    del_texts = [l['text'] for l in all_lines if l['type'] == 'del']
    add_texts = [l['text'] for l in all_lines if l['type'] == 'add']
    assert del_texts == ['line2']
    assert add_texts == ['LINE2']
