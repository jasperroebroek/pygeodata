"""Unit tests for registry_browser/code_service.py."""

import json
from pathlib import Path

from pygeodata.config import JSONKeys, set_config
from pygeodata.registry_browser import code_service
from pygeodata.registry_browser.models import EntryInfo, SpecInfo
from pygeodata.versioning import VersionRegistry

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_code_snapshot(
    registry_path: Path,
    source_hash: str,
    class_name: str,
    source_text: str,
    mtime: str = '2026-01-01T00:00:00+00:00',
    object_type: str = 'Data',
) -> None:
    code_dir = registry_path / 'code' / source_hash
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


def _make_entry(record_id: str, dep_hash: str | None, params_path: str = '') -> EntryInfo:
    return EntryInfo(
        record_id=record_id,
        class_name='MyLoader',
        object_type='Data',
        params_path=params_path,
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash=None,
        instance_hash=None,
        params={},
        spec=SpecInfo(),
        rows=[],
        dep_hash=dep_hash,
    )


def _noop(_path: str):
    return None


# ---------------------------------------------------------------------------
# tree_diff — status branch coverage
# ---------------------------------------------------------------------------


def test_tree_diff_no_dep_hash(tmp_path: Path):
    """Entry with dep_hash=None → no_snapshot."""
    entry = _make_entry('rec1', None)
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=tmp_path / '.source'):
        vreg = VersionRegistry()
        result = code_service.tree_diff('rec1', {'rec1': entry}, vreg)
    assert result == {'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'}


def test_tree_diff_missing_entry(tmp_path: Path):
    """Unknown record_id → __not_found__ sentinel."""
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=tmp_path / '.source'):
        vreg = VersionRegistry()
        result = code_service.tree_diff('missing', {}, vreg)
    assert result.get('__not_found__') is True


def test_tree_diff_no_snapshot_file(tmp_path: Path):
    """dep_hash present but tree.json missing → no_snapshot."""
    registry = tmp_path / '.source'
    entry = _make_entry('rec1', 'nonexistent_dep_hash')
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        result = code_service.tree_diff('rec1', {'rec1': entry}, vreg)
    assert result == {'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'}


def test_tree_diff_changed(tmp_path: Path):
    """Stored hash differs from live hash → status=changed with diff."""
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'h1', 'MyLoader', 'class MyLoader:\n    x = 1\n', mtime='2026-01-01T00:00:00+00:00')
    _write_code_snapshot(registry, 'h2', 'MyLoader', 'class MyLoader:\n    x = 2\n', mtime='2026-06-01T00:00:00+00:00')

    snapshot_dir = registry / 'snapshots' / 'snapshot1'
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / 'tree.json').write_text(
        json.dumps({'nodes': {'MyLoader': {'hash': 'h1', 'object_type': 'Data'}}, 'tree': {}}),
        encoding='utf-8',
    )

    entry = _make_entry('rec1', 'snapshot1')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        result = code_service.tree_diff('rec1', {'rec1': entry}, vreg)

    changes = result['changes']
    assert len(changes) == 1
    c = changes[0]
    assert c['status'] == 'changed'
    assert c['class_name'] == 'MyLoader'
    assert c['hunks'] is not None
    assert len(c['hunks']) > 0
    all_lines = [line for hunk in c['hunks'] for line in hunk['lines']]
    del_texts = [l['text'] for l in all_lines if l['type'] == 'del']
    add_texts = [l['text'] for l in all_lines if l['type'] == 'add']
    assert any('x = 1' in t for t in del_texts)
    assert any('x = 2' in t for t in add_texts)
    assert c['full_old'] == 'class MyLoader:\n    x = 1\n'
    assert c['full_new'] == 'class MyLoader:\n    x = 2\n'


def test_tree_diff_added(tmp_path: Path):
    """Class in live but absent in stored → status=added."""
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'hnew', 'NewClass', 'class NewClass: pass\n')

    snapshot_dir = registry / 'snapshots' / 'snapshot1'
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / 'tree.json').write_text(
        json.dumps({'nodes': {}, 'tree': {}}),
        encoding='utf-8',
    )

    entry = _make_entry('rec1', 'snapshot1')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        result = code_service.tree_diff('rec1', {'rec1': entry}, vreg)

    statuses = {c['class_name']: c['status'] for c in result['changes']}
    assert statuses['NewClass'] == 'added'


def test_tree_diff_removed(tmp_path: Path):
    """Class in stored but absent in live → status=removed."""
    registry = tmp_path / '.source'

    snapshot_dir = registry / 'snapshots' / 'snapshot1'
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / 'tree.json').write_text(
        json.dumps({'nodes': {'OldClass': {'hash': 'hold', 'object_type': 'Data'}}, 'tree': {}}),
        encoding='utf-8',
    )

    entry = _make_entry('rec1', 'snapshot1')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        result = code_service.tree_diff('rec1', {'rec1': entry}, vreg)

    statuses = {c['class_name']: c['status'] for c in result['changes']}
    assert statuses['OldClass'] == 'removed'


def test_tree_diff_unchanged(tmp_path: Path):
    """Stored and live hash identical → status=unchanged, diff=None."""
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'hstable', 'StableClass', 'class StableClass: pass\n')

    snapshot_dir = registry / 'snapshots' / 'snapshot1'
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / 'tree.json').write_text(
        json.dumps({'nodes': {'StableClass': {'hash': 'hstable', 'object_type': 'Data'}}, 'tree': {}}),
        encoding='utf-8',
    )

    entry = _make_entry('rec1', 'snapshot1')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        result = code_service.tree_diff('rec1', {'rec1': entry}, vreg)

    c = result['changes'][0]
    assert c['status'] == 'unchanged'
    assert c['hunks'] is None
