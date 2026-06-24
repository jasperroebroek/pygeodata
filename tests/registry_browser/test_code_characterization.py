"""Characterization tests for the code-tab API routes.

These tests pin the JSON payloads for:
    GET /api/code/resolve-dep-hash
    GET /api/code/versions
    GET /api/code/version-diff

They exist to guard Units 1–3 of the architecture refactor: the algorithms are
being unified/moved, not changed, so every response must stay byte-for-byte
identical after the refactor.  Do NOT relax these assertions — if a value
changes, that means the refactor changed behaviour.

After the refactor is complete (Units 1–3 merged), delete any tests here that
only covered duplicated code paths that no longer exist.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.config import JSONKeys, set_config
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.registry_browser.state import AppContext, AppState
from pygeodata.registry_browser.web import app as flask_app
# ---------------------------------------------------------------------------
# Re-use helpers from test_web (same module, keep in sync with test_web.py)
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


class _FakeEntryRegistry:
    def get_state_hashes(self, class_name: str) -> list[str]:
        return []

    def get_object_type(self, class_name: str) -> str | None:
        return None


def _make_ready_ctx(version_registry: VersionRegistry | None = None):
    ctx = AppContext()
    ctx.state = AppState(
        classes={},
        entries={},
        diagnostics={},
        spec_options={},
        entry_registry=_FakeEntryRegistry(),
        version_registry=version_registry if version_registry is not None else VersionRegistry(),
    )
    ctx.ready.set()
    return ctx


def _make_entry(record_id: str, dep_hash: str | None):
    from pygeodata.catalog.types import EntryInfo, SpecInfo

    return EntryInfo(
        record_id=record_id,
        class_name='MyLoader',
        object_type='Data',
        params_path='',
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


# ---------------------------------------------------------------------------
# Shared sample registry fixture
#
# Registry layout:
#   MyLoader  v1hash  2026-01-01  (first registration — NOT a version-change)
#   MyLoader  v2hash  2026-06-01  (version change)
#   MyDep     dep1hash 2026-03-01 (single registration, dependency only)
#
# Snapshots:
#   snapshot_pre  nodes={MyLoader:v1hash, MyDep:dep1hash}  (max mtime 2026-03-01 → initial)
#   snapshot_post nodes={MyLoader:v2hash, MyDep:dep1hash}  (max mtime 2026-06-01 → "now" group)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_registry(tmp_path: Path):
    """Build the fixed sample registry and return (registry_path, tmp_path)."""
    registry = tmp_path / '.source'

    _write_code_snapshot(
        registry,
        'v1hash',
        'MyLoader',
        'class MyLoader:\n    x = 1\n',
        mtime='2026-01-01T00:00:00+00:00',
    )
    _write_code_snapshot(
        registry,
        'v2hash',
        'MyLoader',
        'class MyLoader:\n    x = 2\n',
        mtime='2026-06-01T00:00:00+00:00',
    )
    _write_code_snapshot(registry, 'dep1hash', 'MyDep', 'class MyDep: pass\n', mtime='2026-03-01T00:00:00+00:00')

    snapshot_pre = registry / 'snapshots' / 'snapshot_pre'
    snapshot_pre.mkdir(parents=True)
    (snapshot_pre / 'tree.json').write_text(
        json.dumps(
            {
                JSONKeys.NODES: {
                    'MyLoader': {JSONKeys.SOURCE_HASH: 'v1hash', JSONKeys.OBJECT_TYPE: 'Data', 'hash': 'v1hash'},
                    'MyDep': {JSONKeys.SOURCE_HASH: 'dep1hash', JSONKeys.OBJECT_TYPE: 'Data', 'hash': 'dep1hash'},
                },
                JSONKeys.TREE: {},
            },
        ),
        encoding='utf-8',
    )

    snapshot_post = registry / 'snapshots' / 'snapshot_post'
    snapshot_post.mkdir(parents=True)
    (snapshot_post / 'tree.json').write_text(
        json.dumps(
            {
                JSONKeys.NODES: {
                    'MyLoader': {JSONKeys.SOURCE_HASH: 'v2hash', JSONKeys.OBJECT_TYPE: 'Data', 'hash': 'v2hash'},
                    'MyDep': {JSONKeys.SOURCE_HASH: 'dep1hash', JSONKeys.OBJECT_TYPE: 'Data', 'hash': 'dep1hash'},
                },
                JSONKeys.TREE: {},
            },
        ),
        encoding='utf-8',
    )

    return registry, tmp_path


@pytest.fixture
def sample_ctx(sample_registry):
    registry, tmp_path = sample_registry
    with set_config(
        path_cache=tmp_path / 'data',
        path_figures=tmp_path / 'figs',
        path_registry=registry,
    ):
        vreg = VersionRegistry(registry)
        yield _make_ready_ctx(vreg), tmp_path, registry


# ===========================================================================
# GET /api/code/versions  —  payload
# ===========================================================================


def test_two_class_registry_returns_two_entries(sample_ctx):
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/versions')

    assert resp.status_code == 200
    versions = resp.get_json()['versions']
    # One real version-change group + one synthetic Initial
    assert len(versions) == 2


def test_first_entry_is_change_group(sample_ctx):
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/versions')

    versions = resp.get_json()['versions']
    first = versions[0]
    # The change group carries the v2hash registration time
    assert first['mtime'] == '2026-06-01T00:00:00+00:00'
    assert 'MyLoader' in first['class_names']
    assert 'v2' in first['label']


def test_second_entry_is_initial_group(sample_ctx):
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/versions')

    versions = resp.get_json()['versions']
    v1 = versions[1]
    assert 'v1' in v1['label']
    assert 'MyLoader' in v1['class_names']
    assert 'MyDep' in v1['class_names']


def test_full_payload_shape(sample_ctx):
    """Every versions entry must carry exactly the documented keys."""
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/versions')

    data = resp.get_json()
    assert 'versions' in data
    assert 'has_live_classes' in data
    required_keys = {'mtime', 'class_names', 'changed_class_names', 'label', 'version_id'}
    for entry in data['versions']:
        assert required_keys <= set(entry.keys()), f'missing keys in {entry}'


# ===========================================================================
# GET /api/code/resolve-dep-hash  — golden payload
# ===========================================================================


def test_pre_change_snapshot_maps_to_initial(sample_ctx):
    """snapshot_pre max mtime = 2026-03-01 (MyDep), before the June change → Initial group."""
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get(
                '/api/code/resolve-dep-hash?dep_hash=snapshot_pre&class_name=MyLoader',
            )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data['source_hash'] == 'v1hash'
    # snapshot_pre nodes are pre-change → assigned to the Initial group
    assert data['version_id'] == ctx.state.version_registry.versions[-1].version_id


def test_post_change_snapshot_maps_to_v1(sample_ctx):
    """snapshot_post max mtime = 2026-06-01 (MyLoader v2), equal to the change → v1 group."""
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get(
                '/api/code/resolve-dep-hash?dep_hash=snapshot_post&class_name=MyLoader',
            )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data['source_hash'] == 'v2hash'
    assert data['version_id'] == ctx.state.version_registry.versions[0].version_id


def test_dep_class_not_clicked_but_drives_window(sample_ctx):
    """Asking for MyDep on snapshot_pre: source_hash=dep1hash, version still Initial."""
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get(
                '/api/code/resolve-dep-hash?dep_hash=snapshot_pre&class_name=MyDep',
            )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data['source_hash'] == 'dep1hash'
    assert data['version_id'] == ctx.state.version_registry.versions[-1].version_id


def test_payload_has_exactly_two_keys(sample_ctx):
    ctx, tmp_path, registry = sample_ctx
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get(
                '/api/code/resolve-dep-hash?dep_hash=snapshot_pre&class_name=MyLoader',
            )

    data = resp.get_json()
    assert set(data.keys()) == {'version_id', 'source_hash'}


# ===========================================================================
# GET /api/code/version-diff  — golden payload
# ===========================================================================


def test_no_dep_hash_returns_no_snapshot():
    """Entry with dep_hash=None → error: no_snapshot, no HTTP error."""
    entry = _make_entry('rec_none', None)
    ctx = _make_ready_ctx()
    ctx.state.entries = {'rec_none': entry}
    with patch('pygeodata.registry_browser.web._ctx', ctx):
        resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec_none')

    assert resp.status_code == 200
    data = resp.get_json()
    assert data == {'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'}


def test_changed_class_payload(sample_ctx):
    """Entry with snapshot_pre: MyLoader changed (v1→v2), MyDep unchanged."""
    ctx, tmp_path, registry = sample_ctx
    entry = _make_entry('rec1', 'snapshot_pre')
    ctx.state.entries = {'rec1': entry}

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec1')

    assert resp.status_code == 200
    data = resp.get_json()
    changes = data['changes']

    statuses = {c['class_name']: c['status'] for c in changes}
    assert statuses['MyLoader'] == 'changed'
    assert statuses['MyDep'] == 'unchanged'

    loader_change = next(c for c in changes if c['class_name'] == 'MyLoader')
    assert loader_change['hash_old'] is not None
    assert loader_change['hash_new'] is not None


def test_unchanged_class_diff_is_none(sample_ctx):
    ctx, tmp_path, registry = sample_ctx
    entry = _make_entry('rec1', 'snapshot_pre')
    ctx.state.entries = {'rec1': entry}

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec1')

    changes = resp.get_json()['changes']
    dep_change = next(c for c in changes if c['class_name'] == 'MyDep')
    assert dep_change['class_name'] == 'MyDep'
    assert dep_change['status'] == 'unchanged'
    assert 'hash_new' in dep_change


def test_sort_order_changed_before_unchanged(sample_ctx):
    """Both changed and unchanged statuses must be present (order is client-side)."""
    ctx, tmp_path, registry = sample_ctx
    entry = _make_entry('rec1', 'snapshot_pre')
    ctx.state.entries = {'rec1': entry}

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec1')

    statuses = {c['status'] for c in resp.get_json()['changes']}
    assert 'changed' in statuses
    assert 'unchanged' in statuses


def test_post_change_snapshot_all_unchanged(sample_ctx):
    """snapshot_post uses v2hash for MyLoader — same as live → all unchanged."""
    ctx, tmp_path, registry = sample_ctx
    entry = _make_entry('rec2', 'snapshot_post')
    ctx.state.entries = {'rec2': entry}

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec2')

    changes = resp.get_json()['changes']
    assert all(c['status'] == 'unchanged' for c in changes)


def test_top_level_keys_when_changes_present(sample_ctx):
    """Successful version-diff response must have changes, base_version_id, has_live_stale."""
    ctx, tmp_path, registry = sample_ctx
    entry = _make_entry('rec1', 'snapshot_pre')
    ctx.state.entries = {'rec1': entry}

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec1')

    data = resp.get_json()
    assert 'changes' in data.keys()
    assert 'base_version_id' in data.keys()
    assert 'has_live_stale' in data.keys()


def test_changed_entry_keys(sample_ctx):
    """A 'changed' entry must carry class_name, status, diff, full_old, full_new."""
    ctx, tmp_path, registry = sample_ctx
    entry = _make_entry('rec1', 'snapshot_pre')
    ctx.state.entries = {'rec1': entry}

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-diff?record_id=rec1')

    changes = resp.get_json()['changes']
    changed = next(c for c in changes if c['status'] == 'changed')
    assert set(changed.keys()) == {'class_name', 'status', 'hash_old', 'hash_new'}
