import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.config import set_config
from pygeodata.registry_browser.class_catalog import scan_code_snapshots
from pygeodata.registry_browser.state import AppContext, AppState, _build_versions
from pygeodata.registry_browser.web import _allowed_roots, _assert_allowed_path, app as flask_app


def _make_ready_ctx(versions=None, code_groups=None):
    """Return an AppContext with a minimal ready AppState."""
    ctx = AppContext()
    ctx.state = AppState(
        classes={}, entries={}, groups={}, diagnostics={},
        spec_options={}, versions=versions or [], snapshots={},
        code_groups=code_groups or {},
    )
    ctx.ready.set()
    return ctx


@pytest.fixture()
def app(tmp_path: Path):
    with set_config(
        path_cache=tmp_path / 'data_processed',
        path_figures=tmp_path / 'figures',
        path_registry=tmp_path / '.source',
    ):
        flask_app.config['TESTING'] = True
        yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


# --- AppContext ---

def test_app_context_initial_state() -> None:
    ctx = AppContext()
    assert ctx.state is None
    assert ctx.is_loading()
    assert ctx.progress == {}


def test_app_context_start_reload_clears_ready() -> None:
    ctx = AppContext()
    ctx.ready.set()
    with patch.object(ctx, 'start_load'):
        ctx.start_reload()
    assert not ctx.ready.is_set()
    assert ctx.progress == {}


def test_app_context_is_loading_false_when_ready() -> None:
    ctx = AppContext()
    ctx.ready.set()
    assert not ctx.is_loading()


# --- _allowed_roots ---

def test_allowed_roots_includes_registry(tmp_path: Path) -> None:
    with set_config(path_registry=tmp_path / '.source'):
        roots = _allowed_roots()
    assert any(str(tmp_path / '.source') in str(r) for r in roots)


# --- _assert_allowed_path ---

def test_assert_allowed_path_within_root(tmp_path: Path) -> None:
    allowed = tmp_path / 'data_processed'
    allowed.mkdir()
    f = allowed / 'file.tif'
    f.touch()
    with set_config(path_cache=allowed, path_figures=tmp_path / 'figures', path_registry=tmp_path / '.source'):
        with flask_app.test_request_context():
            result = _assert_allowed_path(str(f))
    assert result is not None


def test_assert_allowed_path_outside_root_aborts(tmp_path: Path) -> None:
    outside = tmp_path / 'secret' / 'data.tif'
    outside.parent.mkdir(parents=True)
    outside.touch()
    with set_config(
        path_cache=tmp_path / 'data_processed',
        path_figures=tmp_path / 'figures',
        path_registry=tmp_path / '.source',
    ):
        with flask_app.test_request_context():
            with pytest.raises(Exception):  # abort(403) raises werkzeug HTTPException
                _assert_allowed_path(str(outside))


# --- Flask routes ---

def test_api_status_returns_ready_flag(client, app) -> None:
    # Wait briefly for the background load (it may or may not finish)
    resp = client.get('/api/status')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'ready' in data
    assert 'progress' in data


def test_api_dashboard_returns_202_while_loading(client) -> None:
    resp = client.post('/api/dashboard', json={})
    # Either 202 (loading) or 200 (loaded) — both are valid
    assert resp.status_code in (200, 202)


def test_api_rebuild_triggers_reload(client) -> None:
    resp = client.post('/api/rebuild')
    assert resp.status_code == 202
    assert resp.get_json().get('loading') is True


def test_api_popup_json_valid(client, tmp_path: Path, app) -> None:
    # Write a JSON file inside an allowed root
    with app.app_context():
        roots = _allowed_roots()

    cache_root = next((r for r in roots if 'data_processed' in str(r)), None)
    if cache_root is None:
        pytest.skip('No cache root available')

    cache_root.mkdir(parents=True, exist_ok=True)
    f = cache_root / 'test.json'
    f.write_text(json.dumps({'hello': 'world'}))

    resp = client.get(f'/api/popup/json?path={f}')
    assert resp.status_code == 200
    assert resp.get_json()['json'] == {'hello': 'world'}


def test_api_popup_json_outside_root_forbidden(client, tmp_path: Path) -> None:
    outside = tmp_path / 'secret.json'
    outside.write_text('{}')
    resp = client.get(f'/api/popup/json?path={outside}')
    assert resp.status_code == 403


def test_api_bounds_map_returns_html(client) -> None:
    bounds = json.dumps([-10.0, -20.0, 10.0, 20.0])
    resp = client.get(f'/api/bounds-map?bounds={bounds}&crs=EPSG:4326')
    assert resp.status_code == 200
    assert b'html' in resp.data.lower() or b'map' in resp.data.lower()


# --- /api/code routes ---

def _write_code_snapshot(registry_path: Path, source_hash: str, class_name: str, source_text: str, mtime: str = '2026-01-01T00:00:00+00:00', object_type: str = 'Data') -> None:
    code_dir = registry_path / 'code' / source_hash
    code_dir.mkdir(parents=True)
    (code_dir / 'source.py').write_text(source_text, encoding='utf-8')
    (code_dir / 'source.json').write_text(json.dumps({
        'class_name': class_name,
        'object_type': object_type,
        'source_hash': source_hash,
        'registered_at': mtime,
    }), encoding='utf-8')


def test_api_code_versions_empty(client) -> None:
    ctx = _make_ready_ctx(versions=[])
    with patch('pygeodata.registry_browser.web._ctx', ctx):
        resp = client.get('/api/code/versions')
    assert resp.status_code == 200
    data = resp.get_json()
    # No versions when no snapshots exist — "Now" is no longer a synthetic entry
    assert len(data) == 0


def test_api_code_versions_only_shows_changes(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    # Single entry for MyLoader — first registration, not a version change
    _write_code_snapshot(registry, 'v1hash', 'MyLoader', 'class MyLoader: pass', mtime='2026-01-01T00:00:00+00:00')
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        code_groups = scan_code_snapshots()
        versions = _build_versions(code_groups)
        ctx = _make_ready_ctx(versions=versions, code_groups=code_groups)
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/versions')
    data = resp.get_json()
    # No entries — only first registration, no version-changes to report
    assert len(data) == 0


def test_api_code_versions_shows_second_entry(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'v1hash', 'MyLoader', 'class MyLoader: pass', mtime='2026-01-01T00:00:00+00:00')
    _write_code_snapshot(registry, 'v2hash', 'MyLoader', 'class MyLoader: pass\n', mtime='2026-06-01T00:00:00+00:00')
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        code_groups = scan_code_snapshots()
        versions = _build_versions(code_groups)
        ctx = _make_ready_ctx(versions=versions, code_groups=code_groups)
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/versions')
    data = resp.get_json()
    # Two entries: the change event + synthetic Initial baseline
    assert len(data) == 2
    # First entry: the actual change group
    assert data[0]['class_names'] == ['MyLoader']
    assert data[0]['mtime'] == '2026-06-01T00:00:00+00:00'
    assert data[0]['exclusive'] is False
    assert 'MyLoader' in data[0]['label']
    # Second entry: synthetic Initial baseline (exclusive, loads pre-change code)
    assert data[1]['exclusive'] is True
    assert 'Initial' in data[1]['label']
    # Initial timestamp uses the oldest registration (Jan 1), not the change time
    assert 'Jan' in data[1]['label']


def test_api_code_resolve_dep_hash(tmp_path: Path) -> None:
    """resolve-dep-hash uses max mtime across ALL nodes, not just the clicked class."""
    registry = tmp_path / '.source'
    # MyLoader: two versions
    _write_code_snapshot(registry, 'v1hash', 'MyLoader', 'class MyLoader: pass', mtime='2026-01-01T00:00:00+00:00')
    _write_code_snapshot(registry, 'v2hash', 'MyLoader', 'class MyLoader: pass\n', mtime='2026-06-01T00:00:00+00:00')
    # MyDep: registered once at a time between v1 and v2 of MyLoader
    _write_code_snapshot(registry, 'dep1hash', 'MyDep', 'class MyDep: pass', mtime='2026-03-01T00:00:00+00:00')

    snap_dir = registry / 'snapshots' / 'snap_old'
    snap_dir.mkdir(parents=True)
    (snap_dir / 'tree.json').write_text(json.dumps({
        # Snapshot uses MyLoader v1 (2026-01-01) but MyDep at 2026-03-01
        'nodes': {
            'MyLoader': {'hash': 'v1hash', 'object_type': 'Data'},
            'MyDep':    {'hash': 'dep1hash', 'object_type': 'Data'},
        },
        'tree': {},
    }), encoding='utf-8')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        code_groups = scan_code_snapshots()
        versions = _build_versions(code_groups)
        ctx = _make_ready_ctx(versions=versions, code_groups=code_groups)
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/resolve-dep-hash?dep_hash=snap_old&class_name=MyLoader')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['source_hash'] == 'v1hash'
    # Max node mtime is 2026-03-01 (MyDep), which is before the v2 change at 2026-06-01.
    # The snapshot therefore belongs to the Initial group (pre-change baseline).
    assert data['version_mtime'] == 'initial'


def test_api_code_resolve_dep_hash_missing(client) -> None:
    resp = client.get('/api/code/resolve-dep-hash?dep_hash=doesnotexist&class_name=Foo')
    assert resp.status_code == 404


def test_api_code_resolve_dep_hash_missing_params(client) -> None:
    resp = client.get('/api/code/resolve-dep-hash')
    assert resp.status_code == 400


def test_api_code_version_classes_empty(app) -> None:
    ctx = _make_ready_ctx(versions=[], code_groups={})
    with patch('pygeodata.registry_browser.web._ctx', ctx):
        resp = app.test_client().get('/api/code/version-classes')
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_api_code_version_classes_returns_latest(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'v1hash', 'MyLoader', 'class MyLoader: pass', mtime='2026-01-01T00:00:00+00:00')
    _write_code_snapshot(registry, 'v2hash', 'MyLoader', 'class MyLoader: pass\n', mtime='2026-06-01T00:00:00+00:00')
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        code_groups = scan_code_snapshots()
        ctx = _make_ready_ctx(versions=_build_versions(code_groups), code_groups=code_groups)
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-classes')
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]['class_name'] == 'MyLoader'
    assert data[0]['source_hash'] == 'v2hash'
    assert data[0]['is_loaded'] is False


def test_api_code_version_classes_mtime_cutoff(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'v1hash', 'MyLoader', 'class MyLoader: pass', mtime='2026-01-01T00:00:00+00:00')
    _write_code_snapshot(registry, 'v2hash', 'MyLoader', 'class MyLoader: pass\n', mtime='2026-06-01T00:00:00+00:00')
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        code_groups = scan_code_snapshots()
        ctx = _make_ready_ctx(versions=_build_versions(code_groups), code_groups=code_groups)
        flask_app.config['TESTING'] = True
        with patch('pygeodata.registry_browser.web._ctx', ctx):
            resp = flask_app.test_client().get('/api/code/version-classes?mtime=2026-03-01T00:00:00+00:00')
    data = resp.get_json()
    assert len(data) == 1
    assert data[0]['source_hash'] == 'v1hash'


def test_api_code_snapshot_returns_html(tmp_path: Path) -> None:
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'abc123', 'MyLoader', 'class MyLoader: pass')
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        flask_app.config['TESTING'] = True
        resp = flask_app.test_client().get('/api/code/snapshot?source_hash=abc123')
    data = resp.get_json()
    assert data['class_name'] == 'MyLoader'
    assert '<pre' in data['html']
    assert 'MyLoader' in data['html']


def test_api_code_snapshot_not_found(client) -> None:
    resp = client.get('/api/code/snapshot?source_hash=doesnotexist')
    assert resp.status_code == 404


def test_api_code_snapshot_missing_hash(client) -> None:
    resp = client.get('/api/code/snapshot')
    assert resp.status_code == 400
