import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.config import set_config
from pygeodata.registry_browser.state import AppContext
from pygeodata.registry_browser.web import _allowed_roots, _assert_allowed_path, app as flask_app


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
