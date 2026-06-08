import json
import logging
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from pygeodata.artifact import Artifact
from pygeodata.config import get_config
from pygeodata.registry_browser.logging import configure_logging
from pygeodata.registry_browser.path_actions import open_path, reveal_path
from pygeodata.registry_browser.payloads import build_browser_payload
from pygeodata.registry_browser.popups import build_graph_popup, build_json_popup, build_source_popup
from pygeodata.registry_browser.state import AppContext

_ctx = AppContext()
_loading = ({'loading': True}, 202)

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


def _allowed_roots() -> list[Path]:
    roots = [family.get_cache_root() for family in Artifact.__subclasses__()]
    roots.append(get_config().path_registry)
    return roots


def _assert_allowed_path(path_str: str) -> Path | None:
    try:
        p = Path(path_str).resolve()
    except OSError:
        abort(400)
    for root in _allowed_roots():
        try:
            p.relative_to(root.resolve())
        except ValueError:
            continue
        else:
            return p
    abort(403)
    return None


@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/popup/json')
def api_popup_json():
    path = _assert_allowed_path(request.args['path'])
    return jsonify(build_json_popup(str(path)))


@app.get('/api/popup/source')
def api_popup_source():
    return jsonify(
        build_source_popup(
            request.args['class_name'],
            source_path=request.args.get('source_path'),
        ),
    )


@app.get('/api/popup/graph')
def api_popup_graph():
    return jsonify(
        build_graph_popup(
            request.args['class_name'],
            graph_path=request.args.get('graph_path'),
        ),
    )


@app.get('/api/bounds-map')
def api_bounds_map():
    bounds = json.loads(request.args['bounds'])  # [lat_min, lon_min, lat_max, lon_max]
    crs = request.args.get('crs', '')
    lat_min, lon_min, lat_max, lon_max = bounds
    sw = f'{abs(lat_min)}° {"N" if lat_min >= 0 else "S"}, {abs(lon_min)}° {"E" if lon_min >= 0 else "W"}'
    ne = f'{abs(lat_max)}° {"N" if lat_max >= 0 else "S"}, {abs(lon_max)}° {"E" if lon_max >= 0 else "W"}'
    label = f'{sw} → {ne}'
    return render_template(
        'bounds_map.html',
        lat_min=lat_min,
        lon_min=lon_min,
        lat_max=lat_max,
        lon_max=lon_max,
        label=label,
        crs=crs,
    )


@app.get('/api/file')
def api_file():
    path = _assert_allowed_path(request.args['path'])
    return send_file(path)


@app.post('/api/open')
def api_open():
    raw = (request.get_json(force=True) or {}).get('path', '')
    open_path(str(_assert_allowed_path(raw)))
    return jsonify({'ok': True})


@app.post('/api/reveal')
def api_reveal():
    raw = (request.get_json(force=True) or {}).get('path', '')
    reveal_path(str(_assert_allowed_path(raw)))
    return jsonify({'ok': True})


@app.get('/api/status')
def api_status():
    return jsonify({'ready': _ctx.ready.is_set(), 'progress': _ctx.progress})


@app.post('/api/dashboard')
def api_dashboard():
    if _ctx.is_loading() or _ctx.state is None:
        return _loading
    payload = request.get_json(force=True) or {}
    return jsonify(
        build_browser_payload(
            _ctx.state,
            selected_classes=payload.get('selected_classes', []),
            selected_entry=payload.get('selected_entry'),
            kind_filter=payload.get('kind_filter', 'all'),
            spec_filters=payload.get('spec_filters', {}),
            filters=payload.get('filters', []),
            logic_mode=payload.get('logic_mode', 'AND'),
            row_display=payload.get('row_display', 'none'),
            hide_stale=bool(payload.get('hide_stale', False)),
        ),
    )


@app.post('/api/rebuild')
def api_rebuild():
    _ctx.start_reload()
    return _loading


def create_app() -> Flask:
    configure_logging(logging.INFO)
    _ctx.start_load()
    return app
