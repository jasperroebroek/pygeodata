import json
import logging
import threading
from pathlib import Path

from flask import Blueprint, Flask, abort, jsonify, render_template, request, send_file

from pygeodata.artifact import Artifact
from pygeodata.config import get_config

from pygeodata.registry_browser.logging import configure_logging
from pygeodata.registry_browser.path_actions import open_path, reveal_path
from pygeodata.registry_browser.payloads import build_browser_payload
from pygeodata.registry_browser.popups import build_graph_popup, build_json_popup, build_source_popup
from pygeodata.registry_browser.state import AppState, build_state


def _allowed_roots() -> list[Path]:
    """Return the set of filesystem roots that dashboard file routes may access."""
    roots = [family.get_cache_root() for family in Artifact.__subclasses__()]
    roots.append(get_config().path_registry)
    return roots


def _assert_allowed_path(path_str: str) -> Path | None:
    """Resolve path and abort with 403 if it escapes all allowed roots."""
    try:
        p = Path(path_str).resolve()
    except Exception:
        abort(400)
    for root in _allowed_roots():
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    abort(403)
    return None


# ---------------------------------------------------------------------------
# Shared app context — owns the mutable state and loading lifecycle
# ---------------------------------------------------------------------------


class AppContext:
    def __init__(self) -> None:
        self.state: AppState | None = None
        self.ready = threading.Event()
        self.progress: dict = {}

    def start_load(self) -> None:
        def _load() -> None:
            self.state = build_state(progress=self.progress)
            self.ready.set()

        threading.Thread(target=_load, daemon=True).start()

    def start_reload(self) -> None:
        self.ready.clear()
        self.progress.clear()
        self.start_load()

    def is_loading(self) -> bool:
        return not self.ready.is_set()


# ---------------------------------------------------------------------------
# Static blueprint — routes that need no app state
# ---------------------------------------------------------------------------

static_bp = Blueprint('static', __name__)


@static_bp.get('/')
def index():
    return render_template('index.html')


@static_bp.get('/api/popup/json')
def api_popup_json():
    path = _assert_allowed_path(request.args['path'])
    return jsonify(build_json_popup(str(path)))


@static_bp.get('/api/popup/source')
def api_popup_source():
    return jsonify(
        build_source_popup(
            request.args['class_name'],
            source_path=request.args.get('source_path'),
        ),
    )


@static_bp.get('/api/popup/graph')
def api_popup_graph():
    return jsonify(
        build_graph_popup(
            request.args['class_name'],
            graph_path=request.args.get('graph_path'),
        ),
    )


@static_bp.get('/api/bounds-map')
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


@static_bp.get('/api/file')
def api_file():
    path = _assert_allowed_path(request.args['path'])
    return send_file(path)


@static_bp.post('/api/open')
def api_open():
    raw = (request.get_json(force=True) or {}).get('path', '')
    open_path(str(_assert_allowed_path(raw)))
    return jsonify({'ok': True})


@static_bp.post('/api/reveal')
def api_reveal():
    raw = (request.get_json(force=True) or {}).get('path', '')
    reveal_path(str(_assert_allowed_path(raw)))
    return jsonify({'ok': True})


# ---------------------------------------------------------------------------
# State blueprint — routes that read AppContext
# ---------------------------------------------------------------------------


def make_state_blueprint(ctx: AppContext) -> Blueprint:
    bp = Blueprint('state', __name__)

    _loading = ({'loading': True}, 202)

    @bp.get('/api/status')
    def api_status():
        return jsonify({'ready': ctx.ready.is_set(), 'progress': ctx.progress})

    @bp.post('/api/dashboard')
    def api_dashboard():
        if ctx.is_loading() or ctx.state is None:
            return _loading
        payload = request.get_json(force=True) or {}
        return jsonify(
            build_browser_payload(
                ctx.state,
                selected_classes=payload.get('selected_classes', []),
                selected_entry=payload.get('selected_entry'),
                kind_filter=payload.get('kind_filter', 'all'),
                spec_filters=payload.get('spec_filters', {}),
                filters=payload.get('filters', []),
                logic_mode=payload.get('logic_mode', 'AND'),
                row_display=payload.get('row_display', 'none'),
            ),
        )

    @bp.post('/api/rebuild')
    def api_rebuild():
        ctx.start_reload()
        return _loading

    return bp


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    configure_logging(logging.INFO)
    app = Flask(__name__, template_folder='templates')
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    ctx = AppContext()
    ctx.start_load()

    app.register_blueprint(static_bp)
    app.register_blueprint(make_state_blueprint(ctx))

    return app
