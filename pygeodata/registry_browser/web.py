import json
import logging
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from pygeodata.artifact import Artifact
from pygeodata.config import get_config
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.registry_browser.logging import configure_logging
from pygeodata.registry_browser.path_actions import open_path, reveal_path
from pygeodata.registry_browser.payloads import build_browser_payload
from pygeodata.registry_browser.popups import build_graph_popup, build_json_popup, build_source_popup, render_source_html
from pygeodata.registry_browser.state import AppContext, build_version_groups
from pygeodata.tracked_object import TrackedObject

_ctx = AppContext()
_loading = ({'loading': True}, 202)

app = Flask(__name__, template_folder='templates')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.json.sort_keys = False


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
            version_filter=payload.get('version_filter') or None,
        ),
    )


@app.post('/api/rebuild')
def api_rebuild():
    _ctx.start_reload()
    return _loading


@app.get('/api/code/versions')
def api_code_versions():
    """Return merged version groups sorted newest first, with a synthetic Initial entry last."""
    if _ctx.is_loading() or _ctx.state is None:
        return _loading
    result = build_version_groups(_ctx.state.versions, _ctx.state.code_groups)
    return jsonify(result)


@app.get('/api/code/resolve-dep-hash')
def api_code_resolve_dep_hash():
    """Resolve a dep_tree_hash to the Code-view version mtime for a specific class.

    Reads ``snapshots/{dep_hash}/tree.json``, looks up the ``source_hash`` for
    ``class_name`` in its ``nodes`` dict, then finds the earliest version-change
    mtime in the code registry that is >= that hash's own mtime.

    Returns ``{"version_mtime": "<iso>" | "now", "source_hash": "<hex>"}``
    or 404 if the snapshot or class is not found.
    """
    dep_hash = request.args.get('dep_hash', '')
    class_name = request.args.get('class_name', '')
    if not dep_hash or not class_name:
        abort(400)

    tree_path = Path(get_config().path_registry) / 'snapshots' / dep_hash / 'tree.json'
    if not tree_path.exists():
        abort(404)

    try:
        tree_data = json.loads(tree_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        abort(404)

    nodes = tree_data.get('nodes', {})
    node = nodes.get(class_name)
    if not node:
        abort(404)

    source_hash = node.get('hash', '')
    if not source_hash:
        abort(404)

    # Build a hash→registered_at index from the already-scanned code groups.
    hash_to_registered_at = {
        e['source_hash']: e['mtime']
        for entries in _ctx.state.code_groups.values()
        for e in entries
        if e['source_hash']
    }

    # Max registered_at across all nodes — the true "as-of" time for this snapshot.
    # A dependency change counts just as much as a change to the class itself.
    node_mtimes = [
        hash_to_registered_at[n['hash']]
        for n in nodes.values()
        if isinstance(n, dict) and n.get('hash') in hash_to_registered_at
    ]

    if not node_mtimes:
        return jsonify({'version_mtime': 'now', 'source_hash': source_hash})

    snapshot_max_mtime = max(node_mtimes)

    # Find which code-view version group this snapshot belongs to.
    # A snapshot belongs to the group whose cutoff window it falls within.
    # Groups are: [newest-change (cutoff=now), ..., Initial (cutoff=oldest_change, exclusive)]
    # The snapshot belongs to the group whose cutoff is the earliest that is strictly
    # greater than snapshot_max_mtime — that group's window starts before the snapshot.
    version_options = build_version_groups(_ctx.state.versions, _ctx.state.code_groups)
    if not version_options:
        return jsonify({'version_mtime': 'now', 'source_hash': source_hash})

    # Each option has cutoff_mtime ('now' or ISO string) and cutoff_exclusive.
    # A snapshot belongs to the oldest group whose window contains snapshot_max_mtime.
    version_mtime = version_options[-1]['mtime']  # default to Initial
    for opt in version_options:
        cutoff = opt['cutoff_mtime']
        excl = opt['cutoff_exclusive']
        if cutoff == 'now':
            version_mtime = opt['mtime']
        elif excl and snapshot_max_mtime < cutoff:
            version_mtime = opt['mtime']
        elif not excl and snapshot_max_mtime <= cutoff:
            version_mtime = opt['mtime']

    return jsonify({'version_mtime': version_mtime, 'source_hash': source_hash})


@app.get('/api/code/version-classes')
def api_code_version_classes():
    """Return all classes at their most recent version as of the given mtime.

    Pass ``mtime=now`` (or omit) for the latest version of every class.
    Pass an ISO-8601 mtime string to get the state at that point in time.
    Pass ``exclusive=1`` to use strict ``<`` comparison (used for the Initial group,
    which represents code state just before the first change event).
    """
    if _ctx.is_loading() or _ctx.state is None:
        return _loading
    mtime_cutoff = request.args.get('mtime', 'now')
    exclusive = request.args.get('exclusive', '0') == '1'

    result = []
    for class_name, versions in sorted(_ctx.state.code_groups.items()):
        if mtime_cutoff == 'now':
            candidates = versions
        elif exclusive:
            candidates = [v for v in versions if v['mtime'] < mtime_cutoff]
        else:
            candidates = [v for v in versions if v['mtime'] <= mtime_cutoff]
        if not candidates:
            continue
        best = max(candidates, key=lambda v: v['mtime'])

        cls = TrackedObject.find_object_class(class_name)
        is_loaded = cls is not None
        is_stale = False
        if is_loaded:
            is_stale = calculate_cls_source_hash(cls) != best['source_hash']

        result.append({
            'class_name': class_name,
            'object_type': best['object_type'],
            'source_hash': best['source_hash'],
            'is_loaded': is_loaded,
            'is_stale': is_stale,
        })
    return jsonify(result)



@app.get('/api/code/snapshot')
def api_code_snapshot():
    source_hash = request.args.get('source_hash', '')
    if not source_hash:
        abort(400)

    code_dir = Path(get_config().path_registry) / 'code' / source_hash
    source_py = code_dir / 'source.py'
    meta_path = code_dir / 'source.json'

    if not source_py.exists():
        abort(404)

    try:
        source_text = source_py.read_text(encoding='utf-8')
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    except OSError:
        abort(404)

    class_name = meta.get('class_name', '')
    known_classes = frozenset(TrackedObject._registry.keys()) | frozenset(
        (_ctx.state.code_groups if _ctx.state else {}).keys()
    )
    html_body = render_source_html(source_text, known_classes, class_name)
    return jsonify({'class_name': class_name, 'html': html_body})


def create_app() -> Flask:
    configure_logging(logging.INFO)
    _ctx.start_load()
    return app
