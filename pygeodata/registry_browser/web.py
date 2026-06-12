import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import uuid
from contextlib import redirect_stdout
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from pygeodata.artifact import Artifact
from pygeodata.cache import clean_cache
from pygeodata.config import get_config
from pygeodata.registry_browser import code_service, export_service
from pygeodata.registry_browser.logging import configure_logging
from pygeodata.registry_browser.path_actions import open_path, reveal_path
from pygeodata.registry_browser import payloads
from pygeodata.registry_browser.payloads import _build_table_rows, build_browser_payload
from pygeodata.registry_browser.popups import (
    build_graph_popup,
    build_json_popup,
    build_source_popup,
)
from pygeodata.registry_browser.state import AppContext
from pygeodata.versioning import VersionRegistry

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


@app.delete('/api/entry/<record_id>')
def api_delete_entry(record_id: str):
    if _ctx.is_loading() or _ctx.state is None:
        return _loading
    entry = _ctx.state.entries.get(record_id)
    if entry is None:
        abort(404)
    cache_dir = Path(entry.params_path).parent
    _assert_allowed_path(str(cache_dir))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    _ctx.start_reload()
    return jsonify({'ok': True})


@app.post('/api/clean-cache')
def api_clean_cache():
    body = request.get_json(force=True) or {}
    dry_run = bool(body.get('dry_run', True))
    buf = io.StringIO()
    with redirect_stdout(buf):
        clean_cache(dry_run=dry_run, delete_unregistered=False)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    if not dry_run:
        _ctx.start_reload()
    return jsonify({'lines': lines, 'dry_run': dry_run})


@app.get('/api/code/versions')
def api_code_versions():
    """Return merged version groups sorted newest first, with a synthetic Initial entry last."""
    if _ctx.is_loading() or _ctx.state is None:
        return _loading
    return jsonify(payloads.version_groups_payload(VersionRegistry.instance()))


@app.get('/api/code/resolve-dep-hash')
def api_code_resolve_dep_hash():
    """Resolve a dep_tree_hash to the Code-view version mtime for a specific class.

    Returns ``{"version_mtime": "<iso>" | "now", "source_hash": "<hex>"}``
    or 404 if the snapshot or class is not found.
    """
    dep_hash = request.args.get('dep_hash', '')
    class_name = request.args.get('class_name', '')
    if not dep_hash or not class_name:
        abort(400)
    result = code_service.resolve_dep_hash(dep_hash, class_name)
    if result is None:
        abort(404)
    return jsonify(result)


@app.get('/api/code/source-hash-version')
def api_code_source_hash_version():
    """Resolve a source_hash + class_name to the version group mtime it belongs to.

    Returns ``{"version_mtime": "<iso>" | "now"}`` or 404.
    """
    if _ctx.is_loading() or _ctx.state is None:
        return _loading

    source_hash = request.args.get('source_hash', '')
    class_name = request.args.get('class_name', '')
    if not source_hash or not class_name:
        abort(400)

    version_mtime = VersionRegistry.instance().version_mtime_for_source_hash(source_hash)
    if version_mtime is None:
        abort(404)

    return jsonify({'version_mtime': version_mtime})


@app.get('/api/code/version-classes')
def api_code_version_classes():
    """Return all classes at their state for the given version group.

    Pass ``mtime=<VersionInfo.mtime>`` to select a specific group, or omit / pass
    ``mtime=now`` for the newest group.
    """
    if _ctx.is_loading() or _ctx.state is None:
        return _loading
    version_mtime = request.args.get('mtime', 'now').replace(' ', '+')
    return jsonify(code_service.version_classes(version_mtime, VersionRegistry.instance()))


@app.get('/api/code/snapshot')
def api_code_snapshot():
    source_hash = request.args.get('source_hash', '')
    if not source_hash:
        abort(400)
    result = code_service.snapshot_html(source_hash, _ctx.state.code_groups if _ctx.state else None)
    if result is None:
        abort(404)
    return jsonify(result)


@app.get('/api/code/diff')
def api_code_diff():
    """Return unified diff between two source snapshots."""
    hash_a = request.args.get('hash_a', '')
    hash_b = request.args.get('hash_b', '')
    if not hash_a or not hash_b:
        abort(400)
    full = request.args.get('full') == '1'
    result = code_service.unified_diff_payload(hash_a, hash_b, full, _assert_allowed_path)
    if result is None:
        abort(404)
    return jsonify(result)


@app.get('/api/code/tree-diff')
def api_code_tree_diff():
    """Compare stored dep tree for an entry against the live tree.

    Returns per-class change status sorted: changed, removed, added, unchanged.
    """
    if _ctx.is_loading() or _ctx.state is None:
        return _loading

    record_id = request.args.get('record_id', '')
    if not record_id:
        abort(400)

    result = code_service.tree_diff(record_id, _ctx.state.entries, _ctx.state.code_groups)
    if result.get('__not_found__'):
        abort(404)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Export routes
# ---------------------------------------------------------------------------


@app.post('/api/export/start')
def api_export_start():
    if _ctx.is_loading() or _ctx.state is None:
        return _loading

    body = request.get_json(force=True) or {}
    record_ids = body.get('record_ids', [])
    include_snapshots = body.get('include_snapshots', True)

    files = export_service.collect_export_files(
        record_ids,
        _ctx.state.entries,
        include_snapshots,
        _assert_allowed_path,
    )

    job_id = str(uuid.uuid4())
    export_service.create_job(job_id, len(files))
    threading.Thread(
        target=export_service.run_export_job, args=(job_id, files), daemon=True
    ).start()
    return jsonify({'job_id': job_id, 'total': len(files)})


@app.get('/api/export/status/<job_id>')
def api_export_status(job_id: str):
    job = export_service.get_job(job_id)
    if job is None:
        abort(404)
    return jsonify(
        {
            'status': job['status'],
            'done': job['done'],
            'total': job['total'],
            'error': job['error'],
        }
    )


@app.get('/api/export/download/<job_id>')
def api_export_download(job_id: str):
    job = export_service.get_job(job_id)
    if job is None or job['status'] != 'complete' or not job['tmp_path']:
        abort(404)

    tmp_path = job['tmp_path']

    def _cleanup():
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        export_service.pop_job(job_id)

    resp = send_file(
        tmp_path,
        mimetype='application/x-tar',
        as_attachment=True,
        download_name='pygeodata_export.tar',
    )
    resp.call_on_close(_cleanup)
    return resp


@app.post('/api/export/table')
def api_export_table():
    """Return table rows for a specific set of record IDs (for the export view)."""
    if _ctx.is_loading() or _ctx.state is None:
        return _loading

    body = request.get_json(force=True) or {}
    record_ids: list[str] = body.get('record_ids', [])

    entries = _ctx.state.entries
    seen: set[str] = set()
    groups: dict[str, list] = {}
    for rid in record_ids:
        if rid in seen or rid not in entries:
            continue
        seen.add(rid)
        e = entries[rid]
        groups.setdefault(e.class_name, []).append(e)

    visible_groups = list(groups.items())
    rows = _build_table_rows(
        visible_groups=visible_groups,
        classes=_ctx.state.classes,
        selected_entry=None,
        filters=[],
        logic_mode='AND',
        row_display='all',
    )
    return jsonify({'table_rows': rows})


@app.get('/api/export/single/<record_id>')
def api_export_single(record_id: str):
    if _ctx.is_loading() or _ctx.state is None:
        return _loading

    data_path, download_name, needs_tar = export_service.single_entry_tar_path(
        record_id, _ctx.state.entries, _assert_allowed_path
    )
    if data_path is None:
        abort(404)

    if needs_tar:
        tmp_fd, tmp_name = tempfile.mkstemp(suffix='.tar')
        os.close(tmp_fd)
        with tarfile.open(tmp_name, 'w:') as tar:
            tar.add(data_path, arcname=data_path.name)
        return send_file(tmp_name, as_attachment=True, download_name=download_name)

    return send_file(data_path, as_attachment=True, download_name=download_name)


def create_app() -> Flask:
    configure_logging(logging.INFO)
    _ctx.start_load()
    return app
