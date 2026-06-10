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
from difflib import unified_diff
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from pygeodata.artifact import Artifact
from pygeodata.cache import clean_cache
from pygeodata.config import get_config
from pygeodata.hash import calculate_cls_source_hash
from pygeodata.registry_browser.logging import configure_logging
from pygeodata.registry_browser.path_actions import open_path, reveal_path
from pygeodata.registry_browser.payloads import _build_table_rows, build_browser_payload
from pygeodata.registry_browser.popups import (
    build_graph_popup,
    build_json_popup,
    build_source_popup,
    render_source_html,
)
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
        e['source_hash']: e['mtime'] for entries in _ctx.state.code_groups.values() for e in entries if e['source_hash']
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
        if cutoff == 'now' or (excl and snapshot_max_mtime < cutoff) or (not excl and snapshot_max_mtime <= cutoff):
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

        result.append(
            {
                'class_name': class_name,
                'object_type': best['object_type'],
                'source_hash': best['source_hash'],
                'is_loaded': is_loaded,
                'is_stale': is_stale,
            }
        )
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
        (_ctx.state.code_groups if _ctx.state else {}).keys(),
    )
    html_body = render_source_html(source_text, known_classes, class_name)
    return jsonify({'class_name': class_name, 'html': html_body})


@app.get('/api/code/diff')
def api_code_diff():
    """Return unified diff between two source snapshots."""
    hash_a = request.args.get('hash_a', '')
    hash_b = request.args.get('hash_b', '')
    if not hash_a or not hash_b:
        abort(400)

    registry = Path(get_config().path_registry)
    path_a = registry / 'code' / hash_a / 'source.py'
    path_b = registry / 'code' / hash_b / 'source.py'

    # Validate both paths are within the allowed registry root
    _assert_allowed_path(str(path_a))
    _assert_allowed_path(str(path_b))

    if not path_a.exists() or not path_b.exists():
        abort(404)

    try:
        lines_a = path_a.read_text(encoding='utf-8').splitlines(keepends=True)
        lines_b = path_b.read_text(encoding='utf-8').splitlines(keepends=True)
    except OSError:
        abort(404)

    diff = ''.join(unified_diff(lines_a, lines_b, fromfile=hash_a[:8], tofile=hash_b[:8]))
    result: dict = {'diff': diff}
    if request.args.get('full') == '1':
        result['full_a'] = ''.join(lines_a)
        result['full_b'] = ''.join(lines_b)
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

    entry = _ctx.state.entries.get(record_id)
    if entry is None:
        abort(404)

    dep_hash = entry.dep_hash
    if not dep_hash:
        return jsonify({'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'})

    registry = Path(get_config().path_registry)
    tree_path = registry / 'snapshots' / dep_hash / 'tree.json'
    if not tree_path.exists():
        return jsonify({'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'})

    try:
        stored_tree = json.loads(tree_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return jsonify({'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'})

    stored_nodes: dict[str, dict] = stored_tree.get('nodes', {})

    # Build live nodes: prefer loaded class, fall back to latest snapshot in code registry
    live_nodes: dict[str, str] = {}  # class_name → source_hash
    for class_name, entries in _ctx.state.code_groups.items():
        if not entries:
            continue
        best = max(entries, key=lambda e: e['mtime'])
        live_nodes[class_name] = best['source_hash']

    changes = []
    all_classes = set(stored_nodes.keys()) | set(live_nodes.keys())

    for class_name in sorted(all_classes):
        stored_hash = stored_nodes.get(class_name, {}).get('hash') if class_name in stored_nodes else None
        live_hash = live_nodes.get(class_name)

        if stored_hash and live_hash:
            if stored_hash == live_hash:
                changes.append({'class_name': class_name, 'status': 'unchanged', 'diff': None})
            else:
                # Produce diff + full file texts for expand mode
                path_a = registry / 'code' / stored_hash / 'source.py'
                path_b = registry / 'code' / live_hash / 'source.py'
                diff_text = None
                full_a = None
                full_b = None
                if path_a.exists() and path_b.exists():
                    try:
                        lines_a = path_a.read_text(encoding='utf-8').splitlines(keepends=True)
                        lines_b = path_b.read_text(encoding='utf-8').splitlines(keepends=True)
                        diff_text = ''.join(
                            unified_diff(lines_a, lines_b, fromfile=stored_hash[:8], tofile=live_hash[:8])
                        )
                        full_a = ''.join(lines_a)
                        full_b = ''.join(lines_b)
                    except OSError:
                        pass
                changes.append(
                    {
                        'class_name': class_name,
                        'status': 'changed',
                        'diff': diff_text,
                        'full_a': full_a,
                        'full_b': full_b,
                    }
                )
        elif stored_hash and not live_hash:
            changes.append({'class_name': class_name, 'status': 'removed', 'source_hash': stored_hash})
        else:
            changes.append({'class_name': class_name, 'status': 'added', 'source_hash': live_hash})

    order = {'changed': 0, 'removed': 1, 'added': 2, 'unchanged': 3}
    changes.sort(key=lambda c: (order[c['status']], c['class_name']))

    return jsonify({'changes': changes})


# ---------------------------------------------------------------------------
# Export job registry  {job_id: {status, done, total, tmp_path, error}}
# ---------------------------------------------------------------------------

_export_jobs: dict[str, dict] = {}
_export_jobs_lock = threading.Lock()


def _collect_export_files(record_ids: list[str], include_snapshots: bool) -> list[tuple[Path, str]]:
    """Return list of (absolute_path, arcname) for all files to be exported."""
    files: list[tuple[Path, str]] = []
    seen_src_hashes: set[str] = set()
    seen_dep_hashes: set[str] = set()
    registry = Path(get_config().path_registry)

    for record_id in record_ids:
        entry = _ctx.state.entries.get(record_id)
        if entry is None:
            continue

        cache_dir = Path(entry.params_path).parent
        _assert_allowed_path(str(cache_dir))
        for f in cache_dir.iterdir():
            if f.is_file():
                files.append((f, f'cache/{cache_dir.name}/{f.name}'))

        if include_snapshots and entry.dep_hash and entry.dep_hash not in seen_dep_hashes:
            seen_dep_hashes.add(entry.dep_hash)
            tree_path = registry / 'snapshots' / entry.dep_hash / 'tree.json'
            if tree_path.exists():
                files.append((tree_path, f'snapshots/{entry.dep_hash}/tree.json'))
                try:
                    tree_data = json.loads(tree_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    tree_data = {}
                for node in tree_data.get('nodes', {}).values():
                    src_hash = node.get('hash') if isinstance(node, dict) else None
                    if src_hash and src_hash not in seen_src_hashes:
                        seen_src_hashes.add(src_hash)
                        code_dir = registry / 'code' / src_hash
                        if code_dir.exists():
                            for f in code_dir.iterdir():
                                if f.is_file():
                                    files.append((f, f'code/{src_hash}/{f.name}'))

    return files


def _run_export_job(job_id: str, files: list[tuple[Path, str]]) -> None:
    job = _export_jobs[job_id]
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix='.tar')
        os.close(fd)
        with tarfile.open(tmp_path, mode='w') as tar:
            for i, (path, arcname) in enumerate(files):
                tar.add(path, arcname=arcname)
                with _export_jobs_lock:
                    job['done'] = i + 1
        with _export_jobs_lock:
            job['tmp_path'] = tmp_path
            job['status'] = 'complete'
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        with _export_jobs_lock:
            job['status'] = 'error'
            job['error'] = str(exc)


@app.post('/api/export/start')
def api_export_start():
    if _ctx.is_loading() or _ctx.state is None:
        return _loading

    body = request.get_json(force=True) or {}
    record_ids = body.get('record_ids', [])
    include_snapshots = body.get('include_snapshots', True)

    with app.app_context():
        files = _collect_export_files(record_ids, include_snapshots)

    job_id = str(uuid.uuid4())
    with _export_jobs_lock:
        _export_jobs[job_id] = {
            'status': 'running',
            'done': 0,
            'total': len(files),
            'tmp_path': None,
            'error': None,
        }

    threading.Thread(target=_run_export_job, args=(job_id, files), daemon=True).start()
    return jsonify({'job_id': job_id, 'total': len(files)})


@app.get('/api/export/status/<job_id>')
def api_export_status(job_id: str):
    with _export_jobs_lock:
        job = _export_jobs.get(job_id)
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
    with _export_jobs_lock:
        job = _export_jobs.get(job_id)
    if job is None or job['status'] != 'complete' or not job['tmp_path']:
        abort(404)

    tmp_path = job['tmp_path']

    def _cleanup():
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        with _export_jobs_lock:
            _export_jobs.pop(job_id, None)

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
    # Group by class_name preserving the requested order, deduplicated
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

    entry = _ctx.state.entries.get(record_id)
    if entry is None:
        abort(404)

    cache_dir = Path(entry.params_path).parent
    _assert_allowed_path(str(cache_dir))

    data_path = next(
        (f for f in cache_dir.iterdir() if not f.name.startswith('.')),
        None,
    )
    if data_path is None:
        abort(404)

    if data_path.is_dir():
        tmp_fd, tmp_name = tempfile.mkstemp(suffix='.tar')
        os.close(tmp_fd)
        with tarfile.open(tmp_name, 'w:') as tar:
            tar.add(data_path, arcname=data_path.name)
        return send_file(tmp_name, as_attachment=True, download_name=f'{data_path.name}.tar')

    return send_file(data_path, as_attachment=True, download_name=data_path.name)


def create_app() -> Flask:
    configure_logging(logging.INFO)
    _ctx.start_load()
    return app
