"""Domain logic for the /api/code/* routes.

Routes in web.py parse request args, call these functions, and jsonify the
result.  No Flask imports here.
"""

from __future__ import annotations

from difflib import unified_diff

from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryConstructor
from pygeodata.registry_browser.popups import render_source_html
from pygeodata.tracked_object import TrackedObject
from pygeodata.versioning import VersionRegistry


def resolve_dep_hash(dep_hash: str, class_name: str, vreg: VersionRegistry) -> dict | None:
    """Return {'version_mtime': ..., 'source_hash': ...} or None if not found.

    None signals a 404; callers should abort(404).
    """
    tree = vreg.tree_registry.get_snapshot_from_hash(dep_hash)
    if tree is None:
        return None

    source_hash = tree.get_source_hash(class_name)
    if not source_hash:
        return None

    version_mtime = vreg.version_mtime_for_dep_hash(dep_hash)
    return {'version_mtime': version_mtime, 'source_hash': source_hash}


def version_classes(version_mtime: str, vreg: VersionRegistry) -> list[dict]:
    """Return per-class best CodeState as of the given version group, with live staleness.

    Pass a VersionInfo.mtime to select that group, or 'now' for the newest group.
    """
    src = vreg.source_registry

    if version_mtime == 'now':
        vi = vreg.version_groups[0] if vreg.version_groups else None
    else:
        vi = next((v for v in vreg.version_groups if v.mtime == version_mtime), None)

    if vi is None:
        return []

    groups = vreg.version_groups
    vi_idx = groups.index(vi)
    is_newest = vi_idx == 0
    upper = groups[vi_idx - 1].mtime if vi_idx > 0 else None

    result = []
    for class_name in sorted(src.class_names):
        states = src.get_states(class_name)
        if is_newest:
            candidates = states
        elif upper is not None:
            candidates = [s for s in states if s.registered_at < upper]
        else:
            candidates = states
        best = candidates[-1] if candidates else None
        if best is None:
            continue
        cls = TrackedObject.find_object_class(class_name)
        is_loaded = cls is not None
        is_stale = is_loaded and calculate_cls_source_hash(cls) != best.source_hash
        result.append(
            {
                'class_name': class_name,
                'object_type': best.object_type,
                'source_hash': best.source_hash,
                'is_loaded': is_loaded,
                'is_stale': is_stale,
            },
        )
    return result


def snapshot_html(source_hash: str, vreg: VersionRegistry) -> dict | None:
    """Return {'class_name': ..., 'html': ...} for the given source hash.

    Returns None if the snapshot is not found (caller should abort(404)).
    """
    src = vreg.source_registry
    source_text = src.get_source(source_hash)
    if source_text is None:
        return None

    class_name = src.get_class_name_from_hash(source_hash)
    known_classes = frozenset(TrackedObject._registry.keys()) | frozenset(src.class_names)
    html_body = render_source_html(source_text, known_classes, class_name)
    return {'class_name': class_name, 'html': html_body}


def diff_hashes(hash_a: str, hash_b: str, full: bool, vreg: VersionRegistry) -> dict | None:
    """Return {'diff': ..., ['full_a': ..., 'full_b': ...]} or None if either file is missing."""
    src = vreg.source_registry
    text_a = src.get_source(hash_a)
    text_b = src.get_source(hash_b)
    if text_a is None or text_b is None:
        return None

    diff = ''.join(
        unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=hash_a[:8],
            tofile=hash_b[:8],
        ),
    )
    result: dict = {'diff': diff}
    if full:
        result['full_a'] = text_a
        result['full_b'] = text_b
    return result


def unified_diff_payload(
    hash_a: str,
    hash_b: str,
    full: bool,
    assert_allowed_path: callable,
    vreg: VersionRegistry,
) -> dict | None:
    """HTTP-boundary wrapper around diff_hashes that enforces the path guard.

    ``assert_allowed_path`` is the guard from web.py; calling it here keeps
    the security check provably on the HTTP boundary for user-facing diffs.
    Returns None when either source file does not exist (caller should abort(404)).
    """
    assert_allowed_path(str(CodeRegistryConstructor.from_source_hash(hash_a).source_path))
    assert_allowed_path(str(CodeRegistryConstructor.from_source_hash(hash_b).source_path))
    return diff_hashes(hash_a, hash_b, full, vreg)


def tree_diff(record_id: str, entries: dict, vreg: VersionRegistry) -> dict:
    """Compare the stored dep tree for an entry against the live code registry.

    Returns the jsonifiable response dict.  Never raises — errors are encoded
    as {'error': 'no_snapshot', 'message': ...}.
    """
    entry = entries.get(record_id)
    if entry is None:
        return {'__not_found__': True}

    dep_hash = entry.dep_hash
    if not dep_hash:
        return {'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'}

    stored_tree = vreg.tree_registry.get_snapshot_from_hash(dep_hash)
    if stored_tree is None:
        return {'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'}

    src = vreg.source_registry
    live_nodes: dict[str, str] = {
        cn: s.source_hash for cn in src.class_names if (s := src.get_latest_state_for_class(cn)) is not None
    }

    changes = []
    all_classes = set(stored_tree.nodes.keys()) | set(live_nodes.keys())

    for class_name in sorted(all_classes):
        stored_hash = stored_tree.get_source_hash(class_name)
        live_hash = live_nodes.get(class_name)

        if stored_hash and live_hash:
            if stored_hash == live_hash:
                changes.append({'class_name': class_name, 'status': 'unchanged', 'diff': None})
            else:
                payload = diff_hashes(stored_hash, live_hash, full=True, vreg=vreg)
                if payload is not None:
                    changes.append(
                        {
                            'class_name': class_name,
                            'status': 'changed',
                            'diff': payload['diff'],
                            'full_a': payload.get('full_a'),
                            'full_b': payload.get('full_b'),
                        },
                    )
                else:
                    changes.append(
                        {
                            'class_name': class_name,
                            'status': 'changed',
                            'diff': None,
                            'full_a': None,
                            'full_b': None,
                        },
                    )
        elif stored_hash and not live_hash:
            changes.append({'class_name': class_name, 'status': 'removed', 'source_hash': stored_hash})
        else:
            changes.append({'class_name': class_name, 'status': 'added', 'source_hash': live_hash})

    order = {'changed': 0, 'removed': 1, 'added': 2, 'unchanged': 3}
    changes.sort(key=lambda c: (order[c['status']], c['class_name']))
    return {'changes': changes}
