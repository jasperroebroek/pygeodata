"""Domain logic for the /api/code/* routes.

Routes in web.py parse request args, call these functions, and jsonify the
result.  No Flask imports here.
"""

from __future__ import annotations

from difflib import SequenceMatcher, unified_diff

from pygeodata.hash import calculate_cls_source_hash
from pygeodata.paths import CodeRegistryConstructor
from pygeodata.registry_browser.models import CodeClassState
from pygeodata.registry_browser.popups import render_source_html
from pygeodata.tracked_object import TrackedObject
from pygeodata.versioning import VersionRegistry


def _word_segments(text_old: str, text_new: str) -> tuple[list[dict], list[dict]]:
    """Compute word-level diff segments for a paired del/add line.

    Returns (segments_old, segments_new) where each segment is
    {'type': 'eq'|'del'|'ins', 'text': str}.
    """
    def tokenise(s: str) -> list[str]:
        import re
        return re.findall(r'\w+|\W', s)

    tokens_old = tokenise(text_old)
    tokens_new = tokenise(text_new)
    matcher = SequenceMatcher(None, tokens_old, tokens_new, autojunk=False)
    segs_old: list[dict] = []
    segs_new: list[dict] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old_text = ''.join(tokens_old[i1:i2])
        new_text = ''.join(tokens_new[j1:j2])
        if tag == 'equal':
            segs_old.append({'type': 'eq', 'text': old_text})
            segs_new.append({'type': 'eq', 'text': new_text})
        elif tag == 'replace':
            segs_old.append({'type': 'del', 'text': old_text})
            segs_new.append({'type': 'ins', 'text': new_text})
        elif tag == 'delete':
            segs_old.append({'type': 'del', 'text': old_text})
        elif tag == 'insert':
            segs_new.append({'type': 'ins', 'text': new_text})
    return segs_old, segs_new


def _build_structured_hunks(text_old: str, text_new: str) -> list[dict]:
    """Parse a unified diff into structured hunks with line numbers and word segments.

    Each hunk: {header, start_old, start_new, lines: [...]}.
    Each line: {type, text, line_old, line_new, segments?}.
    Paired del/add lines (a del immediately followed by an add) get word segments.
    """
    raw_diff = ''.join(
        unified_diff(
            text_old.splitlines(keepends=True),
            text_new.splitlines(keepends=True),
            fromfile='old',
            tofile='new',
        )
    )

    hunks: list[dict] = []
    current_hunk: dict | None = None
    line_old = 0
    line_new = 0

    for raw in raw_diff.splitlines():
        if raw.startswith('--- ') or raw.startswith('+++ '):
            continue
        if raw.startswith('@@'):
            import re
            m = re.match(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@', raw)
            line_old = int(m.group(1)) if m else 1
            line_new = int(m.group(2)) if m else 1
            current_hunk = {'header': raw, 'start_old': line_old, 'start_new': line_new, 'lines': []}
            hunks.append(current_hunk)
        elif current_hunk is not None:
            if raw.startswith('+'):
                current_hunk['lines'].append({'type': 'add', 'text': raw[1:], 'line_old': None, 'line_new': line_new})
                line_new += 1
            elif raw.startswith('-'):
                current_hunk['lines'].append({'type': 'del', 'text': raw[1:], 'line_old': line_old, 'line_new': None})
                line_old += 1
            elif raw != '\\ No newline at end of file':
                current_hunk['lines'].append({'type': 'ctx', 'text': raw[1:], 'line_old': line_old, 'line_new': line_new})
                line_old += 1
                line_new += 1

    # Pair consecutive del+add lines and add word segments
    for hunk in hunks:
        lines = hunk['lines']
        i = 0
        while i < len(lines):
            if lines[i]['type'] == 'del':
                # Collect run of dels then run of adds
                j = i
                while j < len(lines) and lines[j]['type'] == 'del':
                    j += 1
                k = j
                while k < len(lines) and lines[k]['type'] == 'add':
                    k += 1
                dels = lines[i:j]
                adds = lines[j:k]
                pairs = min(len(dels), len(adds))
                for p in range(pairs):
                    segs_old, segs_new = _word_segments(dels[p]['text'], adds[p]['text'])
                    dels[p]['segments'] = segs_old
                    adds[p]['segments'] = segs_new
                i = k
            else:
                i += 1

    return hunks


def version_classes(version_id: str, vreg: VersionRegistry) -> list[CodeClassState]:
    """Return per-class CodeState at the given version, annotated with live staleness.

    Pass a Version.version_id to select that group, or '' for the newest group.
    """
    src = vreg.source_registry
    vi = vreg.version_by_id(version_id) if version_id else (vreg.versions[0] if vreg.versions else None)
    if vi is None:
        return []

    snap = vreg.class_snapshot_at_version(vi)
    result = []
    for class_name in sorted(src.class_names):
        source_hash = snap.get(class_name)
        if source_hash is None:
            continue
        best = next((s for s in src.get_states(class_name) if s.source_hash == source_hash), None)
        if best is None:
            continue
        cls = TrackedObject.find_object_class(class_name)
        is_loaded = cls is not None
        is_stale = is_loaded and calculate_cls_source_hash(cls) != source_hash
        result.append(CodeClassState(
            class_name=class_name,
            object_type=best.object_type,
            source_hash=source_hash,
            is_loaded=is_loaded,
            is_stale=is_stale,
        ))
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


def diff_hashes(hash_old: str, hash_new: str, full: bool, vreg: VersionRegistry) -> dict | None:
    """Return structured diff payload or None if either file is missing.

    Returns {'hunks': [...], 'full_old'?: str, 'full_new'?: str}.
    """
    src = vreg.source_registry
    text_old = src.get_source(hash_old)
    text_new = src.get_source(hash_new)
    if text_old is None or text_new is None:
        return None

    result: dict = {'hunks': _build_structured_hunks(text_old, text_new)}
    if full:
        result['full_old'] = text_old
        result['full_new'] = text_new
    return result


def unified_diff_payload(
    hash_old: str,
    hash_new: str,
    full: bool,
    assert_allowed_path: callable,
    vreg: VersionRegistry,
) -> dict | None:
    """HTTP-boundary wrapper around diff_hashes that enforces the path guard.

    ``assert_allowed_path`` is the guard from web.py; calling it here keeps
    the security check provably on the HTTP boundary for user-facing diffs.
    Returns None when either source file does not exist (caller should abort(404)).
    """
    assert_allowed_path(str(CodeRegistryConstructor.from_source_hash(hash_old).source_path))
    assert_allowed_path(str(CodeRegistryConstructor.from_source_hash(hash_new).source_path))
    return diff_hashes(hash_old, hash_new, full, vreg)


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
                changes.append({'class_name': class_name, 'status': 'unchanged', 'hunks': None})
            else:
                payload = diff_hashes(stored_hash, live_hash, full=True, vreg=vreg)
                if payload is not None:
                    changes.append({
                        'class_name': class_name,
                        'status': 'changed',
                        'hunks': payload['hunks'],
                        'full_old': payload.get('full_old'),
                        'full_new': payload.get('full_new'),
                    })
                else:
                    changes.append({
                        'class_name': class_name,
                        'status': 'changed',
                        'hunks': None,
                        'full_old': None,
                        'full_new': None,
                    })
        elif stored_hash and not live_hash:
            changes.append({'class_name': class_name, 'status': 'removed', 'source_hash': stored_hash})
        else:
            changes.append({'class_name': class_name, 'status': 'added', 'source_hash': live_hash})

    order = {'changed': 0, 'removed': 1, 'added': 2, 'unchanged': 3}
    changes.sort(key=lambda c: (order[c['status']], c['class_name']))
    return {'changes': changes}
