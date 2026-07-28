/**
 * code-view.js
 *
 * Code view — 3-pane: versions | classes | source.
 * Handles loadCodeView, selectCodeVersion, selectCodeClass,
 * navigateToCodeClass, showView, and browse-mode wiring.
 */

import { $, $$, esc, badge, toast, lastDashboard } from './utils.js';
import { _viewMode, _topView, setTopView, pushHistory, updateNavBtns } from './nav.js';
import { renderClassList, toggleClass } from './class-list.js';
import { state } from './state.js';
import { scheduleSelectFirst } from './entries.js';

// ---------------------------------------------------------------------------
// Code view state
// ---------------------------------------------------------------------------

let _codeVersions        = [];   // [{version_id, mtime, class_names, label}]
let _codeClasses         = [];   // [{class_name, object_type, source_hash, is_loaded, is_stale}]
let _codeAllClasses      = [];   // full class list for class-first mode (always 'now' state)
let _codeSelectedVersion = null;  // version_id string or 'live'
let _codeSelectedClass   = null;  // class_name string
let _codeLoaded          = false;
let _codeBrowseMode      = localStorage.getItem('code_browse_mode') ?? 'version'; // 'version' | 'class'
let _codeKindFilter      = 'all'; // 'all' | 'data' | 'figure'
let _hideEmptyVersions   = localStorage.getItem('hide_empty_versions') === 'true'; // default show
let _codeDiffMode        = false;  // true while showing a unified diff in the source pane
let _diffExpand          = localStorage.getItem('diff_expand')    ?? 'hunks'; // 'hunks' | 'full'
let _hasLiveStale        = false;  // true when in-memory code diverges from newest on-disk version
let _hasLiveClasses      = false;  // true when TrackedObject._registry has any entries

// Accessors used by events.js
export function getCodeState()         { return { version: _codeSelectedVersion, className: _codeSelectedClass }; }
export function codeLoaded()           { return _codeLoaded; }
export function codeSelectedVersion()  { return _codeSelectedVersion; }
export function codeClasses()          { return _codeClasses; }
export function codeSelectedClass()    { return _codeSelectedClass; }
export function codeBrowseMode()       { return _codeBrowseMode; }
export function codeAllClasses()       { return _codeAllClasses; }
export { selectCodeClassFirst };

// Used by nav.js helpers that build snapshots
function _codeState() { return getCodeState(); }


// ---------------------------------------------------------------------------
// loadCodeView
// ---------------------------------------------------------------------------

export async function loadCodeView() {
  const resp = await fetch('/api/code/versions').then((r) => r.json());
  _codeVersions = resp.versions ?? resp;
  _hasLiveClasses = resp.has_live_classes ?? false;
  _hasLiveStale = false;
  if (_hasLiveClasses && _codeVersions.length) {
    const liveCheck = await fetch(
      `/api/code/version-diff?base_version_id=${encodeURIComponent(_codeVersions[0].version_id)}&target_version_id=live`
    ).then((r) => r.ok ? r.json() : null).catch(() => null);
    _hasLiveStale = liveCheck?.has_live_stale ?? false;
  }
  _codeAllClasses = [];
  _codeLoaded = true;
  applyCodeBrowseMode(_codeBrowseMode, { silent: true });
}


// ---------------------------------------------------------------------------
// renderCodeVersionList
// ---------------------------------------------------------------------------

function renderCodeVersionList() {
  if (_codeBrowseMode !== 'version') return;
  const el = $('#code-version-list');
  if (!el) return;

  const visible = _hideEmptyVersions
    ? _codeVersions.filter((v) => v.has_entries)
    : _codeVersions;

  if (!visible.length) {
    const msg = _hideEmptyVersions && _codeVersions.length
      ? 'All versions are empty. <a href="#" id="show-all-versions-link">Show all</a>'
      : 'No code snapshots found.';
    el.innerHTML = `<div class="detail-empty">${msg}</div>`;
    el.querySelector('#show-all-versions-link')?.addEventListener('click', (e) => {
      e.preventDefault();
      _setHideEmptyVersions(false);
    });
    return;
  }

  const liveActive = _codeSelectedVersion === 'live';
  const staleDot = _hasLiveStale ? ' <span class="stale-dot" title="In-memory code differs from disk"></span>' : '';
  const liveHtml = _hasLiveClasses ? `
    <div class="code-version-item ${liveActive ? 'active' : ''}" data-version-id="live">
      Live${staleDot}
    </div>` : '';

  el.innerHTML = liveHtml + visible.map((v) => {
    const isActive = v.version_id === _codeSelectedVersion;
    return `
      <div class="code-version-item ${isActive ? 'active' : ''}" data-version-id="${esc(v.version_id)}">
        ${esc(v.label)}
      </div>`;
  }).join('');

  el.querySelectorAll('.code-version-item').forEach((item) => {
    item.onclick = () => {
      if (item.dataset.versionId === 'live') {
        _selectLiveVersion();
      } else {
        selectCodeVersion(item.dataset.versionId);
      }
    };
  });
}


// ---------------------------------------------------------------------------
// selectCodeVersion
// ---------------------------------------------------------------------------

function _versionClassesUrl(versionMeta) {
  const version_id = versionMeta?.version_id ?? '';
  return `/api/code/version-classes?version_id=${encodeURIComponent(version_id)}`;
}

export async function selectCodeVersion(version_id, { silent = false } = {}) {
  if (!silent) pushHistory(_viewMode, _codeState());
  _codeSelectedVersion = version_id;
  _codeSelectedClass = null;
  renderCodeVersionList();

  const versionMeta = _codeVersions.find((v) => v.version_id === version_id);
  const url = _versionClassesUrl(versionMeta);
  const data = await fetch(url).then((r) => r.json());
  _codeClasses = data;
  renderCodeClassList();

  // If this version has changed classes, show a diff overview instead of auto-selecting first.
  const changedNames = versionMeta?.changed_class_names ?? [];
  if (changedNames.length) {
    await _showVersionChangeSummary(versionMeta);
    return;
  }

  // Fallback (Initial group or no changes listed): select first class
  if (_codeClasses.length) selectCodeClass(_codeClasses[0].class_name, _codeClasses[0].source_hash, { silent: true });
}

async function _selectLiveVersion({ scrollToClass = null } = {}) {
  pushHistory(_viewMode, _codeState());
  _codeSelectedVersion = 'live';
  renderCodeVersionList();
  // Load class list from newest registered version for the sidebar
  if (_codeVersions.length) {
    _codeClasses = await fetch(_versionClassesUrl(_codeVersions[0])).then((r) => r.json());
    renderCodeClassList();
  }
  await _showVersionDiff({ base: _codeVersions[0]?.version_id ?? null, target: 'live', scrollToClass });
}

// ---------------------------------------------------------------------------
// _showVersionDiff — fetch /api/code/version-diff and render with dropdowns
// ---------------------------------------------------------------------------

async function _showVersionDiff({ base, target = 'live', scrollToClass = null } = {}) {
  const panel = $('#code-source-panel');
  if (!panel) return;

  _codeDiffMode = true;
  _summaryMode = true;
  _diffAvailable = true;
  _currentDiffData = null;
  _codeSelectedClass = null;
  renderCodeClassList();
  _syncDiffSegments();

  const targetId = target ?? (_hasLiveClasses ? 'live' : (_codeVersions[0]?.version_id ?? 'live'));
  // Default base: predecessor of target in the version list, or oldest version.
  const targetIdx = targetId === 'live' ? -1 : _codeVersions.findIndex((v) => v.version_id === targetId);
  const defaultBase = targetId === 'live'
    ? (_codeVersions[0]?.version_id ?? 'none')         // Live → compare with newest registered
    : (_codeVersions[targetIdx + 1]?.version_id ?? 'none'); // vN → compare with predecessor (or empty)
  const baseId = base ?? defaultBase;

  const titleEl = $('#code-source-title');
  if (titleEl) {
    titleEl.textContent = targetId === 'live'
      ? 'Live'
      : (_codeVersions.find((v) => v.version_id === targetId)?.label ?? '');
  }
  const findBtn = $('#btn-find-in-entries');
  if (findBtn) findBtn.style.display = 'none';

  let params = `target_version_id=${encodeURIComponent(targetId)}`;
  params += `&base_version_id=${encodeURIComponent(baseId)}`;

  let result;
  try {
    result = await fetch(`/api/code/version-diff?${params}`).then((r) => r.json());
  } catch {
    panel.innerHTML = '<div class="diff-no-snapshot">Could not load diff.</div>';
    return;
  }

  if (result.error) {
    panel.innerHTML = `<div class="diff-no-snapshot">${esc(result.message ?? result.error)}</div>`;
    return;
  }

  // Base-only dropdown — target is whatever is selected in the sidebar.
  // Only include Live when classes are actually loaded in memory.
  const baseOptions = [
    ...(_hasLiveClasses ? [{ value: 'live', label: 'Live' }] : []),
    ..._codeVersions.map((v) => ({ value: v.version_id, label: v.label })),
    { value: 'none', label: '(empty)' },
  ];
  const resolvedBase = result.base_version_id ?? baseId ?? defaultBase;

  const header = document.createElement('div');
  header.className = 'diff-version-header';
  header.innerHTML =
    `<span class="diff-version-label">Compare with</span>` +
    `<select class="diff-version-select" data-role="base">` +
    baseOptions.map((o) =>
      `<option value="${esc(o.value)}"${o.value === resolvedBase ? ' selected' : ''}>${esc(o.label)}</option>`
    ).join('') +
    `</select>`;

  header.querySelector('.diff-version-select').onchange = (e) => {
    _showVersionDiff({ base: e.target.value, target: targetId });
  };

  panel.innerHTML = '';
  panel.appendChild(header);

  if (!result.changes?.length) {
    panel.insertAdjacentHTML('beforeend', '<div class="diff-no-snapshot">No changes between these versions.</div>');
    return;
  }

  _renderChangeSummary(panel, result.changes, { scrollToClass, append: true });
}


// ---------------------------------------------------------------------------
// renderCodeClassList
// ---------------------------------------------------------------------------

function renderCodeClassList() {
  if (_codeBrowseMode !== 'version') return;
  const filter = ($('#code-filter')?.value ?? '').toLowerCase();
  const el = $('#code-class-list');
  if (!el) return;

  let visible = _codeKindFilter !== 'all'
    ? _codeClasses.filter((c) => c.object_type?.toLowerCase() === _codeKindFilter)
    : _codeClasses;
  if (filter) visible = visible.filter((c) => c.class_name.toLowerCase().includes(filter));

  if (!visible.length) {
    el.innerHTML = `<div class="detail-empty">No classes.</div>`;
    return;
  }

  el.innerHTML = visible.map((c) => {
    const isActive = c.class_name === _codeSelectedClass;
    const staleDot = c.is_stale
      ? `<span class="stale-dot" title="Source changed since this version"></span>`
      : '';
    return `
      <div class="code-class-item ${isActive ? 'active' : ''}" data-cls="${esc(c.class_name)}" data-hash="${esc(c.source_hash)}">
        <span class="code-class-name">${esc(c.class_name)}${staleDot}</span>
        <span class="code-class-meta">${badge(c.object_type, 'badge-neutral')}</span>
      </div>`;
  }).join('');

  el.querySelectorAll('.code-class-item').forEach((item) => {
    item.onclick = () => {
      if (item.dataset.cls === _codeSelectedClass) {
        // Deselect — return to version change summary if this version has changed classes
        const versionMeta = _codeVersions.find((v) => v.version_id === _codeSelectedVersion);
        const changedNames = versionMeta?.changed_class_names ?? [];
        if (changedNames.length) {
          _showVersionChangeSummary(versionMeta);
        }
        return;
      }
      selectCodeClass(item.dataset.cls, item.dataset.hash);
    };
  });
}


// ---------------------------------------------------------------------------
// Code browse mode: 'version' or 'class'
// ---------------------------------------------------------------------------

async function applyCodeBrowseMode(mode, { silent = false } = {}) {
  _codeBrowseMode = mode;
  localStorage.setItem('code_browse_mode', mode);
  $$('#code-browse-tabs .kind-tab').forEach((b) =>
    b.classList.toggle('active', b.dataset.browse === mode));

  const filterVersion = $('#code-filter-classes');  // in versions pane, shown in class-first mode
  const filterClasses = $('#code-filter');           // in classes pane, shown in version-first mode

  if (mode === 'version') {
    if (filterVersion) filterVersion.style.display = 'none';
    if (filterClasses) filterClasses.style.display = '';
    renderCodeVersionList();
    if (!silent) {
      const target = _codeSelectedVersion ?? (_codeVersions[0]?.version_id ?? null);
      if (target) selectCodeVersion(target);
    }
  } else {
    if (filterVersion) filterVersion.style.display = '';
    if (filterClasses) filterClasses.style.display = 'none';
    // Build all-classes from the most recent commit on first switch
    if (!_codeAllClasses.length && _codeVersions.length) {
      const classes = await fetch(_versionClassesUrl(_codeVersions[0])).then(r => r.json());
      _codeAllClasses = classes;
    }
    renderCodeClassFirstList();
    if (_codeSelectedClass) {
      renderCodeVersionsForClass(_codeSelectedClass);
    } else if (_codeAllClasses.length && !silent) {
      selectCodeClassFirst(_codeAllClasses[0]);
    }
  }
}

function renderCodeClassFirstList() {
  const filter = ($('#code-filter-classes')?.value ?? '').toLowerCase();
  const el = $('#code-version-list');  // repurposed as class list in class-first mode
  if (!el) return;

  let visible = _codeKindFilter !== 'all'
    ? _codeAllClasses.filter((c) => c.object_type?.toLowerCase() === _codeKindFilter)
    : _codeAllClasses;
  if (filter) visible = visible.filter((c) => c.class_name.toLowerCase().includes(filter));

  if (!visible.length) {
    el.innerHTML = `<div class="detail-empty">No classes.</div>`;
    return;
  }

  el.innerHTML = visible.map((c) => {
    const isActive = c.class_name === _codeSelectedClass;
    const staleDot = c.is_stale
      ? `<span class="stale-dot" title="Source changed since this version"></span>`
      : '';
    return `
      <div class="code-class-item ${isActive ? 'active' : ''}" data-cls="${esc(c.class_name)}" data-hash="${esc(c.source_hash)}">
        <span class="code-class-name">${esc(c.class_name)}${staleDot}</span>
        <span class="code-class-meta">${badge(c.object_type, 'badge-neutral')}</span>
      </div>`;
  }).join('');

  el.querySelectorAll('.code-class-item').forEach((item) => {
    item.onclick = () => {
      const cls = _codeAllClasses.find((c) => c.class_name === item.dataset.cls);
      if (cls) selectCodeClassFirst(cls);
    };
  });
}

function renderCodeVersionsForClass(className) {
  const el = $('#code-class-list');  // repurposed as version list in class-first mode
  if (!el) return;

  // All groups that include this class — change groups first, then Initial
  const relevant = _codeVersions.filter((v) => (v.class_names ?? []).includes(className));

  if (!relevant.length) {
    el.innerHTML = `<div class="detail-empty">No snapshots for this class.</div>`;
    return;
  }

  el.innerHTML = relevant.map((v) => {
    const isActive = v.version_id === _codeSelectedVersion;
    return `
      <div class="code-version-item ${isActive ? 'active' : ''}" data-version-id="${esc(v.version_id)}">
        ${esc(v.label)}
      </div>`;
  }).join('');

  el.querySelectorAll('.code-version-item').forEach((item) => {
    item.onclick = () => selectCodeVersionForClass(item.dataset.versionId, className);
  });
}

async function selectCodeClassFirst(cls) {
  pushHistory(_viewMode, _codeState());
  _codeSelectedClass = cls.class_name;
  renderCodeClassFirstList();
  renderCodeVersionsForClass(cls.class_name);
  // Auto-select the newest version group containing this class
  const relevant = _codeVersions.filter((v) => (v.class_names ?? []).includes(cls.class_name));
  if (relevant.length) {
    const newest = relevant[0];  // _codeVersions is newest-first
    _codeSelectedVersion = newest.version_id;
    renderCodeVersionsForClass(cls.class_name);
    const classes = await fetch(_versionClassesUrl(newest)).then((r) => r.json());
    const match = classes.find((c) => c.class_name === cls.class_name);
    if (match) await selectCodeClass(match.class_name, match.source_hash, { silent: true });
  } else {
    // Fallback: no version groups yet, show current source directly
    await selectCodeClass(cls.class_name, cls.source_hash, { silent: true });
  }
}

async function selectCodeVersionForClass(version_id, className) {
  pushHistory(_viewMode, _codeState());
  _codeSelectedVersion = version_id;
  renderCodeVersionsForClass(className);

  try {
    const versionMeta = _codeVersions.find((v) => v.version_id === version_id);
    const classes = await fetch(_versionClassesUrl(versionMeta)).then((r) => r.json());
    const match = classes.find((c) => c.class_name === className);
    if (match) await selectCodeClass(match.class_name, match.source_hash, { silent: true });
  } catch {
    toast('Could not load version');
  }
}

// ---------------------------------------------------------------------------
// Snapshot renderer — builds a source table from data.lines
// ---------------------------------------------------------------------------

function _renderSnapshotLines(lines) {
  const rows = lines.map((line, i) => {
    let cell;
    if (line.cls_link) {
      cell = `<a class="src-cls-link" data-cls="${esc(line.cls_link)}">${esc(line.text)}</a>`;
    } else {
      cell = esc(line.text);
    }
    return `<tr class="diff-ctx"><td class="diff-ln">${i + 1}</td><td class="diff-code">${cell}</td></tr>`;
  });
  return `<table class="diff-table diff-table--inline source-table">${rows.join('')}</table>`;
}

// ---------------------------------------------------------------------------
// Diff renderer — inline / side-by-side / expand
// ---------------------------------------------------------------------------

let _diffViewMode = localStorage.getItem('diff_view_mode') ?? 'inline'; // 'inline' | 'split'

function _segmentsHtml(segments, showType) {
  if (!segments) return null;
  return segments.map((s) => {
    if (s.type === 'eq') return esc(s.text);
    if (s.type === showType) return `<span class="diff-word-${showType === 'del' ? 'del' : 'add'}">${esc(s.text)}</span>`;
    return '';
  }).join('');
}

function _renderInlineHunk(hunk) {
  const rows = [];
  for (const line of hunk.lines) {
    const lo = line.line_old ?? '';
    const ln = line.line_new ?? '';
    if (line.type === 'ctx') {
      rows.push(`<tr class="diff-ctx"><td class="diff-ln">${lo}</td><td class="diff-ln">${ln}</td><td class="diff-sign"> </td><td class="diff-code">${esc(line.text)}</td></tr>`);
    } else if (line.type === 'del') {
      const html = _segmentsHtml(line.segments, 'del') ?? esc(line.text);
      rows.push(`<tr class="diff-del"><td class="diff-ln">${lo}</td><td class="diff-ln"></td><td class="diff-sign">−</td><td class="diff-code">${html}</td></tr>`);
    } else {
      const html = _segmentsHtml(line.segments, 'ins') ?? esc(line.text);
      rows.push(`<tr class="diff-add"><td class="diff-ln"></td><td class="diff-ln">${ln}</td><td class="diff-sign">+</td><td class="diff-code">${html}</td></tr>`);
    }
  }
  return rows.join('');
}

function _renderSplitHunk(hunk) {
  const rows = [];
  for (const line of hunk.lines) {
    const lo = line.line_old ?? '';
    const ln = line.line_new ?? '';
    if (line.type === 'ctx') {
      rows.push(`<tr class="diff-ctx"><td class="diff-ln">${lo}</td><td class="diff-code">${esc(line.text)}</td><td class="diff-ln">${ln}</td><td class="diff-code">${esc(line.text)}</td></tr>`);
    } else if (line.type === 'del') {
      const html = _segmentsHtml(line.segments, 'del') ?? esc(line.text);
      rows.push(`<tr><td class="diff-ln diff-del">${lo}</td><td class="diff-code diff-del">${html}</td><td class="diff-ln diff-empty-side"></td><td class="diff-code diff-empty-side"></td></tr>`);
    } else {
      const html = _segmentsHtml(line.segments, 'ins') ?? esc(line.text);
      rows.push(`<tr><td class="diff-ln diff-empty-side"></td><td class="diff-code diff-empty-side"></td><td class="diff-ln diff-add">${ln}</td><td class="diff-code diff-add">${html}</td></tr>`);
    }
  }
  return rows.join('');
}

/**
 * Render a structured diff payload into a DOM element.
 * @param {Element} container
 * @param {{hunks: Array, full_old?: string, full_new?: string}} diffData
 */
function _renderDiffInto(container, diffData) {
  const { hunks, full_old, full_new } = diffData;

  if (!hunks || !hunks.length) {
    container.innerHTML = '<div class="diff-empty-msg">Files are identical — no changes.</div>';
    return;
  }

  const isExpanded = _diffExpand === 'full' && !!(full_old && full_new);
  const isSplit    = _diffViewMode === 'split';
  const splitCls   = isSplit ? 'diff-table--split' : 'diff-table--inline';
  const colgroup   = isSplit ? `<colgroup><col style="width:44px"><col><col style="width:44px"><col></colgroup>` : '';
  const renderHunk = isSplit ? _renderSplitHunk : _renderInlineHunk;

  if (isExpanded) {
    const linesOld = full_old.split('\n');
    const linesNew = full_new.split('\n');
    if (linesOld[linesOld.length - 1] === '') linesOld.pop();
    if (linesNew[linesNew.length - 1] === '') linesNew.pop();

    const rows = [];
    let curOld = 1, curNew = 1;

    function ctxRow(lo, ln) {
      if (isSplit) {
        return `<tr class="diff-ctx"><td class="diff-ln">${lo}</td><td class="diff-code">${esc(linesOld[lo - 1] ?? '')}</td><td class="diff-ln">${ln}</td><td class="diff-code">${esc(linesNew[ln - 1] ?? '')}</td></tr>`;
      }
      return `<tr class="diff-ctx"><td class="diff-ln">${lo}</td><td class="diff-ln">${ln}</td><td class="diff-sign"> </td><td class="diff-code">${esc(linesNew[ln - 1] ?? '')}</td></tr>`;
    }

    for (const hunk of hunks) {
      while (curOld < hunk.start_old && curNew < hunk.start_new) {
        rows.push(ctxRow(curOld++, curNew++));
      }
      while (curOld < hunk.start_old) rows.push(ctxRow(curOld++, curNew));
      while (curNew < hunk.start_new) rows.push(ctxRow(curOld, curNew++));
      rows.push(renderHunk(hunk));
      // Advance cursors past hunk lines
      for (const line of hunk.lines) {
        if (line.line_old !== null) curOld = line.line_old + 1;
        if (line.line_new !== null) curNew = line.line_new + 1;
      }
    }
    while (curOld <= linesOld.length && curNew <= linesNew.length) {
      rows.push(ctxRow(curOld++, curNew++));
    }
    container.innerHTML = `<table class="diff-table ${splitCls}">${colgroup}${rows.join('')}</table>`;
  } else {
    const rows = [];
    for (const hunk of hunks) {
      rows.push(`<tr class="diff-hunk-hdr"><td colspan="4" class="diff-hunk-hdr-cell">${esc(hunk.header)}</td></tr>`);
      rows.push(renderHunk(hunk));
    }
    container.innerHTML = `<table class="diff-table ${splitCls}">${colgroup}${rows.join('')}</table>`;
  }
}

// ---------------------------------------------------------------------------
// Topbar segment state — drives #code-src-diff-seg, #code-diff-mode-seg,
// #code-diff-expand-seg. Call _syncDiffSegments() whenever state changes.
// ---------------------------------------------------------------------------

// Holds the last fetched diff data so mode/expand toggle can re-render without refetching.
let _currentDiffData  = null;
// When in diff mode, the available diffs for the current class (prev / current).
let _diffOptions      = []; // [{label, hashA, hashB}]
let _diffOptionIdx    = 0;  // which option is selected
let _diffAvailable    = false; // true when selected class has at least one diffable version
let _selectClassSeq   = 0;
let _summaryMode      = false; // true when showing a multi-class version/tree-diff summary

function _syncDiffSegments() {
  const controls = $('#code-diff-controls');
  if (controls) controls.style.display = _diffAvailable ? 'flex' : 'none';

  // Source | Diff — hidden in summary mode (no single source to toggle back to)
  const srcDiffSeg = $('#code-src-diff-seg');
  if (srcDiffSeg) srcDiffSeg.style.display = _summaryMode ? 'none' : '';
  if (!_summaryMode) {
    $$('#code-src-diff-seg .kind-tab').forEach((b) =>
      b.classList.toggle('active', (b.dataset.srcmode === 'diff') === _codeDiffMode));
  }

  // Inline | Side by side — enabled only in diff mode
  const modeSeg = $('#code-diff-mode-seg');
  if (modeSeg) {
    modeSeg.classList.toggle('seg-disabled', !_codeDiffMode);
    modeSeg.querySelectorAll('.kind-tab').forEach((b) =>
      b.classList.toggle('active', b.dataset.dmode === _diffViewMode));
  }

  // Hunks | Full — enabled only in diff mode and only when full text available
  const expandSeg = $('#code-diff-expand-seg');
  if (expandSeg) {
    // In summary mode (_currentDiffData is null), enable if any rendered body has full_old
    const summaryHasFull = !_currentDiffData && _codeDiffMode
      && [...$$('#code-source-panel .tree-diff-body[data-diff]')].some((b) => {
          try { return !!JSON.parse(b.dataset.diff).full_old; } catch { return false; }
        });
    const canExpand = _codeDiffMode && (!!(_currentDiffData?.full_old) || summaryHasFull);
    expandSeg.classList.toggle('seg-disabled', !canExpand);
    expandSeg.querySelectorAll('.kind-tab').forEach((b) =>
      b.classList.toggle('active', b.dataset.expand === _diffExpand));
  }
}

function _rerenderCurrentDiff() {
  const panel = $('#code-source-panel');
  if (!panel) return;
  if (_currentDiffData) {
    _renderDiffInto(panel, _currentDiffData);
    _renderDiffHeader(panel);
    return;
  }
  // Re-render all open tree-diff bodies (version change summary or tree-diff view).
  // Bodies that were rendered have their diff data stored in a dataset attribute.
  panel.querySelectorAll('.tree-diff-body[data-rendered][data-diff]').forEach((body) => {
    try {
      _renderDiffInto(body, JSON.parse(body.dataset.diff));
    } catch { /* skip corrupt */ }
  });
}

// Resolve diff options for the currently selected class (prev version / current version).
async function _buildDiffOptions() {
  const selectedEntry = _codeClasses.find((c) => c.class_name === _codeSelectedClass);
  if (!selectedEntry) return [];

  const currentVersion   = _codeVersions[0] ?? null;
  const isLive           = _codeSelectedVersion === 'live';
  const effectiveIdx     = isLive ? 0 : _codeVersions.findIndex((v) => v.version_id === _codeSelectedVersion);
  const prevVersion      = effectiveIdx >= 0 ? _codeVersions[effectiveIdx + 1] ?? null : null;
  const isViewingCurrent = isLive || (currentVersion && _codeSelectedVersion === currentVersion.version_id);

  const opts = [];

  if (prevVersion) {
    try {
      const classes = await fetch(_versionClassesUrl(prevVersion)).then((r) => r.json());
      const match   = classes.find((c) => c.class_name === _codeSelectedClass);
      if (match && match.source_hash !== selectedEntry.source_hash) {
        opts.push({ label: prevVersion.label, hash_old: match.source_hash, hash_new: selectedEntry.source_hash });
      }
    } catch { /* skip */ }
  }

  if (!isViewingCurrent && currentVersion) {
    const currentEntry = _codeAllClasses.find((c) => c.class_name === _codeSelectedClass)
      ?? _codeClasses.find((c) => c.class_name === _codeSelectedClass);
    if (currentEntry && currentEntry.source_hash !== selectedEntry.source_hash) {
      const currentLabel = _hasLiveClasses ? 'Live' : currentVersion.label;
      opts.push({ label: currentLabel, hash_old: selectedEntry.source_hash, hash_new: currentEntry.source_hash });
    }
  }

  return opts;
}

function _renderDiffHeader(panel) {
  const existing = panel.querySelector('.diff-version-header');
  if (existing) existing.remove();
  if (!_diffOptions.length) return;

  const header = document.createElement('div');
  header.className = 'diff-version-header';
  header.innerHTML =
    `<span class="diff-version-label">Compare with</span>` +
    `<select class="diff-version-select">` +
    _diffOptions.map((o, i) =>
      `<option value="${i}"${i === _diffOptionIdx ? ' selected' : ''}>${esc(o.label)}</option>`
    ).join('') +
    `</select>`;

  header.querySelector('select').onchange = async (e) => {
    _diffOptionIdx = +e.target.value;
    await _loadAndRenderDiff(_diffOptions[_diffOptionIdx]);
  };
  panel.prepend(header);
}

async function _loadAndRenderDiff(opt) {
  try {
    const data = await fetch(
      `/api/code/diff?hash_old=${encodeURIComponent(opt.hash_old)}&hash_new=${encodeURIComponent(opt.hash_new)}&full=1`
    ).then((r) => r.json());
    _currentDiffData = data;
    const panel = $('#code-source-panel');
    if (panel) {
      _renderDiffInto(panel, data);
      _renderDiffHeader(panel);
    }
    _syncDiffSegments();
  } catch { toast('Diff unavailable'); }
}

export async function enterDiffMode() {
  if (_codeDiffMode) return;
  _diffOptions = await _buildDiffOptions();
  if (!_diffOptions.length) { toast('No previous version to diff against'); return; }
  _diffOptionIdx = 0;
  _codeDiffMode = true;
  await _loadAndRenderDiff(_diffOptions[0]);
}

export async function showDiff(hash_old, hash_new) {
  _diffOptions = [{ label: '', hash_old, hash_new }];
  _diffOptionIdx = 0;
  _codeDiffMode = true;
  await _loadAndRenderDiff(_diffOptions[0]);
}

export function exitDiffMode() {
  _codeDiffMode = false;
  _summaryMode = false;
  _currentDiffData = null;
  _diffOptions = [];
  const entry = _codeClasses.find((c) => c.class_name === _codeSelectedClass);
  if (entry) selectCodeClass(entry.class_name, entry.source_hash, { silent: true });
  else _syncDiffSegments();
}

export async function showWhatChanged(recordId, entryClassName = null) {
  try {
    pushHistory(_viewMode, null);
    showView('code');
    if (!_codeLoaded) await loadCodeView();

    // Resolve the base version for this entry (the version it was produced at).
    const data = await fetch(
      `/api/code/version-diff?record_id=${encodeURIComponent(recordId)}`
    ).then((r) => r.json());

    if (data.error === 'no_snapshot' || data.error === 'bad_request') {
      const panel = $('#code-source-panel');
      if (panel) panel.innerHTML = `<div class="diff-no-snapshot">${esc(data.message ?? 'Snapshot not available for this entry')}</div>`;
      _codeDiffMode = false;
      _summaryMode = false;
      _currentDiffData = null;
      _syncDiffSegments();
      return;
    }

    const baseVersionId = data.base_version_id ?? null;
    _hasLiveStale = data.has_live_stale ?? false;

    // Always diff against live so the stale entry shows what actually changed.
    // Sidebar selection falls back to newest version when classes aren't loaded.
    const sidebarTarget = _hasLiveClasses ? 'live' : (_codeVersions[0]?.version_id ?? 'live');
    _codeSelectedVersion = sidebarTarget;

    const baseVersionMeta = _codeVersions.find((v) => v.version_id === baseVersionId) ?? _codeVersions[0];
    if (baseVersionMeta) {
      _codeClasses = await fetch(_versionClassesUrl(baseVersionMeta)).then((r) => r.json());
    }
    renderCodeVersionList();

    await _showVersionDiff({ base: baseVersionId, target: 'live', scrollToClass: entryClassName });
  } catch { toast('Could not load diff'); }
}

const _STATUS_ORDER = { changed: 0, added: 1, unchanged: 2, removed: 3 };

function _renderChangeSummary(panel, changes, { scrollToClass = null, append = false } = {}) {
  // Sort: changed → added → unchanged → removed
  changes = [...changes].sort((a, b) =>
    (_STATUS_ORDER[a.status] ?? 99) - (_STATUS_ORDER[b.status] ?? 99)
  );

  const wrap = document.createElement('div');
  wrap.className = 'tree-diff-container';
  wrap.innerHTML = changes.map((c) => {
    const open = c.status === 'changed' ? ' open' : '';
    const canExpand = (c.status === 'changed'   && c.hash_old && c.hash_new)
                   || (c.status === 'added'     && c.hash_new)
                   || (c.status === 'removed'   && c.hash_old)
                   || (c.status === 'unchanged' && c.hash_new);
    return `<div class="tree-diff-class ${esc(c.status)}${open}" data-class="${esc(c.class_name)}"
                 data-hash-old="${esc(c.hash_old ?? '')}" data-hash-new="${esc(c.hash_new ?? '')}">
      <div class="tree-diff-header">
        <span class="tree-diff-status">${esc(c.status)}</span>
        <span class="tree-diff-name">${esc(c.class_name)}</span>
        ${canExpand ? '<span class="tree-diff-expand">▶</span>' : ''}
      </div>
      ${canExpand ? `<div class="tree-diff-body"></div>` : ''}
    </div>`;
  }).join('');
  if (!append) panel.innerHTML = '';
  panel.appendChild(wrap);

  wrap.querySelectorAll('.tree-diff-header').forEach((hdr) => {
    const card = hdr.parentElement;
    const body = card.querySelector('.tree-diff-body');
    const hashOld = card.dataset.hashOld;
    const hashNew = card.dataset.hashNew;
    const status = card.classList.contains('unchanged') ? 'unchanged' : null;

    const _maybeRender = async () => {
      if (!body || body.dataset.rendered) return;
      body.dataset.rendered = '1';
      body.innerHTML = '<div class="diff-loading">Loading…</div>';
      try {
        if (hashOld && hashNew && status !== 'unchanged') {
          const data = await fetch(
            `/api/code/diff?hash_old=${encodeURIComponent(hashOld)}&hash_new=${encodeURIComponent(hashNew)}&full=1`
          ).then((r) => r.json());
          body.dataset.diff = JSON.stringify(data);
          _renderDiffInto(body, data);
          _syncDiffSegments();
        } else {
          const sourceHash = hashNew || hashOld;
          const data = await fetch(
            `/api/code/snapshot?source_hash=${encodeURIComponent(sourceHash)}`
          ).then((r) => r.json());
          if (data.lines) {
            body.innerHTML = _renderSnapshotLines(data.lines);
            bindCodeSourceLinks();
          } else {
            body.innerHTML = '<div class="diff-no-snapshot">Source unavailable</div>';
          }
        }
      } catch { body.innerHTML = '<div class="diff-no-snapshot">Unavailable</div>'; }
    };

    hdr.onclick = () => {
      if (!body) return;
      card.classList.toggle('open');
      if (card.classList.contains('open')) _maybeRender();
    };

    if (card.classList.contains('open')) _maybeRender();
  });

  if (scrollToClass) {
    const target = wrap.querySelector(`.tree-diff-class[data-class="${CSS.escape(scrollToClass)}"]`);
    target?.scrollIntoView({ block: 'nearest' });
  }
}

async function _showVersionChangeSummary(versionMeta) {
  await _showVersionDiff({ target: versionMeta.version_id });
}


export async function selectCodeClass(className, sourceHash, { silent = false } = {}) {
  if (!silent) pushHistory(_viewMode, _codeState());
  const seq = ++_selectClassSeq;
  _codeSelectedClass = className;
  _codeDiffMode = false;
  _summaryMode = false;
  _diffAvailable = false;
  renderCodeClassList();

  // Update source pane header
  const titleEl = $('#code-source-title');
  if (titleEl) titleEl.textContent = className;
  const findBtn = $('#btn-find-in-entries');
  if (findBtn) findBtn.style.display = '';

  const [data, diffOpts] = await Promise.all([
    fetch(`/api/code/snapshot?source_hash=${encodeURIComponent(sourceHash)}`).then((r) => r.json()).catch(() => null),
    _buildDiffOptions(),
  ]);

  if (seq !== _selectClassSeq) return;

  _diffAvailable = diffOpts.length > 0;

  if (data?.lines) {
    const panel = $('#code-source-panel');
    if (panel) {
      panel.innerHTML = _renderSnapshotLines(data.lines);
      bindCodeSourceLinks();
    }
  } else {
    toast('Source unavailable');
  }

  _syncDiffSegments();

  // Scroll selected class into view
  $(`#code-class-list .code-class-item[data-cls="${CSS.escape(className)}"]`)
    ?.scrollIntoView({ block: 'nearest' });
}

function bindCodeSourceLinks() {
  $('#code-source-panel')?.querySelectorAll('.src-cls-link').forEach((el) => {
    el.onclick = (e) => {
      e.preventDefault();
      const match = _codeClasses.find((c) => c.class_name === el.dataset.cls);
      if (match) selectCodeClass(match.class_name, match.source_hash);  // not silent: user navigation
    };
  });
}

export async function navigateToCodeClass(className, depHash = null) {
  pushHistory(_viewMode, _topView === 'code' ? _codeState() : null);
  showView('code');
  if (!_codeLoaded) await loadCodeView();

  // Resolve which version to show. When depHash is provided, ask the server to
  // find the version that was current when the entry was computed.
  // Fall back to the most recent commit if no depHash.
  let targetVersion = _codeVersions[0]?.version_id ?? null;
  let resolvedSourceHash = null;
  if (depHash) {
    try {
      const res = await fetch(
        `/api/code/resolve-dep-hash?dep_hash=${encodeURIComponent(depHash)}&class_name=${encodeURIComponent(className)}`
      ).then((r) => r.ok ? r.json() : null);
      if (res) {
        targetVersion = res.version_id;
        resolvedSourceHash = res.source_hash;
      }
    } catch { /* fall through to newest */ }
  }

  // Always use by-version mode for navigateToCodeClass (source/dep navigation)
  if (_codeBrowseMode !== 'version') {
    await applyCodeBrowseMode('version', { silent: true });
  }

  if (targetVersion && targetVersion !== _codeSelectedVersion) {
    await selectCodeVersion(targetVersion, { silent: true });
  }

  // Use the resolved source_hash directly if available, otherwise find by class name
  const match = resolvedSourceHash
    ? _codeClasses.find((c) => c.source_hash === resolvedSourceHash)
    : _codeClasses.find((c) => c.class_name === className);
  if (match) selectCodeClass(match.class_name, match.source_hash, { silent: true });
}

/**
 * Navigate to a specific source_hash snapshot for a class in the Code view.
 * Unlike navigateToCodeClass (which resolves via dep_hash), this directly selects
 * the version group whose snapshot matches source_hash.
 */
export async function navigateToCodeClassBySourceHash(className, sourceHash) {
  if (!className || !sourceHash) return;
  pushHistory(_viewMode, _topView === 'code' ? _codeState() : null);
  showView('code');
  if (!_codeLoaded) await loadCodeView();

  if (_codeBrowseMode !== 'version') {
    await applyCodeBrowseMode('version', { silent: true });
  }

  let targetVersion = _codeVersions[0]?.version_id ?? null;
  try {
    const res = await fetch(
      `/api/code/source-hash-version?source_hash=${encodeURIComponent(sourceHash)}&class_name=${encodeURIComponent(className)}`
    ).then((r) => r.ok ? r.json() : null);
    if (res?.version_id) targetVersion = res.version_id;
  } catch { /* fall through to newest */ }

  if (targetVersion && targetVersion !== _codeSelectedVersion) {
    await selectCodeVersion(targetVersion, { silent: true });
  }

  const match = _codeClasses.find((c) => c.class_name === className && c.source_hash === sourceHash)
    ?? _codeClasses.find((c) => c.class_name === className);
  if (match) selectCodeClass(match.class_name, match.source_hash, { silent: true });
}

export function showView(view, { pushNav = false } = {}) {
  if (pushNav && view !== _topView) {
    pushHistory(_viewMode, _topView === 'code' ? _codeState() : null);
  }
  setTopView(view);
  const toEntries = view === 'entries';
  const toCode    = view === 'code';
  const toExport  = view === 'export';
  document.querySelector('.view-entries').style.display = toEntries ? '' : 'none';
  document.querySelector('.view-code').style.display    = toCode    ? '' : 'none';
  document.querySelector('.view-export').style.display  = toExport  ? '' : 'none';
  document.getElementById('entries-toolbar').style.display       = toEntries ? '' : 'none';
  document.getElementById('entries-toolbar-right').style.display = toEntries ? '' : 'none';
  document.getElementById('code-toolbar').style.display          = toCode    ? '' : 'none';
  document.getElementById('code-toolbar-right').style.display    = toCode    ? '' : 'none';
  $$('.view-tab').forEach((t) => t.classList.toggle('active', t.dataset.view === view));

  // Export tab always shows select controls; restore actual state when leaving.
  const selectBtn = document.getElementById('btn-select-mode');
  if (toExport) {
    document.body.classList.add('select-mode');
    if (selectBtn) selectBtn.classList.add('active');
  } else {
    document.body.classList.toggle('select-mode', !!state.select_mode);
    if (selectBtn) selectBtn.classList.toggle('active', !!state.select_mode);
  }

  if (toExport) _onShowExportView();
  localStorage.setItem('view_mode_top', view);
  updateNavBtns();
}

// Lazy reference — injected from export-view.js at boot time
let _onShowExportView = () => {};
export function setOnShowExportView(fn) { _onShowExportView = fn; }


// ---------------------------------------------------------------------------
// Event wiring specific to code view
// ---------------------------------------------------------------------------

// View tab wiring
$$('.view-tab').forEach((tab) => {
  tab.onclick = () => {
    const view = tab.dataset.view;
    showView(view, { pushNav: true });
    if (view === 'code' && !_codeLoaded) loadCodeView();
  };
});

// Restore saved view — include export
{
  const savedView = localStorage.getItem('view_mode_top') ?? 'entries';
  if (savedView === 'export') showView('export');
}

$('#code-filter')?.addEventListener('input', renderCodeClassList);
$('#code-filter-classes')?.addEventListener('input', () => {
  if (_codeBrowseMode === 'class') renderCodeClassFirstList();
});

$$('#code-browse-tabs .kind-tab').forEach((btn) => {
  btn.onclick = () => applyCodeBrowseMode(btn.dataset.browse);
});

$$('#code-kind-tabs .kind-tab').forEach((btn) => {
  btn.onclick = () => {
    _codeKindFilter = btn.dataset.codeKind;
    $$('#code-kind-tabs .kind-tab').forEach((b) =>
      b.classList.toggle('active', b === btn));
    if (_codeBrowseMode === 'class') renderCodeClassFirstList();
    else renderCodeClassList();
  };
});

document.getElementById('entry-class-filter')?.addEventListener('input', () =>
  renderClassList(lastDashboard?.class_cards ?? [])
);

$('#btn-find-in-entries')?.addEventListener('click', () => {
  if (!_codeSelectedClass) return;
  state.version_filter = _codeSelectedVersion ?? null;
  scheduleSelectFirst();
  showView('entries', { pushNav: true });
  toggleClass(_codeSelectedClass, { navigate: true });
});

// ---------------------------------------------------------------------------
// Topbar diff segment wiring
// ---------------------------------------------------------------------------

$$('#code-src-diff-seg .kind-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (btn.dataset.srcmode === 'diff') {
      if (!_codeDiffMode) enterDiffMode();
    } else {
      if (_codeDiffMode) exitDiffMode();
    }
  });
});

$$('#code-diff-mode-seg .kind-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (!_codeDiffMode) return;
    _diffViewMode = btn.dataset.dmode;
    localStorage.setItem('diff_view_mode', _diffViewMode);
    _syncDiffSegments();
    _rerenderCurrentDiff();
  });
});

$$('#code-diff-expand-seg .kind-tab').forEach((btn) => {
  btn.addEventListener('click', () => {
    if (!_codeDiffMode) return;
    _diffExpand = btn.dataset.expand;
    localStorage.setItem('diff_expand', _diffExpand);
    _syncDiffSegments();
    _rerenderCurrentDiff();
  });
});

// Hide-empty-versions toggle
function _setHideEmptyVersions(hide) {
  _hideEmptyVersions = hide;
  localStorage.setItem('hide_empty_versions', hide ? 'true' : 'false');
  const btn = document.getElementById('btn-hide-empty-versions');
  if (btn) btn.classList.toggle('active', hide);
  renderCodeVersionList();
}

{
  const btn = document.getElementById('btn-hide-empty-versions');
  if (btn) {
    btn.classList.toggle('active', _hideEmptyVersions);
    btn.addEventListener('click', () => _setHideEmptyVersions(!_hideEmptyVersions));
  }
}
