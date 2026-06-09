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

let _codeVersions        = [];   // [{mtime, class_names, label, is_now}]
let _codeClasses         = [];   // [{class_name, object_type, source_hash, is_loaded, is_stale}]
let _codeAllClasses      = [];   // full class list for class-first mode (always 'now' state)
let _codeSelectedVersion = null;  // mtime string, or 'now'
let _codeSelectedClass   = null;  // class_name string
let _codeLoaded          = false;
let _codeBrowseMode      = localStorage.getItem('code_browse_mode') ?? 'version'; // 'version' | 'class'
let _codeKindFilter      = 'all'; // 'all' | 'data' | 'figure'

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
  const versions = await fetch('/api/code/versions').then((r) => r.json());
  _codeVersions = versions;
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

  if (!_codeVersions.length) {
    el.innerHTML = `<div class="detail-empty">No code snapshots found.</div>`;
    return;
  }

  el.innerHTML = _codeVersions.map((v) => {
    const isActive = v.mtime === _codeSelectedVersion;
    return `
      <div class="code-version-item ${isActive ? 'active' : ''}" data-mtime="${esc(v.mtime)}">
        ${esc(v.label)}
      </div>`;
  }).join('');

  el.querySelectorAll('.code-version-item').forEach((item) => {
    item.onclick = () => selectCodeVersion(item.dataset.mtime);
  });
}


// ---------------------------------------------------------------------------
// selectCodeVersion
// ---------------------------------------------------------------------------

function _versionClassesUrl(versionMeta) {
  const cutoff = versionMeta?.cutoff_mtime ?? versionMeta?.mtime ?? 'now';
  const exclusive = versionMeta?.cutoff_exclusive ? '&exclusive=1' : '';
  return `/api/code/version-classes?mtime=${encodeURIComponent(cutoff)}${exclusive}`;
}

export async function selectCodeVersion(mtime, { silent = false } = {}) {
  if (!silent) pushHistory(_viewMode, _codeState());
  _codeSelectedVersion = mtime;
  _codeSelectedClass = null;
  renderCodeVersionList();

  const versionMeta = _codeVersions.find((v) => v.mtime === mtime);
  const url = _versionClassesUrl(versionMeta);
  const data = await fetch(url).then((r) => r.json());
  _codeClasses = data;
  renderCodeClassList();

  // Auto-select: prefer the changed class(es) for this commit, else keep current, else first
  const trigger = versionMeta;
  if (trigger?.class_names?.length) {
    const preferred = trigger.class_names.includes(_codeSelectedClass) ? _codeSelectedClass : trigger.class_names[0];
    const match = _codeClasses.find((c) => c.class_name === preferred);
    if (match) { selectCodeClass(match.class_name, match.source_hash, { silent: true }); return; }
  }
  if (_codeClasses.length) selectCodeClass(_codeClasses[0].class_name, _codeClasses[0].source_hash, { silent: true });
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
    item.onclick = () => selectCodeClass(item.dataset.cls, item.dataset.hash);
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
      const target = _codeSelectedVersion ?? (_codeVersions[0]?.mtime ?? null);
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
    const isActive = v.mtime === _codeSelectedVersion;
    return `
      <div class="code-version-item ${isActive ? 'active' : ''}" data-mtime="${esc(v.mtime)}">
        ${esc(v.label)}
      </div>`;
  }).join('');

  el.querySelectorAll('.code-version-item').forEach((item) => {
    item.onclick = () => selectCodeVersionForClass(item.dataset.mtime, className);
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
    _codeSelectedVersion = newest.mtime;
    renderCodeVersionsForClass(cls.class_name);
    const classes = await fetch(_versionClassesUrl(newest)).then((r) => r.json());
    const match = classes.find((c) => c.class_name === cls.class_name);
    if (match) await selectCodeClass(match.class_name, match.source_hash, { silent: true });
  } else {
    // Fallback: no version groups yet, show current source directly
    await selectCodeClass(cls.class_name, cls.source_hash, { silent: true });
  }
}

async function selectCodeVersionForClass(mtime, className) {
  pushHistory(_viewMode, _codeState());
  _codeSelectedVersion = mtime;
  renderCodeVersionsForClass(className);

  try {
    const versionMeta = _codeVersions.find((v) => v.mtime === mtime);
    const classes = await fetch(_versionClassesUrl(versionMeta)).then((r) => r.json());
    const match = classes.find((c) => c.class_name === className);
    if (match) await selectCodeClass(match.class_name, match.source_hash, { silent: true });
  } catch {
    toast('Could not load version');
  }
}

export async function selectCodeClass(className, sourceHash, { silent = false } = {}) {
  if (!silent) pushHistory(_viewMode, _codeState());
  _codeSelectedClass = className;
  renderCodeClassList();

  // Update source pane header
  const titleEl = $('#code-source-title');
  if (titleEl) titleEl.textContent = className;
  const findBtn = $('#btn-find-in-entries');
  if (findBtn) findBtn.style.display = '';

  try {
    const data = await fetch(`/api/code/snapshot?source_hash=${encodeURIComponent(sourceHash)}`).then((r) => r.json());
    const panel = $('#code-source-panel');
    if (panel) {
      panel.innerHTML = data.html;
      bindCodeSourceLinks();
    }
  } catch {
    toast('Source unavailable');
  }

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
  // find the version mtime that was current when the entry was computed.
  // Fall back to the most recent commit if no depHash.
  let targetVersion = _codeVersions[0]?.mtime ?? null;
  let resolvedSourceHash = null;
  if (depHash) {
    try {
      const res = await fetch(
        `/api/code/resolve-dep-hash?dep_hash=${encodeURIComponent(depHash)}&class_name=${encodeURIComponent(className)}`
      ).then((r) => r.ok ? r.json() : null);
      if (res) {
        targetVersion = res.version_mtime;
        resolvedSourceHash = res.source_hash;
      }
    } catch { /* fall through to 'now' */ }
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

export function showView(view, { pushNav = false } = {}) {
  if (pushNav && view !== _topView) {
    pushHistory(_viewMode, _topView === 'code' ? _codeState() : null);
  }
  setTopView(view);
  const toEntries = view === 'entries';
  document.querySelector('.view-entries').style.display = toEntries ? '' : 'none';
  document.querySelector('.view-code').style.display    = toEntries ? 'none' : '';
  document.getElementById('entries-toolbar').style.display       = toEntries ? '' : 'none';
  document.getElementById('entries-toolbar-right').style.display = toEntries ? '' : 'none';
  document.getElementById('code-toolbar').style.display          = toEntries ? 'none' : '';
  document.getElementById('code-toolbar-right').style.display    = toEntries ? 'none' : '';
  $$('.view-tab').forEach((t) => t.classList.toggle('active', t.dataset.view === view));
  localStorage.setItem('view_mode_top', view);
  updateNavBtns();
}


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
  const versionMeta = _codeVersions.find((v) => v.mtime === _codeSelectedVersion);
  state.version_filter = versionMeta?.mtime ?? null;
  scheduleSelectFirst();
  showView('entries', { pushNav: true });
  toggleClass(_codeSelectedClass, { navigate: true });
});
