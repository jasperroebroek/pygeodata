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
let _codeDiffMode        = false;  // true while showing a unified diff in the source pane
let _diffExpand          = localStorage.getItem('diff_expand')    ?? 'hunks'; // 'hunks' | 'full'

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
  const mtime = versionMeta?.mtime ?? 'now';
  return `/api/code/version-classes?mtime=${encodeURIComponent(mtime)}`;
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

  // If this version has a defined set of changed classes, show a diff overview
  // instead of auto-selecting the first class.
  const changedNames = versionMeta?.class_names ?? [];
  if (changedNames.length) {
    await _showVersionChangeSummary(versionMeta, changedNames);
    return;
  }

  // Fallback (Initial group or no changes listed): select first class
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
    item.onclick = () => {
      if (item.dataset.cls === _codeSelectedClass) {
        // Deselect — return to version change summary if this version has changed classes
        const versionMeta = _codeVersions.find((v) => v.mtime === _codeSelectedVersion);
        const changedNames = versionMeta?.class_names ?? [];
        if (changedNames.length) {
          _showVersionChangeSummary(versionMeta, changedNames);
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

// ---------------------------------------------------------------------------
// Diff renderer — inline / side-by-side / expand
// ---------------------------------------------------------------------------

let _diffViewMode = localStorage.getItem('diff_view_mode') ?? 'inline'; // 'inline' | 'split'

/**
 * Parse a unified diff string into a structured array of hunks.
 * Each hunk: { startA, startB, lines: [{type, text}] }
 * type: 'ctx' | 'add' | 'del' | 'hdr'
 */
function _parseDiff(unifiedDiff) {
  const hunks = [];
  let current = null;
  for (const raw of unifiedDiff.split('\n')) {
    if (raw.startsWith('--- ') || raw.startsWith('+++ ')) {
      // file headers — skip
    } else if (raw.startsWith('@@')) {
      const m = raw.match(/@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
      current = { startA: m ? +m[1] : 1, startB: m ? +m[2] : 1, header: raw, lines: [] };
      hunks.push(current);
    } else if (current) {
      if (raw.startsWith('+'))      current.lines.push({ type: 'add', text: raw.slice(1) });
      else if (raw.startsWith('-')) current.lines.push({ type: 'del', text: raw.slice(1) });
      else if (raw !== '\\')        current.lines.push({ type: 'ctx', text: raw.slice(1) });
    }
  }
  return hunks;
}

/**
 * Compute character-level word diff between two strings.
 * Returns [{type:'eq'|'ins'|'del', text}]
 */
function _wordDiff(textA, textB) {
  // Tokenise by word boundaries so highlighting is word-granular
  const tokenise = (s) => s.match(/\w+|\W/g) ?? [];
  const a = tokenise(textA);
  const b = tokenise(textB);

  // Simple LCS-based diff (Myers-like, O(n·m) but files are small)
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--)
    for (let j = n - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);

  const result = [];
  let i = 0, j = 0;
  while (i < m || j < n) {
    if (i < m && j < n && a[i] === b[j]) {
      result.push({ type: 'eq', text: a[i++] }); j++;
    } else if (j < n && (i >= m || dp[i][j + 1] >= dp[i + 1][j])) {
      result.push({ type: 'ins', text: b[j++] });
    } else {
      result.push({ type: 'del', text: a[i++] });
    }
  }
  return result;
}

function _inlineWordHtml(tokens) {
  return tokens.map((t) => {
    if (t.type === 'eq')  return esc(t.text);
    if (t.type === 'ins') return `<span class="diff-word-add">${esc(t.text)}</span>`;
    return `<span class="diff-word-del">${esc(t.text)}</span>`;
  }).join('');
}

/** Pair up consecutive del/add lines for word-diff highlighting. */
function _pairHunkLines(lines) {
  // Returns array of {type, textA?, textB?, paired}
  const result = [];
  let i = 0;
  while (i < lines.length) {
    if (lines[i].type === 'del') {
      // Collect run of dels then run of adds
      const dels = [], adds = [];
      while (i < lines.length && lines[i].type === 'del') dels.push(lines[i++]);
      while (i < lines.length && lines[i].type === 'add') adds.push(lines[i++]);
      const pairs = Math.min(dels.length, adds.length);
      for (let k = 0; k < pairs; k++)
        result.push({ type: 'change', textA: dels[k].text, textB: adds[k].text });
      for (let k = pairs; k < dels.length; k++)
        result.push({ type: 'del', text: dels[k].text });
      for (let k = pairs; k < adds.length; k++)
        result.push({ type: 'add', text: adds[k].text });
    } else {
      result.push(lines[i++]);
    }
  }
  return result;
}

function _renderInlineHunk(hunk, lineNoA, lineNoB) {
  const paired = _pairHunkLines(hunk.lines);
  let la = lineNoA, lb = lineNoB;
  const rows = [];
  for (const p of paired) {
    if (p.type === 'ctx') {
      rows.push(`<tr class="diff-ctx"><td class="diff-ln">${la++}</td><td class="diff-ln">${lb++}</td><td class="diff-sign"> </td><td class="diff-code">${esc(p.text)}</td></tr>`);
    } else if (p.type === 'add') {
      rows.push(`<tr class="diff-add"><td class="diff-ln"></td><td class="diff-ln">${lb++}</td><td class="diff-sign">+</td><td class="diff-code">${esc(p.text)}</td></tr>`);
    } else if (p.type === 'del') {
      rows.push(`<tr class="diff-del"><td class="diff-ln">${la++}</td><td class="diff-ln"></td><td class="diff-sign">−</td><td class="diff-code">${esc(p.text)}</td></tr>`);
    } else {
      // paired change — word diff
      const tokens = _wordDiff(p.textA, p.textB);
      const delHtml = tokens.filter(t => t.type !== 'ins').map(t => t.type === 'del' ? `<span class="diff-word-del">${esc(t.text)}</span>` : esc(t.text)).join('');
      const addHtml = tokens.filter(t => t.type !== 'del').map(t => t.type === 'ins' ? `<span class="diff-word-add">${esc(t.text)}</span>` : esc(t.text)).join('');
      rows.push(`<tr class="diff-del"><td class="diff-ln">${la++}</td><td class="diff-ln"></td><td class="diff-sign">−</td><td class="diff-code">${delHtml}</td></tr>`);
      rows.push(`<tr class="diff-add"><td class="diff-ln"></td><td class="diff-ln">${lb++}</td><td class="diff-sign">+</td><td class="diff-code">${addHtml}</td></tr>`);
    }
  }
  return { html: rows.join(''), nextA: la, nextB: lb };
}

function _renderSplitHunk(hunk, lineNoA, lineNoB) {
  const paired = _pairHunkLines(hunk.lines);
  let la = lineNoA, lb = lineNoB;
  const rows = [];
  for (const p of paired) {
    if (p.type === 'ctx') {
      rows.push(`<tr class="diff-ctx"><td class="diff-ln">${la++}</td><td class="diff-code">${esc(p.text)}</td><td class="diff-ln">${lb++}</td><td class="diff-code">${esc(p.text)}</td></tr>`);
    } else if (p.type === 'add') {
      rows.push(`<tr><td class="diff-ln diff-empty-side"></td><td class="diff-code diff-empty-side"></td><td class="diff-ln diff-add">${lb++}</td><td class="diff-code diff-add">${esc(p.text)}</td></tr>`);
    } else if (p.type === 'del') {
      rows.push(`<tr><td class="diff-ln diff-del">${la++}</td><td class="diff-code diff-del">${esc(p.text)}</td><td class="diff-ln diff-empty-side"></td><td class="diff-code diff-empty-side"></td></tr>`);
    } else {
      // paired change — word diff side by side
      const tokens = _wordDiff(p.textA, p.textB);
      const delHtml = tokens.filter(t => t.type !== 'ins').map(t => t.type === 'del' ? `<span class="diff-word-del">${esc(t.text)}</span>` : esc(t.text)).join('');
      const addHtml = tokens.filter(t => t.type !== 'del').map(t => t.type === 'ins' ? `<span class="diff-word-add">${esc(t.text)}</span>` : esc(t.text)).join('');
      rows.push(`<tr><td class="diff-ln diff-del">${la++}</td><td class="diff-code diff-del">${delHtml}</td><td class="diff-ln diff-add">${lb++}</td><td class="diff-code diff-add">${addHtml}</td></tr>`);
    }
  }
  return { html: rows.join(''), nextA: la, nextB: lb };
}

/**
 * Render a full diff view into a DOM element.
 * @param {Element} container
 * @param {{diff: string, full_a?: string, full_b?: string}} diffData
 */
function _renderDiffInto(container, diffData) {
  const { diff, full_a, full_b } = diffData;

  if (!diff) {
    container.innerHTML = '<div class="diff-empty-msg">Files are identical — no changes.</div>';
    return;
  }

  const hunks    = _parseDiff(diff);
  const isExpanded = _diffExpand === 'full' && !!(full_a && full_b);
  const isSplit    = _diffViewMode === 'split';

  const splitCls  = isSplit ? 'diff-table--split' : 'diff-table--inline';

  let tableHtml = '';
  if (isExpanded) {
    tableHtml = _renderFullFileTable(full_a, full_b, hunks, isSplit);
  } else {
    const rows = [];
    for (const hunk of hunks) {
      rows.push(`<tr class="diff-hunk-hdr"><td colspan="4" class="diff-hunk-hdr-cell">${esc(hunk.header)}</td></tr>`);
      const render = isSplit ? _renderSplitHunk : _renderInlineHunk;
      const { html } = render(hunk, hunk.startA, hunk.startB);
      rows.push(html);
    }
    const colgroup = isSplit
      ? `<colgroup><col style="width:44px"><col><col style="width:44px"><col></colgroup>`
      : '';
    tableHtml = `<table class="diff-table ${splitCls}">${colgroup}${rows.join('')}</table>`;
  }

  container.innerHTML = tableHtml;
}

function _renderFullFileTable(fullA, fullB, hunks, isSplit) {
  // Build the full-file view by replaying hunk rendering, inserting plain context rows
  // for all unchanged lines between hunks. This reuses the same hunk renderers so
  // alignment is guaranteed correct in both inline and split modes.
  const linesA = fullA.split('\n');
  const linesB = fullB.split('\n');
  if (linesA[linesA.length - 1] === '') linesA.pop();
  if (linesB[linesB.length - 1] === '') linesB.pop();

  const render = isSplit ? _renderSplitHunk : _renderInlineHunk;
  const rows = [];

  let curA = 1, curB = 1; // next unrendered line (1-based)

  function ctxRow(la, lb) {
    if (isSplit) {
      return `<tr class="diff-ctx"><td class="diff-ln">${la}</td><td class="diff-code">${esc(linesA[la - 1] ?? '')}</td><td class="diff-ln">${lb}</td><td class="diff-code">${esc(linesB[lb - 1] ?? '')}</td></tr>`;
    }
    return `<tr class="diff-ctx"><td class="diff-ln">${la}</td><td class="diff-ln">${lb}</td><td class="diff-sign"> </td><td class="diff-code">${esc(linesB[lb - 1] ?? '')}</td></tr>`;
  }

  for (const hunk of hunks) {
    // Emit unchanged context lines before this hunk
    while (curA < hunk.startA && curB < hunk.startB) {
      rows.push(ctxRow(curA, curB));
      curA++; curB++;
    }
    // Render the hunk
    const { html, nextA, nextB } = render(hunk, hunk.startA, hunk.startB);
    rows.push(html);
    curA = nextA;
    curB = nextB;
  }

  // Trailing context lines after the last hunk
  while (curA <= linesA.length && curB <= linesB.length) {
    rows.push(ctxRow(curA, curB));
    curA++; curB++;
  }

  const splitCls = isSplit ? 'diff-table--split' : 'diff-table--inline';
  return `<table class="diff-table ${splitCls}">${rows.join('')}</table>`;
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
    // In summary mode (_currentDiffData is null), enable if any rendered body has full_a
    const summaryHasFull = !_currentDiffData && _codeDiffMode
      && [...$$('#code-source-panel .tree-diff-body[data-diff]')].some((b) => {
          try { return !!JSON.parse(b.dataset.diff).full_a; } catch { return false; }
        });
    const canExpand = _codeDiffMode && (!!(_currentDiffData?.full_a) || summaryHasFull);
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
  const selectedIdx      = _codeVersions.findIndex((v) => v.mtime === _codeSelectedVersion);
  const prevVersion      = selectedIdx >= 0 ? _codeVersions[selectedIdx + 1] ?? null : null;
  const isViewingCurrent = currentVersion && _codeSelectedVersion === currentVersion.mtime;

  const opts = [];

  if (prevVersion) {
    try {
      const classes = await fetch(_versionClassesUrl(prevVersion)).then((r) => r.json());
      const match   = classes.find((c) => c.class_name === _codeSelectedClass);
      if (match && match.source_hash !== selectedEntry.source_hash) {
        opts.push({ label: 'vs. prev', hashA: selectedEntry.source_hash, hashB: match.source_hash });
      }
    } catch { /* skip */ }
  }

  if (!isViewingCurrent && currentVersion) {
    const currentEntry = _codeAllClasses.find((c) => c.class_name === _codeSelectedClass)
      ?? _codeClasses.find((c) => c.class_name === _codeSelectedClass);
    if (currentEntry && currentEntry.source_hash !== selectedEntry.source_hash) {
      opts.push({ label: 'vs. current', hashA: selectedEntry.source_hash, hashB: currentEntry.source_hash });
    }
  }

  return opts;
}

// Render the diff-target selector (small pills above the diff table)
function _renderDiffTargetBar(panel) {
  const existing = panel.querySelector('.diff-target-bar');
  if (existing) existing.remove();
  if (_diffOptions.length <= 1) return; // only one option — no selector needed

  const bar = document.createElement('div');
  bar.className = 'diff-target-bar';
  bar.innerHTML = _diffOptions.map((o, i) =>
    `<button class="kind-tab${i === _diffOptionIdx ? ' active' : ''}" data-idx="${i}">${esc(o.label)}</button>`
  ).join('');
  bar.querySelectorAll('[data-idx]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      _diffOptionIdx = +btn.dataset.idx;
      await _loadAndRenderDiff(_diffOptions[_diffOptionIdx]);
    });
  });
  panel.prepend(bar);
}

async function _loadAndRenderDiff(opt) {
  try {
    const data = await fetch(
      `/api/code/diff?hash_a=${encodeURIComponent(opt.hashA)}&hash_b=${encodeURIComponent(opt.hashB)}&full=1`
    ).then((r) => r.json());
    _currentDiffData = data;
    const panel = $('#code-source-panel');
    if (panel) {
      _renderDiffInto(panel, data);
      _renderDiffTargetBar(panel);
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

export async function showDiff(hashA, hashB) {
  _diffOptions = [{ label: '', hashA, hashB }];
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

export async function showWhatChanged(recordId) {
  try {
    const data = await fetch(`/api/code/tree-diff?record_id=${encodeURIComponent(recordId)}`).then((r) => r.json());
    showView('code');
    if (!_codeLoaded) await loadCodeView();
    if (data.error === 'no_snapshot') {
      const panel = $('#code-source-panel');
      if (panel) panel.innerHTML = `<div class="diff-no-snapshot">${esc(data.message ?? 'Snapshot not available for this entry')}</div>`;
      _codeDiffMode = false;
      _summaryMode = false;
      _currentDiffData = null;
      _syncDiffSegments();
      return;
    }
    _codeDiffMode = true;
    _summaryMode = true;
    _diffAvailable = true;
    _currentDiffData = null;
    _codeSelectedClass = null;
    _syncDiffSegments();
    renderTreeDiffResult(data.changes);
  } catch { toast('Could not load tree diff'); }
}

async function _showVersionChangeSummary(versionMeta, changedNames) {
  const panel = $('#code-source-panel');
  if (!panel) return;

  _codeDiffMode = true;
  _summaryMode = true;
  _diffAvailable = true;
  _currentDiffData = null;
  _codeSelectedClass = null;
  renderCodeClassList();
  _syncDiffSegments();

  // Find previous version to diff against
  const idx = _codeVersions.findIndex((v) => v.mtime === versionMeta.mtime);
  const prevVersion = idx >= 0 ? _codeVersions[idx + 1] ?? null : null;

  // Build a hash lookup for both current and previous version
  let prevHashMap = {};
  if (prevVersion) {
    try {
      const prevClasses = await fetch(_versionClassesUrl(prevVersion)).then((r) => r.json());
      for (const c of prevClasses) prevHashMap[c.class_name] = c.source_hash;
    } catch { /* skip */ }
  }

  const changes = changedNames.map((cn) => {
    const current = _codeClasses.find((c) => c.class_name === cn);
    const hashA = prevHashMap[cn] ?? null;
    const hashB = current?.source_hash ?? null;
    const status = !hashA ? 'added' : !hashB ? 'removed' : hashA === hashB ? 'unchanged' : 'changed';
    return { class_name: cn, status, hashA, hashB, diff: null };
  }).filter((c) => c.status !== 'unchanged');

  const wrap = document.createElement('div');
  wrap.className = 'tree-diff-container';
  wrap.innerHTML = changes.map((c) => {
    const open = c.status === 'changed' ? ' open' : '';
    const canExpand = (c.status === 'changed' && c.hashA && c.hashB)
                   || (c.status === 'added'   && c.hashB)
                   || (c.status === 'removed' && c.hashA);
    return `<div class="tree-diff-class ${esc(c.status)}${open}" data-class="${esc(c.class_name)}"
                 data-hash-a="${esc(c.hashA ?? '')}" data-hash-b="${esc(c.hashB ?? '')}">
      <div class="tree-diff-header">
        <span class="tree-diff-status">${esc(c.status)}</span>
        <span class="tree-diff-name">${esc(c.class_name)}</span>
        ${canExpand ? '<span class="tree-diff-expand">▶</span>' : ''}
      </div>
      ${canExpand ? `<div class="tree-diff-body"></div>` : ''}
    </div>`;
  }).join('');
  panel.innerHTML = '';
  panel.appendChild(wrap);

  wrap.querySelectorAll('.tree-diff-header').forEach((hdr) => {
    const card = hdr.parentElement;
    const body = card.querySelector('.tree-diff-body');
    const hashA = card.dataset.hashA;
    const hashB = card.dataset.hashB;
    if (!body) return;

    const _maybeRender = async () => {
      if (body.dataset.rendered) return;
      body.dataset.rendered = '1';
      body.innerHTML = '<div class="diff-loading">Loading…</div>';
      try {
        if (hashA && hashB) {
          const data = await fetch(
            `/api/code/diff?hash_a=${encodeURIComponent(hashA)}&hash_b=${encodeURIComponent(hashB)}&full=1`
          ).then((r) => r.json());
          body.dataset.diff = JSON.stringify(data);
          _renderDiffInto(body, data);
          _syncDiffSegments();
        } else {
          const sourceHash = hashA || hashB;
          const data = await fetch(
            `/api/code/snapshot?source_hash=${encodeURIComponent(sourceHash)}`
          ).then((r) => r.json());
          body.innerHTML = data.html ?? '<div class="diff-no-snapshot">Source unavailable</div>';
          bindCodeSourceLinks();
        }
      } catch { body.innerHTML = '<div class="diff-no-snapshot">Unavailable</div>'; }
    };

    hdr.onclick = () => {
      card.classList.toggle('open');
      if (card.classList.contains('open')) _maybeRender();
    };

    if (card.classList.contains('open')) _maybeRender();
  });
}


function renderTreeDiffResult(changes) {
  const panel = $('#code-source-panel');
  if (!panel) return;
  if (!changes?.length) {
    panel.innerHTML = '<div class="diff-no-snapshot">No dependency changes found.</div>';
    return;
  }
  const wrap = document.createElement('div');
  wrap.className = 'tree-diff-container';
  wrap.innerHTML = changes.map((c) => {
    const open = (c.status === 'changed' || c.status === 'removed') ? ' open' : '';
    const hasDiff = c.diff && c.status === 'changed';
    const hasSource = (c.status === 'added' || c.status === 'removed') && c.source_hash;
    const canExpand = hasDiff || hasSource;
    return `<div class="tree-diff-class ${esc(c.status)}${open}"
               data-class="${esc(c.class_name)}"
               data-source-hash="${esc(c.source_hash ?? '')}">
      <div class="tree-diff-header">
        <span class="tree-diff-status">${esc(c.status)}</span>
        <span class="tree-diff-name">${esc(c.class_name)}</span>
        ${canExpand ? '<span class="tree-diff-expand">▶</span>' : ''}
      </div>
      ${canExpand ? `<div class="tree-diff-body"></div>` : ''}
    </div>`;
  }).join('');
  panel.innerHTML = '';
  panel.appendChild(wrap);

  // Wire headers — click to expand/collapse, lazy-render on first open
  wrap.querySelectorAll('.tree-diff-header').forEach((hdr) => {
    const card = hdr.parentElement;
    const body = card.querySelector('.tree-diff-body');
    const className = card.dataset.class;
    const change = changes.find((c) => c.class_name === className);
    if (!body || !change) return;

    const _maybeRender = async () => {
      if (body.dataset.rendered) return;
      body.dataset.rendered = '1';
      if (change.diff) {
        const diffData = { diff: change.diff, full_a: change.full_a ?? null, full_b: change.full_b ?? null };
        body.dataset.diff = JSON.stringify(diffData);
        _renderDiffInto(body, diffData);
        _syncDiffSegments();
      } else if (change.source_hash) {
        body.innerHTML = '<div class="diff-loading">Loading…</div>';
        try {
          const data = await fetch(
            `/api/code/snapshot?source_hash=${encodeURIComponent(change.source_hash)}`
          ).then((r) => r.json());
          body.innerHTML = data.html ?? '<div class="diff-no-snapshot">Source unavailable</div>';
          bindCodeSourceLinks();
        } catch { body.innerHTML = '<div class="diff-no-snapshot">Source unavailable</div>'; }
      }
    };

    hdr.onclick = () => {
      card.classList.toggle('open');
      if (card.classList.contains('open')) _maybeRender();
    };

    if (card.classList.contains('open')) _maybeRender();
  });
}


export async function selectCodeClass(className, sourceHash, { silent = false } = {}) {
  if (!silent) pushHistory(_viewMode, _codeState());
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

  _diffAvailable = diffOpts.length > 0;

  if (data) {
    const panel = $('#code-source-panel');
    if (panel) {
      panel.innerHTML = data.html;
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

  let targetVersion = _codeVersions[0]?.mtime ?? 'now';
  try {
    const res = await fetch(
      `/api/code/source-hash-version?source_hash=${encodeURIComponent(sourceHash)}&class_name=${encodeURIComponent(className)}`
    ).then((r) => r.ok ? r.json() : null);
    if (res?.version_mtime) targetVersion = res.version_mtime;
  } catch { /* fall through to newest */ }

  if (targetVersion !== _codeSelectedVersion) {
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
  const versionMeta = _codeVersions.find((v) => v.mtime === _codeSelectedVersion);
  state.version_filter = versionMeta?.mtime ?? null;
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
