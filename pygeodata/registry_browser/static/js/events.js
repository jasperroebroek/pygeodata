/**
 * events.js
 *
 * All event wiring — mode tabs, kind tabs, logic mode, sidebar toggles,
 * resize handle, arrow keys, back/forward navigation.
 */

import { $$, toast } from './utils.js';
import { state } from './state.js';
import { navigateBack, navigateForward, pushHistory, _viewMode, _topView, updateNavBtns } from './nav.js';
import { loadEntries, loadEntriesOnly, applyViewMode, showDiagnostics, selectEntry, loadDetail } from './entries.js';
import { renderFilterRows } from './filters.js';
import { renderClassList, _multiSelectEnabled, setMultiSelectEnabled, _showEmptyClasses, setShowEmptyClasses } from './class-list.js';
import { applyTableSelection, _visibleEntryIds, _tableRows, _tableEntryCount, _appendTableRows, PAGE_ENTRIES } from './table.js';
import { lastDashboard } from './utils.js';
import { postRebuild } from './api.js';
import { runCleanCache } from './entries.js';

// ---------------------------------------------------------------------------
// Mode tabs (Compact / Detailed)
// ---------------------------------------------------------------------------

$$("#entries-screen-tabs .kind-tab").forEach((tab) => {
  tab.onclick = () => applyViewMode(tab.dataset.mode);
});

// ---------------------------------------------------------------------------
// Sidebar collapsible sections — accordion: at most one open at a time
// ---------------------------------------------------------------------------

$$(".sb-section-hd[data-target]").forEach((hd) => {
  hd.onclick = () => {
    const section = document.getElementById(hd.dataset.target);
    const isOpen = section.classList.contains("open");
    // Close all collapsible sections
    $$(".sb-section.collapsible").forEach((s) => s.classList.remove("open"));
    // If it wasn't open, open it
    if (!isOpen) section.classList.add("open");
  };
});

// ---------------------------------------------------------------------------
// Kind tabs
// ---------------------------------------------------------------------------

$$("#kind-tabs .kind-tab").forEach((tab) => {
  tab.onclick = () => {
    $$("#kind-tabs .kind-tab").forEach((t) =>
      t.classList.toggle("active", t === tab)
    );
    state.kind_filter = tab.dataset.kind;
    loadEntries();
  };
});

// ---------------------------------------------------------------------------
// Logic mode
// ---------------------------------------------------------------------------

$$("#logic-seg .kind-tab").forEach((btn) => {
  btn.onclick = () => {
    $$("#logic-seg .kind-tab").forEach((b) =>
      b.classList.toggle("active", b === btn)
    );
    state.logic_mode = btn.dataset.val;
    loadEntries();
  };
});

// ---------------------------------------------------------------------------
// Display mode (Selected params / All params — only active in Detailed view)
// ---------------------------------------------------------------------------

$$("#display-seg .kind-tab").forEach((btn) => {
  btn.onclick = () => {
    if (_viewMode === "compact") return;
    $$("#display-seg .kind-tab").forEach((b) =>
      b.classList.toggle("active", b === btn)
    );
    state.row_display = btn.dataset.val;
    loadEntries();
  };
});

// Initialise display state to "all" (Entries only is gone)
state.row_display = "all";

// ---------------------------------------------------------------------------
// Add filter
// ---------------------------------------------------------------------------

document.getElementById("add-filter").onclick = () => {
  state.filters.push({ target: "all", operator: "contains", value: "" });
  renderFilterRows();
  // focus the new row's value input
  const rows = document.getElementById("filter-rows").querySelectorAll(".fr-v");
  rows[rows.length - 1]?.focus();
};

// ---------------------------------------------------------------------------
// Diagnostics button
// ---------------------------------------------------------------------------

document.getElementById("btn-diag").onclick = showDiagnostics;

// ---------------------------------------------------------------------------
// Reload from disk
// ---------------------------------------------------------------------------

document.getElementById("btn-reload").onclick = async () => {
  try {
    await postRebuild();
    loadEntries();  // will poll until ready
  } catch (e) {
    toast(`Reload failed: ${e}`);
  }
};

// ---------------------------------------------------------------------------
// Clean cache — btn-clean-run delegated via #modal-body in detail.js
// ---------------------------------------------------------------------------

document.getElementById("modal-body").addEventListener("click", (e) => {
  if (e.target.id === "btn-clean-run") runCleanCache();
});

// ---------------------------------------------------------------------------
// Show/hide zero-entry classes toggle
// ---------------------------------------------------------------------------

document.getElementById("btn-show-empty").onclick = (e) => {
  e.stopPropagation();
  const next = !_showEmptyClasses;
  setShowEmptyClasses(next);
  document.getElementById("btn-show-empty").classList.toggle("active", next);
  if (lastDashboard) renderClassList(lastDashboard.class_cards ?? []);
};

// ---------------------------------------------------------------------------
// Hide stale toggle
// ---------------------------------------------------------------------------

document.getElementById("btn-hide-stale").classList.toggle("active", state.hide_stale);
document.getElementById("btn-hide-stale").onclick = (e) => {
  e.stopPropagation();
  state.hide_stale = !state.hide_stale;
  localStorage.setItem("hide_stale", state.hide_stale);
  document.getElementById("btn-hide-stale").classList.toggle("active", state.hide_stale);
  loadEntries();
};

// ---------------------------------------------------------------------------
// Multi-select toggle
// ---------------------------------------------------------------------------

document.getElementById("btn-multi-select").onclick = (e) => {
  e.stopPropagation();
  const next = !_multiSelectEnabled;
  setMultiSelectEnabled(next);
  document.getElementById("btn-multi-select").classList.toggle("active", next);
  // Collapsing to single-select: keep only the last selected class
  if (!next && state.selected_classes.length > 1) {
    pushHistory(_viewMode);
    const last = state.selected_classes[state.selected_classes.length - 1];
    state.selected_classes = [last];
    state.selected_entry = null;
    loadEntries();
  }
};

// ---------------------------------------------------------------------------
// Back / forward navigation
// ---------------------------------------------------------------------------

export async function _syncUIAfterRestore(snap) {
  const targetTopView = snap.code_version != null ? 'code' : (snap.view_mode === 'code' ? 'code' : 'entries');

  if (targetTopView === 'code') {
    _showView('code');
    if (!_codeLoaded()) await _loadCodeView();
    // Restore Code view selection without overwriting history again
    const version = snap.code_version ?? 'now';
    if (version !== _codeSelectedVersion()) {
      await _selectCodeVersion(version, { silent: true });
    }
    if (snap.code_class_name) {
      const match = _codeClasses().find((c) => c.class_name === snap.code_class_name);
      if (match) await _selectCodeClass(match.class_name, match.source_hash, { silent: true });
    }
    updateNavBtns();
    return;
  }

  _showView('entries');
  if (snap.view_mode && snap.view_mode !== _viewMode) applyViewMode(snap.view_mode, false);
  // Sync kind tabs
  $$("#kind-tabs .kind-tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.kind === state.kind_filter)
  );
  // Sync logic mode segment
  $$("#logic-seg .kind-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.val === state.logic_mode)
  );
  // Sync display mode segment
  $$("#display-seg .kind-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.val === state.row_display)
  );
  loadEntries();
}


// ---------------------------------------------------------------------------
// Clear all filters
// ---------------------------------------------------------------------------

document.getElementById("btn-clear-all").onclick = () => {
  pushHistory(_viewMode);
  state.selected_classes = [];
  state.spec_filters = { crs: [], resolution: [], bounds: [] };
  state.version_filter = null;
  state.filters = [{ target: "all", operator: "contains", value: "" }];
  loadEntries();
};

// ---------------------------------------------------------------------------
// Detail pane resize
// ---------------------------------------------------------------------------

(function () {
  const handle = document.getElementById("resize-handle");
  const pane   = document.getElementById("detail-pane");
  const STORE_KEY = "registry.detail-w";

  const saved = localStorage.getItem(STORE_KEY);
  if (saved) pane.style.flex = `0 0 ${saved}px`;

  let startX, startW;

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    startW = pane.getBoundingClientRect().width;
    handle.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  document.addEventListener("mousemove", (e) => {
    if (!handle.classList.contains("dragging")) return;
    const delta = startX - e.clientX;   // drag left = wider
    const newW  = Math.max(200, Math.min(window.innerWidth * 0.8, startW + delta));
    pane.style.flex = `0 0 ${newW}px`;
  });

  document.addEventListener("mouseup", () => {
    if (!handle.classList.contains("dragging")) return;
    handle.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem(STORE_KEY, pane.getBoundingClientRect().width);
  });
})();

// ---------------------------------------------------------------------------
// Incremental scroll: append more table rows when near the bottom
// ---------------------------------------------------------------------------

document.getElementById("table-scroll-wrap").addEventListener("scroll", () => {
  if (_viewMode !== "detailed" || _tableEntryCount === 0) return;
  const el = document.getElementById("table-scroll-wrap");
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 200) {
    _appendTableRows(PAGE_ENTRIES);
  }
}, { passive: true });

// ---------------------------------------------------------------------------
// ⌘[ / ⌘] — back / forward navigation
// ---------------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (!e.metaKey) return;
  if (e.key === "[") {
    e.preventDefault();
    const snap = navigateBack(_viewMode, _topView === 'code' ? _getCodeState() : null);
    if (snap) _syncUIAfterRestore(snap);
  } else if (e.key === "]") {
    e.preventDefault();
    const snap = navigateForward(_viewMode, _topView === 'code' ? _getCodeState() : null);
    if (snap) _syncUIAfterRestore(snap);
  }
});

// ---------------------------------------------------------------------------
// Arrow key navigation
// ---------------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();

  if (_topView === "code") {
    const isByClass = _codeBrowseMode() === 'class';
    const codeClasses = isByClass ? _codeAllClasses() : _codeClasses();
    if (!codeClasses.length) return;
    const cur = codeClasses.findIndex((c) => c.class_name === _codeSelectedClass());
    let next;
    if (e.key === "ArrowDown") next = cur === -1 ? 0 : Math.min(cur + 1, codeClasses.length - 1);
    else                       next = cur === -1 ? codeClasses.length - 1 : Math.max(cur - 1, 0);
    const c = codeClasses[next];
    if (isByClass) _selectCodeClassFirst(c);
    else           _selectCodeClass(c.class_name, c.source_hash);
    return;
  }

  if (!_visibleEntryIds.length) return;

  const ids = _visibleEntryIds;
  const cur = ids.indexOf(state.selected_entry);
  let next;
  if (e.key === "ArrowDown") next = cur === -1 ? 0 : Math.min(cur + 1, ids.length - 1);
  else                       next = cur === -1 ? ids.length - 1 : Math.max(cur - 1, 0);

  state.selected_entry = ids[next];
  applyTableSelection();

  // Scroll the active item into view — pill list in compact, table row in detailed
  if (_viewMode === "compact") {
    const activePill = document.querySelector(`#entry-pill-list .entry-pill[data-entry="${CSS.escape(state.selected_entry)}"]`);
    activePill?.scrollIntoView({ block: "nearest" });
  } else {
    // Ensure the target entry is rendered before scrolling to it
    let activeRow = document.querySelector(`#table-body tr.row-entry[data-entry="${CSS.escape(state.selected_entry)}"]`);
    if (!activeRow) {
      const targetIdx = _tableRows.findIndex(
        (r) => r.row_type === "header" && r.record_id === state.selected_entry
      );
      if (targetIdx !== -1) {
        // Count how many header rows precede it and render up to that point
        const headersNeeded = _tableRows.slice(0, targetIdx + 1).filter((r) => r.row_type === "header").length;
        while (_tableEntryCount < headersNeeded) _appendTableRows(PAGE_ENTRIES);
        applyTableSelection();
        activeRow = document.querySelector(`#table-body tr.row-entry[data-entry="${CSS.escape(state.selected_entry)}"]`);
      }
    }
    activeRow?.scrollIntoView({ block: "nearest" });
  }

  loadDetail();
});


// ---------------------------------------------------------------------------
// Lazy references to code-view state (injected from code-view.js)
// ---------------------------------------------------------------------------

let _getCodeState    = () => null;
let _codeLoaded      = () => false;
let _loadCodeView        = async () => {};
let _codeSelectedVersion = () => null;
let _selectCodeVersion   = async () => {};
let _codeClasses         = () => [];
let _codeAllClasses      = () => [];
let _codeBrowseMode      = () => 'version';
let _codeSelectedClass   = () => null;
let _selectCodeClass     = async () => {};
let _selectCodeClassFirst = async () => {};
let _showView            = () => {};

export function setEventsCodeView({
  getCodeState,
  codeLoaded,
  loadCodeView,
  codeSelectedVersion,
  selectCodeVersion,
  codeClasses,
  codeAllClasses,
  codeBrowseMode,
  codeSelectedClass,
  selectCodeClass,
  selectCodeClassFirst,
  showView,
}) {
  _getCodeState         = getCodeState;
  _codeLoaded           = codeLoaded;
  _loadCodeView         = loadCodeView;
  _codeSelectedVersion  = codeSelectedVersion;
  _selectCodeVersion    = selectCodeVersion;
  _codeClasses          = codeClasses;
  _codeAllClasses       = codeAllClasses;
  _codeBrowseMode       = codeBrowseMode;
  _codeSelectedClass    = codeSelectedClass;
  _selectCodeClass      = selectCodeClass;
  _selectCodeClassFirst = selectCodeClassFirst;
  _showView             = showView;
}
