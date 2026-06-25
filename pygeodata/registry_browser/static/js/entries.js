/**
 * entries.js
 *
 * loadEntries, loadEntriesOnly, loading overlay, diagnostics modal, spec pills,
 * version select, selectEntry, loadDetail, applyViewMode.
 */

import { $, esc, toast, boundsLatLonText, setLastDashboard, lastDashboard, setLastVersionOptions } from './utils.js';
import { state, buildPayload } from './state.js';
import { _viewMode, _topView, setViewMode, pushHistory } from './nav.js';
import { renderFilterRows, setFilterLoaders } from './filters.js';
import { renderPills, setPillsLoaders, setPillsCodeState } from './pills.js';
import {
  renderTableHead, renderTableBody, applyTableSelection,
  _appendTableRows, PAGE_ENTRIES,
  setVisibleEntryIds, setTableActions,
} from './table.js';
import { renderClassList, toggleClass, setClassListLoaders, setClassListDashboard, setClassListCodeState, _multiSelectEnabled as _clMultiSelect } from './class-list.js';
import { renderDetail, renderEntryPills, setDetailActions, openModal as _openModal } from './detail.js';
import { fetchDashboard, postCleanCache, postCleanSource } from './api.js';


// ---------------------------------------------------------------------------
// Version select dropdown
// ---------------------------------------------------------------------------

function _renderVersionSelect(options) {
  const sel = $("#version-select");
  if (!sel) return;
  sel.innerHTML =
    `<option value="all">All snapshots</option>` +
    options.map((o) => `<option value="${esc(o.version_id)}">${esc(o.label)}</option>`).join("");
  // Restore selection if still valid
  if (state.version_filter && options.some((o) => o.version_id === state.version_filter)) {
    sel.value = state.version_filter;
  } else {
    sel.value = 'all';
    state.version_filter = null;
  }
}

$("#version-select")?.addEventListener("change", (e) => {
  pushHistory(_viewMode);
  const v = e.target.value;
  state.version_filter = v === 'all' ? null : v;
  loadEntries();
});


// ---------------------------------------------------------------------------
// Spec pill helpers
// ---------------------------------------------------------------------------

export function renderSpecPills(selector, options, selectedSet, labelFn = null) {
  const el = $(selector);
  if (!options?.length) {
    el.innerHTML = `<span style="color:var(--faint);font-size:11px">—</span>`;
    return;
  }
  el.innerHTML = options
    .map((v) => {
      const active = selectedSet.includes(v);
      const label = labelFn ? labelFn(v) : v;
      return `<button class="spec-pill ${active ? "active" : ""}" data-v="${esc(v)}" title="${esc(label)}">${esc(label)}</button>`;
    })
    .join("");
  el.querySelectorAll(".spec-pill").forEach((btn) => {
    btn.onclick = () => {
      const v = btn.dataset.v;
      const i = selectedSet.indexOf(v);
      if (i === -1) selectedSet.push(v);
      else selectedSet.splice(i, 1);
      renderSpecPills(selector, options, selectedSet);
      loadEntriesOnly();
    };
  });
}


// ---------------------------------------------------------------------------
// Loading overlay — polls /api/status until ready, then fires callback
// ---------------------------------------------------------------------------

let _loadingOverlayActive = false;
export let hasLiveClasses = false;

// phase: 'reload' (default) | 'reimport'
export function showLoadingOverlay(phase = 'reload') {
  if (_loadingOverlayActive) return;
  _loadingOverlayActive = true;
  $("#loading-bar-fill").style.width = "0%";
  $("#loading-label").textContent = phase === 'reimport' ? "Reimporting…" : "Loading entries…";
  $("#loading-sub").textContent = "";
  $("#loading-overlay").classList.remove("hidden");
  _pollUntilReady();
}

function hideLoadingOverlay() {
  _loadingOverlayActive = false;
  $("#loading-overlay").classList.add("hidden");
}

async function _pollUntilReady() {
  while (_loadingOverlayActive) {
    try {
      const status = await (await fetch("/api/status")).json();
      const p = status.progress ?? {};

      const reimportTotal = p.reimport_total ?? 0;
      const reimportDone  = p.reimport_done  ?? 0;
      const scanTotal = p.total ?? 0;
      const scanDone  = p.done  ?? 0;

      if (reimportTotal > 0 && reimportDone < reimportTotal) {
        // Phase 1: reimporting Python files
        const pct = Math.round((reimportDone / reimportTotal) * 100);
        $("#loading-bar-fill").style.width = `${pct}%`;
        $("#loading-label").textContent = "Reimporting…";
        $("#loading-sub").textContent = `${reimportDone} / ${reimportTotal} files`;
      } else if (scanTotal > 0) {
        // Phase 2: scanning entries
        const pct = Math.round((scanDone / scanTotal) * 100);
        $("#loading-bar-fill").style.width = `${pct}%`;
        $("#loading-label").textContent = "Loading entries…";
        $("#loading-sub").textContent = `${scanDone} / ${scanTotal}`;
      } else if (reimportTotal > 0) {
        // Reimport finished, scan not yet started
        $("#loading-bar-fill").style.width = "100%";
        $("#loading-label").textContent = "Reimporting…";
        $("#loading-sub").textContent = "Done — scanning entries…";
      }

      if (status.has_live_classes != null) hasLiveClasses = status.has_live_classes;

      if (status.ready) {
        hideLoadingOverlay();
        loadEntriesOnly();
        return;
      }
    } catch {
      // server not up yet — keep polling
    }
    await new Promise((r) => setTimeout(r, 300));
  }
}


// ---------------------------------------------------------------------------
// selectEntry / loadDetail
// ---------------------------------------------------------------------------

export function selectEntry(id, className = null) {
  pushHistory(_viewMode, _topView === 'code' ? _getCodeState() : null);

  if (id === state.selected_entry) {
    state.selected_entry = null;
    applyTableSelection();
    loadDetail();
    return;
  }

  state.selected_entry = id;

  // Only update class selection when navigating via a linked-entry (className hint provided).
  const ownerClass = className;

  if (ownerClass && !state.selected_classes.includes(ownerClass)) {
    if (state.kind_filter !== "all") {
      const card = (lastDashboard?.class_cards ?? []).find((c) => c.class_name === ownerClass);
      if (card && (card.object_type ?? "").toLowerCase() !== state.kind_filter) {
        state.kind_filter = "all";
        document.querySelectorAll("#kind-tabs .kind-tab").forEach((t) =>
          t.classList.toggle("active", t.dataset.kind === "all")
        );
      }
    }
    if (_clMultiSelect) {
      state.selected_classes.push(ownerClass);
    } else {
      state.selected_classes = [ownerClass];
    }
    loadEntries();
  } else {
    applyTableSelection();
    loadDetail();
  }
}

export async function loadDetail() {
  const data = await fetchDashboard({ ...buildPayload(), row_display: "none" });
  if (data.loading) { showLoadingOverlay(); return; }
  setLastDashboard({ ...lastDashboard, ...data });
  state.selected_entry = data.selected_entry ?? null;
  renderDetail(data.detail);
}


// ---------------------------------------------------------------------------
// Main load functions
// ---------------------------------------------------------------------------

let _loadSeq = 0;
let _selectFirstOnLoad = false;   // consumed once by loadEntriesOnly

export function scheduleSelectFirst() { _selectFirstOnLoad = true; }

export async function loadEntriesOnly() {
  const seq = ++_loadSeq;
  const selectFirst = _selectFirstOnLoad;
  _selectFirstOnLoad = false;

  const data = await fetchDashboard(buildPayload());
  if (seq !== _loadSeq) return;  // superseded by a newer request
  if (data.loading) {
    showLoadingOverlay();
    return;
  }
  setLastDashboard(data);

  const ids = data.visible_entry_ids ?? [];
  const autoSelected = selectFirst && !data.selected_entry && ids.length > 0;
  state.selected_entry = autoSelected ? ids[0] : (data.selected_entry ?? null);
  setVisibleEntryIds(ids);

  $("#entry-count").textContent = `${data.counts.visible_entries} shown`;
  renderPills();
  renderClassList(data.class_cards);

  const opts = data.spec_options ?? {};
  renderSpecPills("#spec-crs",        opts.crs        ?? [], state.spec_filters.crs);
  renderSpecPills("#spec-resolution", opts.resolution ?? [], state.spec_filters.resolution);
  renderSpecPills("#spec-bounds",     opts.bounds     ?? [], state.spec_filters.bounds, (v) => {
    try { return boundsLatLonText(JSON.parse(v)) ?? v; } catch { return v; }
  });

  const versionOptions = data.version_options ?? [];
  setLastVersionOptions(versionOptions);
  _renderVersionSelect(versionOptions);

  if (_viewMode === "compact") {
    renderEntryPills(data.table_rows);
  } else {
    renderTableHead();
    renderTableBody(data.table_rows);
  }

  if (autoSelected) {
    applyTableSelection();
    loadDetail();
  } else {
    renderDetail(data.detail);
  }
}

export async function loadEntries() {
  renderFilterRows();
  await loadEntriesOnly();
}


// ---------------------------------------------------------------------------
// Diagnostics modal
// ---------------------------------------------------------------------------

export function showDiagnostics() {
  if (!lastDashboard) {
    toast("No data loaded yet");
    return;
  }

  const counts = lastDashboard.counts     ?? {};
  const diag   = lastDashboard.diagnostics ?? {};

  const scanned     = diag.scanned_hash_paths ?? 0;
  const created     = diag.created_entries   ?? 0;
  const missingHash = diag.missing_state_hash ?? 0;
  const staleHidden = diag.stale_hidden       ?? 0;

  const totalClasses   = counts.classes        ?? 0;
  const loadedClasses  = counts.classes_loaded ?? 0;
  const unloadedClasses = totalClasses - loadedClasses;

  function row(label, value, warn = false) {
    const cls = warn && value > 0 ? ' class="diag-warn"' : "";
    return `<tr${cls}>
      <td class="diag-label">${label}</td>
      <td class="diag-val">${value}</td>
    </tr>`;
  }

  _openModal(
    "Registry diagnostics",
    `<table class="diag-table">
       <tbody>
         <tr class="diag-section-row"><td colspan="2">Classes</td></tr>
         ${row("Total",                           totalClasses)}
         ${row("In registry",                   loadedClasses)}
         ${row("Cache-only",                    unloadedClasses, true)}

         <tr class="diag-section-row"><td colspan="2">Scan</td></tr>
         ${row("Params files scanned", scanned)}
         ${row("Entries created",      created)}

         <tr class="diag-section-row"><td colspan="2">Entry quality</td></tr>
         ${row("Missing state hash",      missingHash, true)}
         ${staleHidden > 0 ? row("Stale entries hidden", staleHidden, false) : ""}
       </tbody>
     </table>`,
    "sm"
  );
}


// ---------------------------------------------------------------------------
// View mode (Compact / Detailed)
// ---------------------------------------------------------------------------

export function applyViewMode(mode, reload = true) {
  setViewMode(mode);

  const layout = document.querySelector(".app-layout");
  layout.classList.toggle("mode-compact",  mode === "compact");
  layout.classList.toggle("mode-detailed", mode === "detailed");

  // Restore saved detail pane width when entering detailed mode
  if (mode === "detailed") {
    const saved = localStorage.getItem("registry.detail-w");
    if (saved) $("#detail-pane").style.flex = `0 0 ${saved}px`;
  } else {
    $("#detail-pane").style.flex = "";
  }

  document.querySelectorAll("#entries-screen-tabs .kind-tab").forEach((tab) =>
    tab.classList.toggle("active", tab.dataset.mode === mode)
  );
  $("#display-seg")?.classList.toggle("seg-disabled", mode === "compact");

  // In compact mode, header rows only; in detailed mode default to "all".
  if (mode === "compact") {
    state.row_display = "none";
  } else {
    if (state.row_display === "none") state.row_display = "all";
    document.querySelectorAll("#display-seg .kind-tab").forEach((b) =>
      b.classList.toggle("active", b.dataset.val === state.row_display)
    );
  }

  if (reload) loadEntriesOnly();
}


// ---------------------------------------------------------------------------
// Export cart helpers
// ---------------------------------------------------------------------------

export function toggleCartEntry(id) {
  if (state.selected_entries.has(id)) {
    state.selected_entries.delete(id);
  } else {
    state.selected_entries.add(id);
  }
  const inCart = state.selected_entries.has(id);
  // Update any rendered + / − buttons and row highlights for this entry
  document.querySelectorAll(`.select-icon[data-entry="${CSS.escape(id)}"]`).forEach((btn) => {
    btn.classList.toggle("select-icon--in", inCart);
    btn.title = inCart ? "Remove from export" : "Add to export";
  });
  document.querySelectorAll(`[data-entry="${CSS.escape(id)}"].entry-pill, tr.row-entry[data-entry="${CSS.escape(id)}"]`).forEach((el) => {
    el.classList.toggle("selected-for-export", inCart);
  });
  // Also sync the cart button in the table (text node)
  document.querySelectorAll(`tr.row-entry[data-entry="${CSS.escape(id)}"] .select-icon`).forEach((btn) => {
    btn.classList.toggle("select-icon--in", inCart);
    btn.title = inCart ? "Remove from export" : "Add to export";
  });
  _updateCartTab();
  if (document.querySelector('.view-export')?.style.display !== 'none') {
    _rerenderExportView();
  }
}

// Lazy reference to cart tab badge updater — injected from export-view.js at boot time
let _updateCartTab = () => {};
export function setUpdateCartTab(fn) { _updateCartTab = fn; }

// Lazy reference to export view re-renderer — injected from export-view.js at boot time
let _rerenderExportView = () => {};
export function setRerenderExportView(fn) { _rerenderExportView = fn; }

export function toggleSelectMode() {
  state.select_mode = !state.select_mode;
  const btn = $("#btn-select-mode");
  if (btn) btn.classList.toggle("active", state.select_mode);
  document.body.classList.toggle("select-mode", state.select_mode);
}


// ---------------------------------------------------------------------------
// Code-state getter (injected from code-view.js at boot time)
// ---------------------------------------------------------------------------

let _getCodeState = () => null;

export function setEntriesCodeState(getCodeState) {
  _getCodeState = getCodeState;
}


// ---------------------------------------------------------------------------
// Clean-cache modal (launched from diagnostics)
// ---------------------------------------------------------------------------

export function _buildCleanCacheModal() {
  return `
    <div class="clean-cache-modal">
      <label class="clean-cache-opt">
        <input type="checkbox" id="clean-dry-run" checked>
        Dry run (preview only — no files will be deleted)
      </label>
      <pre class="clean-cache-output" id="clean-cache-output">Press Run to start…</pre>
      <div class="clean-cache-footer">
        <button class="act-btn" id="btn-clean-run">Run</button>
      </div>
    </div>`;
}

export async function runCleanCache() {
  const dryRun = document.getElementById("clean-dry-run").checked;
  const output = document.getElementById("clean-cache-output");
  const runBtn = document.getElementById("btn-clean-run");
  runBtn.disabled = true;
  output.textContent = "Running…";
  try {
    const result = await postCleanCache(dryRun);
    output.textContent = result.lines.length ? result.lines.join("\n") : "(nothing to clean)";
    if (!dryRun) loadEntries();
  } catch (e) {
    output.textContent = `Error: ${e}`;
  } finally {
    runBtn.disabled = false;
  }
}


// ---------------------------------------------------------------------------
// Clean-source modal (launched from diagnostics)
// ---------------------------------------------------------------------------

export function _buildCleanSourceModal() {
  return `
    <div class="clean-cache-modal">
      <p style="margin:0 0 8px;font-size:0.85em;color:var(--text-muted)">
        Removes orphaned code snapshots and dependency trees from <code>.source/</code>.
        Keeps the latest snapshot per class and anything referenced by a live cache entry.
      </p>
      <label class="clean-cache-opt">
        <input type="checkbox" id="clean-source-dry-run" checked>
        Dry run (preview only — no files will be deleted)
      </label>
      <pre class="clean-cache-output" id="clean-source-output">Press Run to start…</pre>
      <div class="clean-cache-footer">
        <button class="act-btn" id="btn-clean-source-run">Run</button>
      </div>
    </div>`;
}

export async function runCleanSource() {
  const dryRun = document.getElementById("clean-source-dry-run").checked;
  const output = document.getElementById("clean-source-output");
  const runBtn = document.getElementById("btn-clean-source-run");
  runBtn.disabled = true;
  output.textContent = "Running…";
  try {
    const result = await postCleanSource(dryRun);
    output.textContent = result.lines.length ? result.lines.join("\n") : "(nothing to clean)";
    if (!dryRun) loadEntries();
  } catch (e) {
    output.textContent = `Error: ${e}`;
  } finally {
    runBtn.disabled = false;
  }
}


// ---------------------------------------------------------------------------
// Wire up cross-module loaders at init time (called from boot.js)
// ---------------------------------------------------------------------------

export function initEntries(navigateToCodeClass, getCodeState, showWhatChanged, navigateToCodeClassBySourceHash) {
  _getCodeState = getCodeState;
  fetch('/api/status').then(r => r.json()).then(s => { if (s.has_live_classes != null) hasLiveClasses = s.has_live_classes; }).catch(() => {});

  // Wire filters
  setFilterLoaders(loadEntries, loadEntriesOnly);

  // Wire pills
  setPillsLoaders(loadEntries);
  setPillsCodeState(getCodeState);

  // Wire class list
  setClassListLoaders(loadEntries);
  setClassListDashboard(() => lastDashboard);
  setClassListCodeState(getCodeState);

  // Wire table
  setTableActions(navigateToCodeClass, selectEntry, toggleCartEntry);

  // Wire detail pane
  setDetailActions(navigateToCodeClass, selectEntry, toggleClass, showWhatChanged ?? (() => {}), toggleCartEntry, loadEntries, navigateToCodeClassBySourceHash);
}
