/**
 * table.js
 *
 * Main entries table — renderTable, incremental rendering, PAGE_ENTRIES, scroll logic.
 */

import { $, esc, softBreak, badge } from './utils.js';
import { state } from './state.js';
import { _viewMode } from './nav.js';
import { downloadSingleEntry } from './api.js';

export const PAGE_ENTRIES = 50;   // entries rendered per chunk

// Track the flat list of visible entry IDs for arrow-key navigation
export let _visibleEntryIds = [];
export function setVisibleEntryIds(v) { _visibleEntryIds = v; }

// Internal incremental state
export let _tableRows = [];       // full flat row array from last fetch
export let _tableEntryCount = 0;  // number of header rows rendered so far

export function renderTableHead() {
  $("#table-head").innerHTML = `
    <th class="col-indent"></th>
    <th class="col-scope">Scope</th>
    <th class="col-param">Parameter</th>
    <th class="col-value">Value</th>`;
}

export function _buildRowsHtml(rows, start, maxEntries) {
  const html = [];
  let entriesSeen = 0;
  for (let i = start; i < rows.length; i++) {
    if (entriesSeen >= maxEntries) break;
    const r = rows[i];
    const next = rows[i + 1];
    const isActive = r.record_id === state.selected_entry;

    if (r.row_type === "header") {
      entriesSeen++;
      const spec = r.spec ?? {};
      const specParts = [
        spec.crs, spec.resolution_display, spec.shape,
        spec.bounds_display,
      ].filter(Boolean).map((s) => `<span class="tbl-spec-pill">${esc(s)}</span>`).join("");
      const warns    = r.warning_count ? `<span class="badge badge--sm badge-warn">${r.warning_count}w</span>` : "";
      const err      = r.error         ? `<span class="badge badge--sm badge-danger">!</span>` : "";
      const staleness = r.format_version_stale
        ? `<span class="badge badge--sm badge-danger" title="pygeodata version changed — entry must be regenerated">version</span>`
        : r.dep_hash_stale
          ? `<span class="badge badge--sm badge-warn" title="Dependencies changed — entry may be outdated">stale</span>`
          : "";
      const inCart = state.selected_entries.has(r.record_id);
      const cartBtn = `<button class="select-icon ${inCart ? "select-icon--in" : ""}" data-entry="${esc(r.record_id)}" title="${inCart ? "Remove from export" : "Add to export"}"></button>`;
      const dlBtn = `<button class="dl-icon" data-entry="${esc(r.record_id)}" title="Download this entry"></button>`;
      html.push(`
        <tr class="row-entry ${isActive ? "active" : ""} ${inCart ? "selected-for-export" : ""}" data-entry="${esc(r.record_id)}">
          <td colspan="4" class="cell-entry-hd">
            <div class="cell-entry-hd-inner">
              ${cartBtn}${dlBtn}
              <span class="cell-entry-left">
                <span class="entry-cls">${esc(r.class_name)}</span>
                <span class="badge badge--sm badge-neutral">${esc(r.object_type)}</span>
                ${staleness}${warns}${err}
              </span>
              <span class="cell-entry-spec">${specParts}</span>
            </div>
          </td>
        </tr>`);
    } else {
      const isLast    = !next || next.row_type === "header";
      const isRef     = r.value_type === "data_ref";
      const scope     = r.group && r.group !== "---" ? r.group : "";
      html.push(`
        <tr class="row-param ${isActive ? "active" : ""} ${isLast ? "entry-last" : ""}" data-entry="${esc(r.record_id)}">
          <td class="col-indent"></td>
          <td class="col-scope">${softBreak(scope)}</td>
          <td class="col-param">${softBreak(r.parameter ?? "")}</td>
          <td class="col-value${isRef ? " col-value--ref" : ""}">${softBreak(r.value ?? "")}</td>
        </tr>`);
    }
  }
  return { html, entriesSeen };
}

export function _appendTableRows(count = PAGE_ENTRIES) {
  const tbody = $("#table-body");
  if (!tbody || !_tableRows.length) return;

  // Find the flat index where the next unrendered entry starts
  let flatStart = _tableRows.length;  // assume all rendered
  let headersSeen = 0;
  for (let i = 0; i < _tableRows.length; i++) {
    if (_tableRows[i].row_type === "header") {
      if (headersSeen === _tableEntryCount) { flatStart = i; break; }
      headersSeen++;
    }
  }
  if (flatStart >= _tableRows.length) return;

  const { html, entriesSeen } = _buildRowsHtml(_tableRows, flatStart, count);
  if (!html.length) return;
  tbody.insertAdjacentHTML("beforeend", html.join(""));
  _tableEntryCount += entriesSeen;
}

export function applyTableSelection() {
  if (_viewMode === "compact") {
    $("#entry-pill-list")?.querySelectorAll(".entry-pill").forEach((pill) => {
      pill.classList.toggle("active", pill.dataset.entry === state.selected_entry);
    });
  } else {
    const tbody = $("#table-body");
    if (!tbody) return;
    tbody.querySelectorAll("tr.row-entry").forEach((row) => {
      const id = row.dataset.entry;
      row.classList.toggle("active", id === state.selected_entry);
      const inCart = state.selected_entries.has(id);
      row.classList.toggle("selected-for-export", inCart);
      const btn = row.querySelector(".select-icon");
      if (btn) {
        btn.classList.toggle("select-icon--in", inCart);
        btn.title = inCart ? "Remove from export" : "Add to export";
      }
    });
    tbody.querySelectorAll("tr.row-param").forEach((row) => {
      row.classList.toggle("active", row.dataset.entry === state.selected_entry);
    });
  }
}

export function renderTableBody(rows) {
  const tbody = $("#table-body");
  _tableRows = rows ?? [];
  _tableEntryCount = 0;

  if (!_tableRows.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" class="detail-empty" style="padding:14px">
          No entries match the current filters.
        </td>
      </tr>`;
    tbody.onclick = null;
    tbody.onmouseover = null;
    tbody.onmouseout = null;
    return;
  }

  tbody.innerHTML = "";
  _appendTableRows(PAGE_ENTRIES);

  // Single delegated listener on tbody — no per-row handlers
  tbody.onclick = (e) => {
    const clsLink = e.target.closest(".jmp-cls");
    if (clsLink) {
      e.preventDefault();
      _navigateToCodeClass(clsLink.dataset.cls, clsLink.dataset.depHash || null);
      return;
    }
    const cartBtn = e.target.closest(".select-icon");
    if (cartBtn) {
      _toggleCartEntry(cartBtn.dataset.entry);
      return;
    }
    const dlBtn = e.target.closest(".dl-icon");
    if (dlBtn) {
      downloadSingleEntry(dlBtn.dataset.entry);
      return;
    }
    const entryRow = e.target.closest("tr.row-entry");
    if (entryRow && !e.target.closest("a")) {
      _selectEntry(entryRow.dataset.entry);
      return;
    }
    const paramRow = e.target.closest("tr.row-param");
    if (paramRow) {
      _selectEntry(paramRow.dataset.entry);
      return;
    }
  };

  // Group hover: highlight all rows belonging to the same entry together
  tbody.onmouseover = (e) => {
    const row = e.target.closest("tr.row-entry, tr.row-param");
    if (!row) return;
    const id = row.dataset.entry;
    if (id === tbody._hoverId) return;
    _clearGroupHover(tbody);
    tbody._hoverId = id;
    tbody.querySelectorAll(`tr[data-entry="${CSS.escape(id)}"]`).forEach((r) =>
      r.classList.add("row-hover")
    );
  };

  tbody.onmouseout = (e) => {
    const to = e.relatedTarget;
    if (to && to.closest && to.closest("#table-body")) return;
    _clearGroupHover(tbody);
  };
}

function _clearGroupHover(tbody) {
  tbody._hoverId = null;
  tbody.querySelectorAll("tr.row-hover").forEach((r) => r.classList.remove("row-hover"));
}

// Lazy references — set by entries.js and code-view.js at init time.
let _navigateToCodeClass = () => {};
let _selectEntry = () => {};
let _toggleCartEntry = () => {};

export function setTableActions(navigateToCodeClass, selectEntry, toggleCartEntry) {
  _navigateToCodeClass = navigateToCodeClass;
  _selectEntry         = selectEntry;
  if (toggleCartEntry) _toggleCartEntry = toggleCartEntry;
}
