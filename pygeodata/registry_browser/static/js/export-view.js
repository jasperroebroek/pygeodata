import { $, toast } from './utils.js';
import { state } from './state.js';
import { startExportJob, pollExportStatus, downloadExport, fetchExportTable, downloadSingleEntry } from './api.js';
import { setUpdateCartTab, toggleCartEntry as _toggleCartEntry } from './entries.js';
import { _buildRowsHtml } from './table.js';


// ---------------------------------------------------------------------------
// Cart tab badge
// ---------------------------------------------------------------------------

export function updateCartTab() {
  const badgeEl = $("#export-tab-badge");
  const count = state.selected_entries.size;
  if (badgeEl) {
    badgeEl.textContent = count > 0 ? String(count) : "";
    badgeEl.classList.toggle("hidden", count === 0);
  }
}


// ---------------------------------------------------------------------------
// Render the export table view
// ---------------------------------------------------------------------------

export async function renderExportView() {
  updateCartTab();

  const count = state.selected_entries.size;
  const countEl = $("#export-toolbar-count");
  if (countEl) countEl.textContent = `${count} entr${count === 1 ? 'y' : 'ies'}`;

  const emptyEl = $("#export-empty");
  const tableScroll = $("#export-table-scroll");
  const thead = $("#export-table-head");
  const tbody = $("#export-table-body");

  if (!count) {
    if (emptyEl) emptyEl.classList.remove("hidden");
    if (tableScroll) tableScroll.style.display = "none";
    _syncButtons();
    return;
  }

  if (emptyEl) emptyEl.classList.add("hidden");
  if (tableScroll) tableScroll.style.display = "";

  if (thead) {
    thead.innerHTML = `
      <th class="col-indent"></th>
      <th class="col-scope">Scope</th>
      <th class="col-param">Parameter</th>
      <th class="col-value">Value</th>`;
  }
  if (tbody) tbody.innerHTML = "";

  if (tbody) {
    tbody.onclick = (e) => {
      e.stopPropagation();
      const cartBtn = e.target.closest(".select-icon");
      if (cartBtn) { _toggleCartEntry(cartBtn.dataset.entry); return; }
      const dlBtn = e.target.closest(".dl-icon");
      if (dlBtn) { downloadSingleEntry(dlBtn.dataset.entry); return; }
    };
  }

  try {
    const data = await fetchExportTable([...state.selected_entries]);
    const rows = data.table_rows ?? [];
    if (tbody) {
      if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="4" class="detail-empty" style="padding:14px">No entries selected.</td></tr>`;
      } else {
        const savedEntry = state.selected_entry;
        const savedCart  = state.selected_entries;
        state.selected_entry   = null;
        state.selected_entries = new Set();
        const { html } = _buildRowsHtml(rows, 0, rows.length);
        state.selected_entry   = savedEntry;
        state.selected_entries = savedCart;
        tbody.innerHTML = html.join("");
      }
    }
  } catch (err) {
    toast(`Failed to load export table: ${err}`);
  }

  _syncButtons();
}

function _syncButtons() {
  const btn = $("#btn-export-download");
  if (btn) btn.disabled = state.selected_entries.size === 0 || _exporting;
}


// ---------------------------------------------------------------------------
// Download with progress indicator
// ---------------------------------------------------------------------------

let _exporting = false;

export async function startExport() {
  if (_exporting || !state.selected_entries.size) return;
  _exporting = true;
  _syncButtons();

  const includeSnapshots = true;
  const prog      = $("#export-progress");
  const progBar   = $("#export-progress-bar");
  const progLabel = $("#export-progress-label");

  if (prog) prog.classList.remove("hidden");
  if (progLabel) progLabel.textContent = "Starting…";
  if (progBar) progBar.style.width = "0%";

  try {
    const { job_id, total } = await startExportJob([...state.selected_entries], includeSnapshots);

    while (true) {
      await new Promise((r) => setTimeout(r, 500));
      const status = await pollExportStatus(job_id);

      if (status.error) throw new Error(status.error);

      const pct = total > 0 ? Math.round((status.done / total) * 100) : 0;
      if (progBar) progBar.style.width = `${pct}%`;
      if (progLabel) progLabel.textContent = `${status.done} / ${total} files`;

      if (status.status === 'complete') break;
    }

    if (progBar) progBar.style.width = "100%";
    if (progLabel) progLabel.textContent = "Done — downloading…";
    downloadExport(job_id);

    setTimeout(() => {
      if (prog) prog.classList.add("hidden");
      if (progLabel) progLabel.textContent = "";
      if (progBar) progBar.style.width = "0%";
    }, 2000);
  } catch (err) {
    if (prog) prog.classList.add("hidden");
    toast(`Export failed: ${err}`);
  } finally {
    _exporting = false;
    _syncButtons();
  }
}


// ---------------------------------------------------------------------------
// Init — called from boot.js
// ---------------------------------------------------------------------------

export function leaveExportView() {
  if (_selectModeBeforeExport === false && state.select_mode) toggleSelectMode();
  _selectModeBeforeExport = null;
}

export function initExportView() {
  setUpdateCartTab(updateCartTab);

  const btn = $("#btn-export-download");
  if (btn) btn.onclick = startExport;
}
