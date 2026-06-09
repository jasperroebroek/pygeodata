/**
 * class-list.js
 *
 * Entries sidebar class list — renderClassList, toggleClass,
 * _showEmptyClasses, _multiSelectEnabled.
 */

import { $, $$, esc, badge } from './utils.js';
import { state } from './state.js';
import { _viewMode, _topView, pushHistory } from './nav.js';

export let _showEmptyClasses = false;
export let _multiSelectEnabled = false;

export function setShowEmptyClasses(v) { _showEmptyClasses = v; }
export function setMultiSelectEnabled(v) { _multiSelectEnabled = v; }

export function renderClassList(classCards) {
  const el = $("#class-list");
  const nameFilter = ($('#entry-class-filter')?.value ?? '').toLowerCase();

  const kindOk = (c) =>
    state.kind_filter === "all" || (c.object_type ?? "").toLowerCase() === state.kind_filter;

  const all = classCards.filter(kindOk);
  const afterEmpty = _showEmptyClasses
    ? all
    : all.filter((c) => c.visible_record_count > 0 || c.selected);
  const visible = nameFilter
    ? afterEmpty.filter((c) => c.class_name.toLowerCase().includes(nameFilter))
    : afterEmpty;

  const total = all.length;
  $("#class-count-badge").textContent =
    visible.length < total ? `${visible.length} / ${total}` : total;

  if (!visible?.length) {
    el.innerHTML = `<div class="empty-list">No classes.</div>`;
    return;
  }

  el.innerHTML = visible
    .map((c) => {
      const cls = [
        "class-card",
        c.selected ? "filtered" : "",
        c.visible_record_count === 0 ? "class-card--dim" : "",
      ].filter(Boolean).join(" ");

      // Single dot: source stale (amber) > deps stale (orange) > cache-only (grey).
      // Stale dots only shown when the class has actually been run (has entries).
      const hasEntries = c.total_record_count > 0;
      const dot = (hasEntries && c.source_stale)
        ? `<span class="status-dot status-dot--source" title="Source code changed since last run — entries may be outdated"></span>`
        : (hasEntries && c.deps_stale)
          ? `<span class="status-dot status-dot--deps" title="An upstream dependency changed since last run — entries may be outdated"></span>`
          : !c.loaded
            ? `<span class="status-dot status-dot--cache" title="Cache-only — not loaded in Python registry"></span>`
            : "";

      return `
        <div class="${cls}" data-cls="${esc(c.class_name)}" title="${esc(c.class_name)}">
          <span class="class-card-name">${esc(c.class_name)}${dot}</span>
          <span class="class-card-meta">
            <span class="class-card-count">${c.visible_record_count}/${c.total_record_count}</span>
            ${badge(c.object_type, "badge-neutral")}
          </span>
        </div>`;
    })
    .join("");

  el.querySelectorAll("[data-cls]").forEach((card) => {
    card.onclick = () => toggleClass(card.dataset.cls);
  });
}

export function toggleClass(cn, { navigate = false } = {}) {
  pushHistory(_viewMode, _topView === 'code' ? _getCodeState() : null);
  if (_multiSelectEnabled) {
    const i = state.selected_classes.indexOf(cn);
    if (i === -1) state.selected_classes.push(cn);
    else state.selected_classes.splice(i, 1);
  } else {
    if (state.selected_classes.length === 1 && state.selected_classes[0] === cn) {
      state.selected_classes = [];
    } else {
      state.selected_classes = [cn];
    }
  }
  state.selected_entry = null;

  // When navigating via a link (dependency/json explorer), sync the kind filter
  if (navigate && state.selected_classes.includes(cn)) {
    const card = (_getLastDashboard()?.class_cards ?? []).find((c) => c.class_name === cn);
    if (card && state.kind_filter !== "all" && (card.object_type ?? "").toLowerCase() !== state.kind_filter) {
      state.kind_filter = "all";
      $$("#kind-tabs .kind-tab").forEach((t) =>
        t.classList.toggle("active", t.dataset.kind === "all")
      );
    }
  }

  _loadEntries();
}

// Lazy references — set at init time.
let _loadEntries = () => {};
let _getCodeState = () => null;
let _getLastDashboard = () => null;

export function setClassListLoaders(loadEntries) {
  _loadEntries = loadEntries;
}

export function setClassListCodeState(getCodeState) {
  _getCodeState = getCodeState;
}

export function setClassListDashboard(getLastDashboard) {
  _getLastDashboard = getLastDashboard;
}
