/**
 * filters.js
 *
 * Filter rows UI — add/remove filter rows, renderFilterRows, parseFiltersFromDOM.
 */

import { $, esc } from './utils.js';
import { state, BOOLEAN_TARGETS } from './state.js';

export const FILTER_TARGETS = [
  ["all",          "All"],
  ["class",        "Class"],
  ["crs",          "CRS"],
  ["key_group",    "Group"],
  ["key",          "Key"],
  ["value",        "Value"],
  ["path",         "Path"],
  ["has_warnings", "Has warnings"],
  ["has_error",    "Has error"],
];

export const FILTER_OPERATORS = [
  ["contains",     "contains",     "~"],
  ["equals",       "equals",       "="],
  ["starts",       "starts with",  "^"],
  ["not_contains", "not contains", "!~"],
];

function targetOptions(current) {
  return FILTER_TARGETS.map(
    ([val, label]) =>
      `<option value="${val}" ${val === current ? "selected" : ""}>${label}</option>`
  ).join("");
}

function operatorOptions(current) {
  return FILTER_OPERATORS.map(
    ([val, wordLabel]) =>
      `<option value="${val}" ${val === current ? "selected" : ""}>${wordLabel}</option>`
  ).join("");
}

export function renderFilterRows() {
  const host = $("#filter-rows");

  host.innerHTML = state.filters
    .map((f, i) => {
      const isBoolean = BOOLEAN_TARGETS.has(f.target);
      const rmBtn = i === 0
        ? `<button class="fr-rm rm-filter-btn fr-clear" data-i="${i}" title="Clear">✕</button>`
        : `<button class="fr-rm rm-filter-btn" data-i="${i}" title="Remove">✕</button>`;
      if (isBoolean) {
        return `
          <div class="filter-row filter-row--bool">
            <select class="fr-t" data-i="${i}" style="grid-column:1/-2">${targetOptions(f.target)}</select>
            ${rmBtn}
          </div>`;
      }
      return `
        <div class="filter-row">
          <select class="fr-t" data-i="${i}">${targetOptions(f.target)}</select>
          <select class="fr-op" data-i="${i}">${operatorOptions(f.operator)}</select>
          <input  class="fr-v" data-i="${i}" value="${esc(f.value)}" placeholder="…" autocomplete="off">
          ${rmBtn}
        </div>`;
    })
    .join("");

  host.querySelectorAll(".fr-t").forEach((sel) => {
    sel.onchange = (e) => {
      state.filters[+e.target.dataset.i].target = e.target.value;
      renderFilterRows();  // re-render to show/hide operator+value for boolean targets
      // loadEntriesOnly imported lazily via entries.js to avoid circular deps
      _loadEntriesOnly();
    };
  });

  host.querySelectorAll(".fr-op").forEach((sel) => {
    sel.onchange = (e) => {
      state.filters[+e.target.dataset.i].operator = e.target.value;
      _loadEntries();
    };
  });

  host.querySelectorAll(".fr-v").forEach((input) => {
    input.oninput = (e) => {
      state.filters[+e.target.dataset.i].value = e.target.value;
      // Don't call renderFilterRows() here — that would destroy the focused input.
      _loadEntriesOnly();
    };
  });

  host.querySelectorAll(".fr-rm").forEach((btn) => {
    btn.onclick = (e) => {
      const i = +e.target.dataset.i;
      if (i === 0) {
        state.filters[0].value = "";
        renderFilterRows();
      } else {
        state.filters.splice(i, 1);
        renderFilterRows();
      }
      _loadEntries();
    };
  });
}

// Lazy references to avoid circular imports — set by entries.js at init time.
let _loadEntries = () => {};
let _loadEntriesOnly = () => {};

export function setFilterLoaders(loadEntries, loadEntriesOnly) {
  _loadEntries = loadEntries;
  _loadEntriesOnly = loadEntriesOnly;
}
