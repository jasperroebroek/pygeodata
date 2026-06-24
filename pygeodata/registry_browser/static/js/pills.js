/**
 * pills.js
 *
 * Active-class pills (shown above the table) — renderPills, clearAllFilters, hasActiveFilters.
 */

import { $$, esc, _lastVersionOptions } from './utils.js';
import { state, BOOLEAN_TARGETS } from './state.js';
import { _viewMode, _topView, updateNavBtns, pushHistory } from './nav.js';
import { FILTER_TARGETS, FILTER_OPERATORS, renderFilterRows } from './filters.js';

export function hasActiveFilters() {
  if (state.selected_classes.length) return true;
  // kind_filter is primary navigation, not a filter — doesn't trigger "Clear all"
  if (Object.values(state.spec_filters).some((a) => a.length)) return true;
  if (state.version_filter) return true;
  if (state.filters.some((f) => BOOLEAN_TARGETS.has(f.target) || (f.value ?? "").trim())) return true;
  return false;
}

export function renderPills() {
  const el = document.querySelector("#active-pills");
  el.innerHTML = "";
  const hasFilters = hasActiveFilters();
  document.querySelector("#btn-clear-all").classList.toggle("hidden", !hasFilters);
  document.querySelector("#pills-row").classList.toggle("pills-empty", !hasFilters);
  updateNavBtns();

  // Import codeState lazily from code-view to avoid circular dependency.
  // _topView and _viewMode are re-read at call time.
  const codeState = _topView === 'code' ? _getCodeState() : null;

  state.selected_classes.forEach((cn) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.innerHTML = `${esc(cn)}<button class="pill-rm">✕</button>`;
    pill.querySelector("button").onclick = () => {
      pushHistory(_viewMode, _topView === 'code' ? _getCodeState() : null);
      state.selected_classes.splice(state.selected_classes.indexOf(cn), 1);
      _loadEntries();
    };
    el.appendChild(pill);
  });

  state.filters.forEach((f, i) => {
    const isBoolean = BOOLEAN_TARGETS.has(f.target);
    if (!isBoolean && !(f.value ?? "").trim()) return;
    const targetLabel = FILTER_TARGETS.find(([v]) => v === f.target)?.[1] ?? f.target;
    const pill = document.createElement("span");
    pill.className = "pill pill-filter";
    if (isBoolean) {
      pill.innerHTML = `${esc(targetLabel)}<button class="pill-rm">✕</button>`;
    } else {
      const opLabel = FILTER_OPERATORS.find(([v]) => v === f.operator)?.[2] ?? f.operator;
      pill.innerHTML = `<span class="pill-meta">${esc(targetLabel)} ${esc(opLabel)}</span> ${esc(f.value)}<button class="pill-rm">✕</button>`;
    }
    pill.querySelector("button").onclick = () => {
      pushHistory(_viewMode, _topView === 'code' ? _getCodeState() : null);
      if (i === 0) {
        state.filters[0] = { target: "all", operator: "contains", value: "" };
      } else {
        state.filters.splice(i, 1);
      }
      renderFilterRows();
      _loadEntries();
    };
    el.appendChild(pill);
  });

  [["crs", "CRS"], ["resolution", "Res"], ["bounds", "Bounds"]].forEach(([dim, label]) => {
    (state.spec_filters[dim] ?? []).forEach((v) => {
      const pill = document.createElement("span");
      pill.className = "pill pill-filter";
      pill.innerHTML = `<span class="pill-meta">${esc(label)}</span> ${esc(v)}<button class="pill-rm">✕</button>`;
      pill.querySelector("button").onclick = () => {
        const arr = state.spec_filters[dim];
        arr.splice(arr.indexOf(v), 1);
        _loadEntries();
      };
      el.appendChild(pill);
    });
  });

  if (state.version_filter) {
    const opt = (_lastVersionOptions ?? []).find((o) => o.version_id === state.version_filter);
    const label = opt?.label ?? state.version_filter;
    const pill = document.createElement("span");
    pill.className = "pill pill-filter";
    pill.innerHTML = `<span class="pill-meta">Snapshot</span> ${esc(label)}<button class="pill-rm">✕</button>`;
    pill.querySelector("button").onclick = () => {
      state.version_filter = null;
      document.querySelector("#version-select").value = "all";
      _loadEntries();
    };
    el.appendChild(pill);
  }
}

// Lazy references — set by entries.js and code-view.js at init time.
let _loadEntries = () => {};
let _getCodeState = () => null;

export function setPillsLoaders(loadEntries) {
  _loadEntries = loadEntries;
}

export function setPillsCodeState(getCodeState) {
  _getCodeState = getCodeState;
}
