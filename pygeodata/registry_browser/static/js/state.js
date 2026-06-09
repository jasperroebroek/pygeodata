/**
 * state.js
 *
 * Single source of truth for the UI state.  The `buildPayload()` function
 * serialises it into the JSON body that /api/dashboard expects (matching the
 * signature of `build_dashboard_payload` in payloads.py).
 */

export const state = {
  // --- selection -----------------------------------------------------------
  selected_classes: [],    // class_name string[]
  selected_entry:   null,  // record_id string  | null

  // --- filters -------------------------------------------------------------
  kind_filter:  "all",     // "all" | "data" | "figure"
  logic_mode:   "AND",     // "AND" | "OR" | "NOT"
  row_display:  "all",     // "selected" | "all" | "none"
  hide_stale:   false,     // when true, stale entries are excluded

  /** Version filter — mtime string of the selected code version, or null for all. */
  version_filter: null,

  /** Spec pill filters.  Empty array = no filter. */
  spec_filters: {
    crs:        [],
    resolution: [],
    bounds:     [],
  },

  /**
   * Text filter rows.  Each row maps to the `filters` list in payloads.py.
   * Shape: { target, operator, value }
   */
  filters: [{ target: "all", operator: "contains", value: "" }],
};

// Restore hide_stale preference from localStorage
state.hide_stale = localStorage.getItem("hide_stale") === "true";


// ---------------------------------------------------------------------------
// Navigation history — proper back/forward with view mode
// ---------------------------------------------------------------------------

const _back    = [];   // stack of past snapshots
const _forward = [];   // stack of future snapshots (cleared on new navigation)

/**
 * Snapshot everything that constitutes "where the user is", including view mode.
 * codeState is an optional {version, className} for the Code view.
 */
function _snapshot(viewMode, codeState = null) {
  return {
    view_mode:        viewMode,
    selected_classes: [...state.selected_classes],
    selected_entry:   state.selected_entry,
    kind_filter:      state.kind_filter,
    logic_mode:       state.logic_mode,
    row_display:      state.row_display,
    version_filter:   state.version_filter,
    spec_filters: {
      crs:        [...state.spec_filters.crs],
      resolution: [...state.spec_filters.resolution],
      bounds:     [...state.spec_filters.bounds],
    },
    filters: state.filters.map((f) => ({ ...f })),
    code_version:    codeState?.version    ?? null,
    code_class_name: codeState?.className  ?? null,
  };
}

function _restore(snap) {
  state.selected_classes  = snap.selected_classes;
  state.selected_entry    = snap.selected_entry;
  state.kind_filter       = snap.kind_filter;
  state.logic_mode        = snap.logic_mode       ?? state.logic_mode;
  state.row_display       = snap.row_display      ?? state.row_display;
  state.version_filter    = snap.version_filter    ?? null;
  state.spec_filters      = snap.spec_filters;
  state.filters           = snap.filters;
}

/**
 * Call before every user-driven navigation.
 * Pass codeState = {version, className} when in Code view.
 * Clears the forward stack (new navigation invalidates forward history).
 */
export function pushHistory(viewMode, codeState = null) {
  _back.push(_snapshot(viewMode, codeState));
  _forward.length = 0;
}

/**
 * Go back one step. Returns the snapshot (including view_mode, code_version, code_class_name) or null.
 * Pass codeState for the current position so it can be pushed onto forward stack.
 */
export function navigateBack(currentViewMode, codeState = null) {
  if (!_back.length) return null;
  _forward.push(_snapshot(currentViewMode, codeState));
  const snap = _back.pop();
  _restore(snap);
  return snap;
}

/**
 * Go forward one step. Returns the snapshot (including view_mode, code_version, code_class_name) or null.
 * Pass codeState for the current position so it can be pushed onto back stack.
 */
export function navigateForward(currentViewMode, codeState = null) {
  if (!_forward.length) return null;
  _back.push(_snapshot(currentViewMode, codeState));
  const snap = _forward.pop();
  _restore(snap);
  return snap;
}

export function hasBack()    { return _back.length > 0; }
export function hasForward() { return _forward.length > 0; }


/**
 * Filter targets that are boolean flags — no operator or value needed.
 * Exported so main.js can use the same set without re-defining it.
 */
export const BOOLEAN_TARGETS = new Set(["has_warnings", "has_error"]);

/**
 * Build the POST body for /api/dashboard.
 * Only includes filters that have a non-empty value so the backend doesn't
 * waste time evaluating blank rows.
 */
export function buildPayload() {
  const activeFilters = state.filters.filter(
    (f) => BOOLEAN_TARGETS.has(f.target) || (f.value ?? "").trim() !== ""
  );

  return {
    selected_classes: state.selected_classes,
    selected_entry:   state.selected_entry,
    kind_filter:      state.kind_filter,
    logic_mode:       state.logic_mode,
    row_display:      state.row_display,
    hide_stale:       state.hide_stale,
    version_filter:   state.version_filter,
    spec_filters:     state.spec_filters,
    filters:          activeFilters,
  };
}
