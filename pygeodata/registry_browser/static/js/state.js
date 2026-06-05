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


// ---------------------------------------------------------------------------
// Navigation history — proper back/forward with view mode
// ---------------------------------------------------------------------------

const _back    = [];   // stack of past snapshots
const _forward = [];   // stack of future snapshots (cleared on new navigation)

/** Snapshot everything that constitutes "where the user is", including view mode. */
function _snapshot(viewMode) {
  return {
    view_mode:        viewMode,
    selected_classes: [...state.selected_classes],
    selected_entry:   state.selected_entry,
    kind_filter:      state.kind_filter,
    logic_mode:       state.logic_mode,
    row_display:      state.row_display,
    spec_filters: {
      crs:        [...state.spec_filters.crs],
      resolution: [...state.spec_filters.resolution],
      bounds:     [...state.spec_filters.bounds],
    },
    filters: state.filters.map((f) => ({ ...f })),
  };
}

function _restore(snap) {
  state.selected_classes = snap.selected_classes;
  state.selected_entry   = snap.selected_entry;
  state.kind_filter      = snap.kind_filter;
  state.logic_mode       = snap.logic_mode       ?? state.logic_mode;
  state.row_display      = snap.row_display      ?? state.row_display;
  state.spec_filters     = snap.spec_filters;
  state.filters          = snap.filters;
}

/**
 * Call before every user-driven navigation.
 * Clears the forward stack (new navigation invalidates forward history).
 */
export function pushHistory(viewMode) {
  _back.push(_snapshot(viewMode));
  _forward.length = 0;
}

/**
 * Go back one step. Returns the snapshot (including view_mode) or null.
 */
export function navigateBack(currentViewMode) {
  if (!_back.length) return null;
  _forward.push(_snapshot(currentViewMode));
  const snap = _back.pop();
  _restore(snap);
  return snap;
}

/**
 * Go forward one step. Returns the snapshot (including view_mode) or null.
 */
export function navigateForward(currentViewMode) {
  if (!_forward.length) return null;
  _back.push(_snapshot(currentViewMode));
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
    spec_filters:     state.spec_filters,
    filters:          activeFilters,
  };
}
