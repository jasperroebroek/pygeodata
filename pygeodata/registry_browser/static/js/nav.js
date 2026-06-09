/**
 * nav.js
 *
 * Navigation/history helpers. Re-exports state-level push/navigate functions
 * and manages the top-level view variable and nav-button state.
 */

import { $, $$ } from './utils.js';
import { pushHistory, navigateBack, navigateForward, hasBack, hasForward } from './state.js';

export { pushHistory, navigateBack, navigateForward, hasBack, hasForward };

// ---------------------------------------------------------------------------
// View-mode state (Compact / Detailed) — kept here so all modules can read it
// ---------------------------------------------------------------------------

/** Current view mode for entries screen: 'compact' | 'detailed' */
export let _viewMode = "compact";
export function setViewMode(v) { _viewMode = v; }

/** Current top-level view: 'entries' | 'code'. Kept in sync by showView(). */
export let _topView = 'entries';
export function setTopView(v) { _topView = v; }

// ---------------------------------------------------------------------------
// Nav button updater
// ---------------------------------------------------------------------------

export function updateNavBtns() {
  // no nav buttons in topbar — keyboard-only (⌘[ / ⌘])
}
