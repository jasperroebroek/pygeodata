/**
 * boot.js
 *
 * Boot sequence — wires cross-module dependencies and kicks off the first load.
 */

import { toast } from './utils.js';
import { $$, lastDashboard } from './utils.js';
import { _viewMode, updateNavBtns } from './nav.js';
import {
  loadEntries, applyViewMode, initEntries,
} from './entries.js';
import {
  navigateToCodeClass, getCodeState, loadCodeView, showView,
  codeLoaded, codeSelectedVersion, codeClasses, codeSelectedClass,
  selectCodeVersion, selectCodeClass,
  codeBrowseMode, codeAllClasses, selectCodeClassFirst,
} from './code-view.js';
import { setEventsCodeView } from './events.js';

// ---------------------------------------------------------------------------
// Wire cross-module lazy references
// ---------------------------------------------------------------------------

// entries.js needs navigateToCodeClass and getCodeState from code-view.js
initEntries(navigateToCodeClass, getCodeState);

// events.js needs the full code-view API as accessor functions
setEventsCodeView({
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
});

// ---------------------------------------------------------------------------
// Boot sequence
// ---------------------------------------------------------------------------

// Apply initial layout classes without triggering a load (loadEntries does the first fetch)
applyViewMode("compact", false);
updateNavBtns();

// Restore browse mode button state
$$('#code-browse-tabs .kind-tab').forEach((b) =>
  b.classList.toggle('active', b.dataset.browse === (localStorage.getItem('code_browse_mode') ?? 'version')));

// Restore top-level view mode (entries / code)
{
  const savedView = localStorage.getItem('view_mode_top') ?? 'entries';
  if (savedView === 'code') {
    showView('code');
    loadCodeView();
  }
}

loadEntries().catch((e) => toast(`Load error: ${e}`));
