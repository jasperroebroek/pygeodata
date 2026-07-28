/**
 * boot.js
 *
 * Boot sequence — wires cross-module dependencies and kicks off the first load.
 */

import { toast, $$ } from './utils.js';
import { _viewMode, updateNavBtns } from './nav.js';
import {
  loadEntries, applyViewMode, initEntries, toggleSelectMode,
} from './entries.js';
import {
  navigateToCodeClass, navigateToCodeClassBySourceHash, getCodeState, loadCodeView, showView,
  codeLoaded, codeSelectedVersion, codeClasses, codeSelectedClass,
  selectCodeVersion, selectCodeClass,
  codeBrowseMode, codeAllClasses, selectCodeClassFirst,
  showWhatChanged, setOnShowExportView,
} from './code-view.js';
import { setEventsCodeView } from './events.js';
import { initExportView, renderExportView } from './export-view.js';

// ---------------------------------------------------------------------------
// Wire cross-module lazy references
// ---------------------------------------------------------------------------

// entries.js needs navigateToCodeClass, getCodeState, and showWhatChanged from code-view.js
initEntries(navigateToCodeClass, getCodeState, showWhatChanged, navigateToCodeClassBySourceHash);

// export-view.js wires its cart-tab updater into entries.js
initExportView();

// Tell code-view.js what to do when the Export tab is shown
setOnShowExportView(renderExportView);

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
// Default to detailed mode on normal/wide screens, compact on narrow ones.
const DEFAULT_MODE_BREAKPOINT = 1000;
applyViewMode(window.innerWidth >= DEFAULT_MODE_BREAKPOINT ? "detailed" : "compact", false);
updateNavBtns();

// Restore browse mode button state
$$('#code-browse-tabs .kind-tab').forEach((b) =>
  b.classList.toggle('active', b.dataset.browse === (localStorage.getItem('code_browse_mode') ?? 'version')));

// Restore top-level view mode (entries / code / export)
{
  const savedView = localStorage.getItem('view_mode_top') ?? 'entries';
  if (savedView === 'code') {
    showView('code');
    loadCodeView();
  }
  // export view restore is handled inside code-view.js module-level code
}

// Wire select-mode toggle button
document.getElementById('btn-select-mode')?.addEventListener('click', toggleSelectMode);

loadEntries().catch((e) => toast(`Load error: ${e}`));
