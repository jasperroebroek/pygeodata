/**
 * api.js
 *
 * Thin wrappers around the Flask backend endpoints.
 * All functions are async and return parsed JSON on success,
 * or throw an Error with the response text on failure.
 */


// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

async function postJSON(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok && response.status !== 202) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}


// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

/**
 * Fetch the full dashboard payload.
 * @param {object} payload - Serialised state from buildPayload().
 */
export function fetchDashboard(payload) {
  return postJSON("/api/dashboard", payload);
}


// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rebuild (refresh state from disk)
// ---------------------------------------------------------------------------

export function postRebuild() {
  return postJSON("/api/rebuild");
}


// ---------------------------------------------------------------------------
// Popups
// ---------------------------------------------------------------------------

export function fetchSourcePopup(className, sourcePath = null) {
  const base = `/api/popup/source?class_name=${encodeURIComponent(className)}`;
  const url = sourcePath ? `${base}&source_path=${encodeURIComponent(sourcePath)}` : base;
  return getJSON(url);
}

export function fetchGraphPopup(className, graphPath = null) {
  const base = `/api/popup/graph?class_name=${encodeURIComponent(className)}`;
  const url = graphPath ? `${base}&graph_path=${encodeURIComponent(graphPath)}` : base;
  return getJSON(url);
}

export function fetchJsonPopup(path) {
  return getJSON(`/api/popup/json?path=${encodeURIComponent(path)}`);
}


// ---------------------------------------------------------------------------
// File actions
// ---------------------------------------------------------------------------

/**
 * Ask the OS to open a file or directory.
 * @param {string} path - Absolute filesystem path.
 */
export function openPath(path) {
  return postJSON("/api/open", { path });
}

/**
 * Ask the OS to reveal a file in the file manager.
 * @param {string} path - Absolute filesystem path.
 */
export function revealPath(path) {
  return postJSON("/api/reveal", { path });
}

/**
 * Build a URL that serves a file through the backend.
 * @param {string} path - Absolute filesystem path.
 */
export function fileURL(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}
