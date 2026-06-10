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


// ---------------------------------------------------------------------------
// Export (background job)
// ---------------------------------------------------------------------------

/** Fetch table rows for selected entries (for the export view). */
export function fetchExportTable(recordIds) {
  return postJSON('/api/export/table', { record_ids: recordIds });
}

/** Start an export job. Returns { job_id, total }. */
export async function startExportJob(recordIds, includeSnapshots = true) {
  const response = await fetch("/api/export/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record_ids: recordIds, include_snapshots: includeSnapshots }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

/** Poll job status. Returns { status, done, total, error }. */
export async function pollExportStatus(jobId) {
  const response = await fetch(`/api/export/status/${encodeURIComponent(jobId)}`);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

/** Trigger browser download of completed job. */
export function downloadExport(jobId) {
  const a = document.createElement("a");
  a.href = `/api/export/download/${encodeURIComponent(jobId)}`;
  a.download = "pygeodata_export.tar";
  a.click();
}

/** Trigger direct download of a single entry's data file (no sidecars). */
export function downloadSingleEntry(recordId) {
  const a = document.createElement("a");
  a.href = `/api/export/single/${encodeURIComponent(recordId)}`;
  a.click();
}


// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

/** Delete a single cache entry by record ID. */
export async function deleteEntry(recordId) {
  const r = await fetch(`/api/entry/${encodeURIComponent(recordId)}`, { method: "DELETE" });
  if (!r.ok && r.status !== 202) throw new Error(await r.text());
  return r.json();
}

/** Run clean-cache on the server. Returns { lines, dry_run }. */
export function postCleanCache(dryRun = true) {
  return postJSON("/api/clean-cache", { dry_run: dryRun });
}
