/**
 * utils.js
 *
 * DOM helpers, toast, coordinate/bounds formatters, and last-dashboard cache.
 */

// ---------------------------------------------------------------------------
// Last dashboard response cache (used by the diagnostics modal and class list)
// ---------------------------------------------------------------------------

export let lastDashboard = null;
export let _lastVersionOptions = [];

export function setLastDashboard(v) { lastDashboard = v; }
export function setLastVersionOptions(v) { _lastVersionOptions = v; }


// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

export const $  = (sel) => document.querySelector(sel);
export const $$ = (sel) => Array.from(document.querySelectorAll(sel));

export function esc(value) {
  return String(value ?? "")
    .replaceAll("&",  "&amp;")
    .replaceAll("<",  "&lt;")
    .replaceAll(">",  "&gt;")
    .replaceAll('"',  "&quot;")
    .replaceAll("'",  "&#39;");
}

// Insert zero-width spaces after punctuation (except _) to allow line breaks there
export function softBreak(value) {
  return esc(value).replace(/([.\/\-:,;@#!?=+*|\\[\]{}()<>~`^%$&])/g, "$1&#8203;");
}

export function shortHash(value) {
  if (!value) return "";
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

export function badge(text, cls = "badge-accent") {
  return text != null
    ? `<span class="badge ${cls}">${esc(text)}</span>`
    : `<span class="kv-nil">unknown</span>`;
}

/** Format a coordinate with N/S or E/W suffix. */
export function fmtCoord(v, pos, neg) {
  return `${Math.abs(v)}° ${v >= 0 ? pos : neg}`;
}

/** Format bounds_latlon [lat_min, lon_min, lat_max, lon_max] as two corner points. */
export function boundsLatLonText(bl) {
  if (!bl || bl.length !== 4) return null;
  const [latMin, lonMin, latMax, lonMax] = bl;
  const sw = `${fmtCoord(latMin, "N", "S")}, ${fmtCoord(lonMin, "E", "W")}`;
  const ne = `${fmtCoord(latMax, "N", "S")}, ${fmtCoord(lonMax, "E", "W")}`;
  return `${sw} → ${ne}`;
}

export const _RASTER_EXTS = new Set([".tif", ".tiff", ".nc", ".vrt", ".npy", ".zarr"]);
export const _fileExt = (p) => p.slice(p.lastIndexOf(".")).toLowerCase();

/** Build a URL for the server-rendered bounds map. */
export function buildBoundsMapUrl(bl, crs) {
  return `/api/bounds-map?bounds=${encodeURIComponent(JSON.stringify(bl))}&crs=${encodeURIComponent(crs ?? "")}`;
}


// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

let toastTimer;

export function toast(message) {
  const el = $("#toast");
  el.textContent = String(message);
  el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), 1800);
}
