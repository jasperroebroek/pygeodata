/**
 * main.js
 *
 * Wires up all UI interactions and delegates rendering to helper functions.
 * Imports state from state.js and API calls from api.js.
 */

import { state, buildPayload, BOOLEAN_TARGETS, pushHistory, navigateBack, navigateForward, hasBack, hasForward } from "./state.js";
import {
  fetchDashboard,
  fetchSourcePopup,
  fetchGraphPopup,
  fetchJsonPopup,
  postRebuild,
  openPath,
  revealPath,
  fileURL,
} from "./api.js";


// ---------------------------------------------------------------------------
// Last dashboard response cache (used by the diagnostics modal)
// ---------------------------------------------------------------------------

let lastDashboard = null;


// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function esc(value) {
  return String(value ?? "")
    .replaceAll("&",  "&amp;")
    .replaceAll("<",  "&lt;")
    .replaceAll(">",  "&gt;")
    .replaceAll('"',  "&quot;")
    .replaceAll("'",  "&#39;");
}

function shortHash(value) {
  if (!value) return "";
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

function badge(text, cls = "badge-accent") {
  return text != null
    ? `<span class="badge ${cls}">${esc(text)}</span>`
    : `<span class="kv-nil">unknown</span>`;
}

/** Format a coordinate with N/S or E/W suffix. */
function fmtCoord(v, pos, neg) {
  return `${Math.abs(v)}° ${v >= 0 ? pos : neg}`;
}

/** Format bounds_latlon [lat_min, lon_min, lat_max, lon_max] as two corner points. */
function boundsLatLonText(bl) {
  if (!bl || bl.length !== 4) return null;
  const [latMin, lonMin, latMax, lonMax] = bl;
  const sw = `${fmtCoord(latMin, "N", "S")}, ${fmtCoord(lonMin, "E", "W")}`;
  const ne = `${fmtCoord(latMax, "N", "S")}, ${fmtCoord(lonMax, "E", "W")}`;
  return `${sw} → ${ne}`;
}

const _RASTER_EXTS = new Set([".tif", ".tiff", ".nc", ".vrt", ".npy", ".zarr"]);
const _fileExt = (p) => p.slice(p.lastIndexOf(".")).toLowerCase();

/** Build a URL for the server-rendered bounds map. */
function buildBoundsMapUrl(bl, crs) {
  return `/api/bounds-map?bounds=${encodeURIComponent(JSON.stringify(bl))}&crs=${encodeURIComponent(crs ?? "")}`;
}


// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

let toastTimer;

function toast(message) {
  const el = $("#toast");
  el.textContent = String(message);
  el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), 1800);
}


// ---------------------------------------------------------------------------
// Navigation helpers — push history before every user-driven state change
// ---------------------------------------------------------------------------

function toggleClass(cn, { navigate = false } = {}) {
  pushHistory(_viewMode);
  if (_multiSelectEnabled) {
    const i = state.selected_classes.indexOf(cn);
    if (i === -1) state.selected_classes.push(cn);
    else state.selected_classes.splice(i, 1);
  } else {
    if (state.selected_classes.length === 1 && state.selected_classes[0] === cn) {
      state.selected_classes = [];
    } else {
      state.selected_classes = [cn];
    }
  }
  state.selected_entry = null;

  // When navigating via a link (dependency/json explorer), sync the kind filter
  if (navigate && state.selected_classes.includes(cn)) {
    const card = (lastDashboard?.class_cards ?? []).find((c) => c.class_name === cn);
    if (card && state.kind_filter !== "all" && (card.object_type ?? "").toLowerCase() !== state.kind_filter) {
      state.kind_filter = "all";
      $$("#kind-tabs .kind-tab").forEach((t) =>
        t.classList.toggle("active", t.dataset.kind === "all")
      );
    }
  }

  loadEntries();
}

function selectEntry(id, className = null) {
  pushHistory(_viewMode);

  if (id === state.selected_entry) {
    state.selected_entry = null;
    applyTableSelection();
    loadDetail();
    return;
  }

  state.selected_entry = id;

  // Only update class selection when navigating via a linked-entry (className hint provided).
  // Direct table/pill clicks should never modify the class filter.
  const ownerClass = className;

  if (ownerClass && !state.selected_classes.includes(ownerClass)) {
    if (_multiSelectEnabled) {
      state.selected_classes.push(ownerClass);
    } else {
      state.selected_classes = [ownerClass];
    }
    // Switch kind filter if the owner class isn't visible under the current one
    if (state.kind_filter !== "all") {
      const card = (lastDashboard?.class_cards ?? []).find((c) => c.class_name === ownerClass);
      if (card && (card.object_type ?? "").toLowerCase() !== state.kind_filter) {
        state.kind_filter = "all";
        $$("#kind-tabs .kind-tab").forEach((t) =>
          t.classList.toggle("active", t.dataset.kind === "all")
        );
      }
    }
    loadEntries();
  } else {
    applyTableSelection();
    loadDetail();
  }
}


// ---------------------------------------------------------------------------
// Compact view – entry pill list
// ---------------------------------------------------------------------------

function renderEntryPills(rows) {
  const el = $("#entry-pill-list");
  if (!rows?.length) {
    el.innerHTML = `<div class="detail-empty">No entries match the current filters.</div>`;
    el.onclick = null;
    return;
  }

  // Only render header rows (one pill per entry)
  const headers = rows.filter((r) => r.row_type === "header");
  if (!headers.length) {
    el.innerHTML = `<div class="detail-empty">No entries match the current filters.</div>`;
    el.onclick = null;
    return;
  }

  el.innerHTML = headers.map((r) => {
    const isActive = r.record_id === state.selected_entry;
    const tinyHash = r.record_id ? r.record_id.slice(0, 6) : "";
    const staleDot = r.dep_hash_stale
      ? `<span class="status-dot status-dot--source" title="Dependencies changed — entry may be outdated"></span>`
      : "";
    const flags = [
      r.warning_count ? `<span class="pill-flag pill-flag--warn">${r.warning_count}⚠</span>` : "",
      r.error         ? `<span class="pill-flag pill-flag--err">!</span>` : "",
    ].join("");
    return `
      <div class="entry-pill ${isActive ? "active" : ""}" data-entry="${esc(r.record_id)}">
        <span class="entry-pill-name">${esc(r.class_name)}${staleDot}</span>
        <span class="entry-pill-right">
          ${badge(r.object_type, "badge-neutral")}
          ${tinyHash ? `<span class="entry-pill-hash">${esc(tinyHash)}</span>` : ""}
          ${flags}
        </span>
      </div>`;
  }).join("");

  el.onclick = (e) => {
    const pill = e.target.closest(".entry-pill");
    if (pill) selectEntry(pill.dataset.entry);
  };
}


// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

function openModal(title, html, size = "") {
  const card = $("#modal").querySelector(".modal-card");
  card.classList.toggle("modal-card--sm", size === "sm");
  $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = html;
  $("#modal").classList.add("open");
}

function closeModal() {
  $("#modal").classList.remove("open");
  $("#modal-body").innerHTML = "";
}

$("#modal-close").onclick = closeModal;
$("#modal").onclick = (e) => {
  if (e.target.id === "modal") closeModal();
};

// Delegate clicks on class links injected into modal content (source + graph popups)
$("#modal-body").addEventListener("click", (e) => {
  const link = e.target.closest(".src-cls-link, .graph-node-link");
  if (link) {
    e.preventDefault();
    const cn = link.dataset.cls;
    if (!cn) return;
    closeModal();
    toggleClass(cn, { navigate: true });
    return;
  }

  const jmpCls = e.target.closest(".jmp-cls");
  if (jmpCls) {
    e.preventDefault();
    const cn = jmpCls.dataset.cls;
    if (!cn) return;
    closeModal();
    toggleClass(cn, { navigate: true });
  }
});


// ---------------------------------------------------------------------------
// File action buttons
// Attaches click handlers to elements with the action classes inside `root`.
// ---------------------------------------------------------------------------

function bindFileActions(root) {
  root.querySelectorAll(".js-copy").forEach((btn) => {
    btn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.path);
        toast("Copied");
      } catch {
        toast("Copy failed");
      }
    };
  });

  root.querySelectorAll(".js-copy-hash").forEach((btn) => {
    btn.onclick = async () => {
      try {
        await navigator.clipboard.writeText(btn.dataset.hash);
        toast("Hash copied");
      } catch {
        toast("Copy failed");
      }
    };
  });

  root.querySelectorAll(".js-open").forEach((btn) => {
    btn.onclick = () =>
      openPath(btn.dataset.path)
        .then(() => toast("Opened"))
        .catch(() => toast("Failed to open"));
  });

  root.querySelectorAll(".js-reveal").forEach((btn) => {
    btn.onclick = () =>
      revealPath(btn.dataset.path)
        .then(() => toast("Revealed"))
        .catch(() => toast("Failed to reveal"));
  });

  root.querySelectorAll(".co-params-toggle").forEach((btn) => {
    btn.onclick = () => {
      const card = document.getElementById(btn.dataset.card);
      if (!card) return;
      const showingAll = btn.classList.contains("active");
      card.querySelectorAll(".co-params-diff").forEach(el => { el.style.display = showingAll ? "" : "none"; });
      card.querySelectorAll(".co-params-all").forEach(el => { el.style.display = showingAll ? "none" : ""; });
      btn.classList.toggle("active", !showingAll);
      btn.textContent = showingAll ? "Show all params" : "Show diff only";
    };
  });

  root.querySelectorAll(".js-popup").forEach((btn) => {
    btn.onclick = async () => {
      const path = btn.dataset.path;
      if (path && path.endsWith(".json")) {
        try {
          const data = await fetchJsonPopup(path);
          const el = document.createElement("div");
          el.className = "jx-popup-wrap";
          el.appendChild(buildJsonExplorer(data.json, 0, { filterHidden: false }));
          openModal(data.title, el.outerHTML);
          // Bind toggle handlers after modal is in the DOM
          $("#modal-body").querySelectorAll(".jx-toggle").forEach(bindJxToggle);
        } catch {
          toast("Failed to load JSON");
        }
      } else {
        openModal(
          "File",
          `<iframe src="${fileURL(path)}" style="width:100%;height:74vh;border:0"></iframe>`
        );
      }
    };
  });

  root.querySelectorAll(".js-bounds-map").forEach((btn) => {
    btn.onclick = () => {
      const bl = JSON.parse(btn.dataset.bounds);
      const crs = btn.dataset.crs ?? "";
      openModal(`Bounds — ${crs}`, `<iframe src="${buildBoundsMapUrl(bl, crs)}" style="width:100%;height:74vh;border:0;border-radius:0"></iframe>`);
    };
  });

  root.querySelectorAll(".js-src").forEach((btn) => {
    btn.onclick = async () => {
      try {
        const data = await fetchSourcePopup(btn.dataset.cls, btn.dataset.srcPath ?? null);
        openModal(data.title, data.html);
      } catch {
        toast("Source unavailable");
      }
    };
  });

  root.querySelectorAll(".js-graph").forEach((btn) => {
    btn.onclick = async () => {
      try {
        const graphPath = btn.dataset.graphPath || null;
        const data = await fetchGraphPopup(btn.dataset.cls, graphPath);
        if (data.svg) {
          openModal(data.title, `<div class="svg-zoom-wrap">${data.svg}</div>`);
          const svg = $("#modal-body").querySelector("svg");
          if (svg) {
            svg.style.display = "block";
            svg.style.margin = "auto";
            bindZoom(svg);
          }
          const graphNodes = Array.from($("#modal-body").querySelectorAll(".graph-node-link"));
          graphNodes.forEach((node) => {
            node.addEventListener("mousemove", (e) => {
              const deepest = e.target.closest(".graph-node-link");
              graphNodes.forEach((n) => n.classList.toggle("hovered", n === deepest));
            });
          });
          $("#modal-body").querySelector("svg")?.addEventListener("mouseleave", () => {
            graphNodes.forEach((n) => n.classList.remove("hovered"));
          });
        } else if (data.pdf_path) {
          openModal(data.title, `<iframe src="${fileURL(data.pdf_path)}" style="width:100%;height:74vh;border:0"></iframe>`);
        }
      } catch {
        toast("Graph unavailable");
      }
    };
  });

  root.querySelectorAll(".js-img").forEach((img) => {
    img.onclick = () => {
      const html = `<div style="display:flex;align-items:center;justify-content:center;height:100%;overflow:hidden">
           <img src="${fileURL(img.dataset.path)}"
                style="max-width:100%;max-height:76vh;object-fit:contain;cursor:grab;user-select:none"
                id="modal-zoom-img">
         </div>`;
      openModal("Preview", html);
      const mi = $("#modal-zoom-img");
      if (mi) bindZoom(mi);
    };
  });
}

function bindZoom(img) {
  let scale = 1, ox = 0, oy = 0;
  let dragging = false, startX = 0, startY = 0, startOx = 0, startOy = 0;

  function clamp() {
    const r = img.getBoundingClientRect();
    const pr = img.parentElement.getBoundingClientRect();
    const maxX = Math.max(0, (r.width  - pr.width)  / 2);
    const maxY = Math.max(0, (r.height - pr.height) / 2);
    ox = Math.max(-maxX, Math.min(maxX, ox));
    oy = Math.max(-maxY, Math.min(maxY, oy));
  }

  function apply() {
    img.style.transform = `translate(${ox}px, ${oy}px) scale(${scale})`;
    img.style.transformOrigin = "center center";
    img.style.cursor = scale > 1 ? (dragging ? "grabbing" : "grab") : "default";
  }

  img.parentElement.addEventListener("wheel", (e) => {
    if (scale === 1 && !e.ctrlKey) return;
    e.preventDefault();
    if (e.ctrlKey) {
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
      scale = Math.max(1, Math.min(10, scale * factor));
      if (scale === 1) { ox = 0; oy = 0; }
    } else {
      ox -= e.deltaX;
      oy -= e.deltaY;
    }
    clamp();
    apply();
  }, { passive: false });

  img.addEventListener("mousedown", (e) => {
    if (scale <= 1) return;
    dragging = true; startX = e.clientX; startY = e.clientY;
    startOx = ox; startOy = oy;
    img.style.cursor = "grabbing";
    e.preventDefault();
  });

  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    ox = startOx + (e.clientX - startX);
    oy = startOy + (e.clientY - startY);
    clamp();
    apply();
  });

  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false; apply();
  });
}


// ---------------------------------------------------------------------------
// Action button group  (Open / Reveal / Copy / Popup)
// ---------------------------------------------------------------------------

function actionsHTML(path) {
  if (!path) {
    return `<span style="color:var(--faint);font-size:11px">Unavailable</span>`;
  }
  const p = esc(path);
  return `
    <div class="actions">
      <button class="act-btn js-open"   data-path="${p}">Open</button>
      <button class="act-btn js-reveal" data-path="${p}">Reveal</button>
      <button class="act-btn js-copy"   data-path="${p}">Copy path</button>
      <button class="act-btn js-popup"  data-path="${p}">Popup</button>
    </div>`;
}


// ---------------------------------------------------------------------------
// Spec pill helpers
// ---------------------------------------------------------------------------

function renderSpecPills(selector, options, selectedSet, labelFn = null) {
  const el = $(selector);
  if (!options?.length) {
    el.innerHTML = `<span style="color:var(--faint);font-size:11px">—</span>`;
    return;
  }
  el.innerHTML = options
    .map((v) => {
      const active = selectedSet.includes(v);
      const label = labelFn ? labelFn(v) : v;
      return `<button class="spec-pill ${active ? "active" : ""}" data-v="${esc(v)}" title="${esc(label)}">${esc(label)}</button>`;
    })
    .join("");
  el.querySelectorAll(".spec-pill").forEach((btn) => {
    btn.onclick = () => {
      const v = btn.dataset.v;
      const i = selectedSet.indexOf(v);
      if (i === -1) selectedSet.push(v);
      else selectedSet.splice(i, 1);
      renderSpecPills(selector, options, selectedSet);
      loadEntriesOnly();
    };
  });
}


// ---------------------------------------------------------------------------
// Filter rows (search section in sidebar)
// ---------------------------------------------------------------------------

const FILTER_TARGETS = [
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

const FILTER_OPERATORS = [
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

function renderFilterRows() {
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
      loadEntriesOnly();
    };
  });

  host.querySelectorAll(".fr-op").forEach((sel) => {
    sel.onchange = (e) => {
      state.filters[+e.target.dataset.i].operator = e.target.value;
      loadEntries();
    };
  });

  host.querySelectorAll(".fr-v").forEach((input) => {
    input.oninput = (e) => {
      state.filters[+e.target.dataset.i].value = e.target.value;
      // Don't call renderFilterRows() here — that would destroy the focused input.
      loadEntriesOnly();
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
      loadEntries();
    };
  });
}


// ---------------------------------------------------------------------------
// Active-class pills (shown above the table)
// ---------------------------------------------------------------------------

function hasActiveFilters() {
  if (state.selected_classes.length) return true;
  // kind_filter is primary navigation, not a filter — doesn't trigger "Clear all"
  if (Object.values(state.spec_filters).some((a) => a.length)) return true;
  if (state.filters.some((f) => BOOLEAN_TARGETS.has(f.target) || (f.value ?? "").trim())) return true;
  return false;
}

function updateNavBtns() {
  $("#btn-back").disabled    = !hasBack();
  $("#btn-forward").disabled = !hasForward();
}

function renderPills() {
  const el = $("#active-pills");
  el.innerHTML = "";
  const hasFilters = hasActiveFilters();
  $("#btn-clear-all").classList.toggle("hidden", !hasFilters);
  $("#pills-row").classList.toggle("pills-empty", !hasFilters);
  updateNavBtns();

  state.selected_classes.forEach((cn) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.innerHTML = `${esc(cn)}<button class="pill-rm">✕</button>`;
    pill.querySelector("button").onclick = () => {
      pushHistory(_viewMode);
      state.selected_classes.splice(state.selected_classes.indexOf(cn), 1);
      loadEntries();
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
      if (i === 0) {
        state.filters[0] = { target: "all", operator: "contains", value: "" };
      } else {
        state.filters.splice(i, 1);
      }
      renderFilterRows();
      loadEntries();
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
        loadEntries();
      };
      el.appendChild(pill);
    });
  });
}


// ---------------------------------------------------------------------------
// Entries screen – main table
// ---------------------------------------------------------------------------

function renderTableHead() {
  $("#table-head").innerHTML = `
    <th class="col-indent"></th>
    <th class="col-scope">Scope</th>
    <th class="col-param">Parameter</th>
    <th class="col-value">Value</th>`;
}

// Track the flat list of visible entry IDs for arrow-key navigation
let _visibleEntryIds = [];

// ---------------------------------------------------------------------------
// Incremental table rendering — renders PAGE_ENTRIES entries at a time,
// appending more as the user scrolls toward the bottom.
// ---------------------------------------------------------------------------

const PAGE_ENTRIES = 50;   // entries rendered per chunk
let _tableRows = [];       // full flat row array from last fetch
let _tableEntryCount = 0;  // number of header rows rendered so far

function _buildRowsHtml(rows, start, maxEntries) {
  const html = [];
  let entriesSeen = 0;
  for (let i = start; i < rows.length; i++) {
    if (entriesSeen >= maxEntries) break;
    const r = rows[i];
    const next = rows[i + 1];
    const isActive = r.record_id === state.selected_entry;

    if (r.row_type === "header") {
      entriesSeen++;
      const spec = r.spec ?? {};
      const bl = spec.bounds_latlon;
      const specParts = [
        spec.crs, spec.resolution, spec.shape,
        bl ? boundsLatLonText(bl) : null,
      ].filter(Boolean).map((s) => `<span class="tbl-spec-pill">${esc(s)}</span>`).join("");
      const warns    = r.warning_count ? `<span class="badge badge--sm badge-warn">${r.warning_count}w</span>` : "";
      const err      = r.error         ? `<span class="badge badge--sm badge-danger">!</span>` : "";
      const staleness = r.dep_hash_stale
        ? `<span class="badge badge--sm badge-warn" title="Dependencies changed — entry may be outdated">stale</span>` : "";
      html.push(`
        <tr class="row-entry ${isActive ? "active" : ""}" data-entry="${esc(r.record_id)}">
          <td colspan="4" class="cell-entry-hd">
            <div class="cell-entry-hd-inner">
              <span class="cell-entry-left">
                <a href="#" class="jmp-cls entry-cls" data-cls="${esc(r.class_name)}">${esc(r.class_name)}</a>
                <span class="badge badge--sm badge-neutral">${esc(r.object_type)}</span>
                ${staleness}${warns}${err}
              </span>
              <span class="cell-entry-spec">${specParts}</span>
            </div>
          </td>
        </tr>`);
    } else {
      const isLast    = !next || next.row_type === "header";
      const isRef     = r.value_type === "data_ref";
      const scope     = r.group && r.group !== "---" ? r.group : "";
      html.push(`
        <tr class="row-param ${isActive ? "active" : ""} ${isLast ? "entry-last" : ""}" data-entry="${esc(r.record_id)}">
          <td class="col-indent"></td>
          <td class="col-scope">${esc(scope)}</td>
          <td class="col-param">${esc(r.parameter ?? "")}</td>
          <td class="col-value${isRef ? " col-value--ref" : ""}">${esc(r.value ?? "")}</td>
        </tr>`);
    }
  }
  return { html, entriesSeen };
}

function _appendTableRows(count = PAGE_ENTRIES) {
  const tbody = $("#table-body");
  if (!tbody || !_tableRows.length) return;

  // Find the flat index where the next unrendered entry starts
  let flatStart = _tableRows.length;  // assume all rendered
  let headersSeen = 0;
  for (let i = 0; i < _tableRows.length; i++) {
    if (_tableRows[i].row_type === "header") {
      if (headersSeen === _tableEntryCount) { flatStart = i; break; }
      headersSeen++;
    }
  }
  if (flatStart >= _tableRows.length) return;

  const { html, entriesSeen } = _buildRowsHtml(_tableRows, flatStart, count);
  if (!html.length) return;
  tbody.insertAdjacentHTML("beforeend", html.join(""));
  _tableEntryCount += entriesSeen;
}

function applyTableSelection() {
  if (_viewMode === "compact") {
    $("#entry-pill-list")?.querySelectorAll(".entry-pill").forEach((pill) => {
      pill.classList.toggle("active", pill.dataset.entry === state.selected_entry);
    });
  } else {
    const tbody = $("#table-body");
    if (!tbody) return;
    tbody.querySelectorAll("tr.row-entry").forEach((row) => {
      row.classList.toggle("active", row.dataset.entry === state.selected_entry);
    });
    tbody.querySelectorAll("tr.row-param").forEach((row) => {
      row.classList.toggle("active", row.dataset.entry === state.selected_entry);
    });
  }
}

function renderTableBody(rows) {
  const tbody = $("#table-body");
  _tableRows = rows ?? [];
  _tableEntryCount = 0;

  if (!_tableRows.length) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" class="detail-empty" style="padding:14px">
          No entries match the current filters.
        </td>
      </tr>`;
    tbody.onclick = null;
    return;
  }

  tbody.innerHTML = "";
  _appendTableRows(PAGE_ENTRIES);

  // Single delegated listener on tbody — no per-row handlers
  tbody.onclick = (e) => {
    const clsLink = e.target.closest(".jmp-cls");
    if (clsLink) {
      e.preventDefault();
      toggleClass(clsLink.dataset.cls, { navigate: true });
      return;
    }
    const entryRow = e.target.closest("tr.row-entry");
    if (entryRow && !e.target.closest("a")) {
      selectEntry(entryRow.dataset.entry);
      return;
    }
  };
}


// ---------------------------------------------------------------------------
// Entries screen – sidebar class list
// ---------------------------------------------------------------------------

let _showEmptyClasses = false;
let _multiSelectEnabled = false;

function renderClassList(classCards) {
  const el = $("#class-list");

  const kindOk = (c) =>
    state.kind_filter === "all" || (c.object_type ?? "").toLowerCase() === state.kind_filter;

  const all = classCards.filter(kindOk);
  const visible = _showEmptyClasses
    ? all
    : all.filter((c) => c.visible_record_count > 0 || c.selected);

  const total = all.length;
  $("#class-count-badge").textContent =
    visible.length < total ? `${visible.length} / ${total}` : total;

  if (!visible?.length) {
    el.innerHTML = `<div class="empty-list">No classes.</div>`;
    return;
  }

  el.innerHTML = visible
    .map((c) => {
      const cls = [
        "class-card",
        c.selected ? "filtered" : "",
        c.visible_record_count === 0 ? "class-card--dim" : "",
      ].filter(Boolean).join(" ");

      // Single dot: source stale (amber) > deps stale (orange) > cache-only (grey).
      const dot = c.source_stale
        ? `<span class="status-dot status-dot--source" title="Source code changed since last run — entries may be outdated"></span>`
        : c.deps_stale
          ? `<span class="status-dot status-dot--deps" title="An upstream dependency changed since last run — entries may be outdated"></span>`
          : !c.loaded
            ? `<span class="status-dot status-dot--cache" title="Cache-only — not loaded in Python registry"></span>`
            : "";

      return `
        <div class="${cls}" data-cls="${esc(c.class_name)}">
          <span class="class-card-name">${esc(c.class_name)}${dot}</span>
          <span class="class-card-meta">
            ${badge(c.object_type, "badge-neutral")}
            <span class="class-card-count">${c.visible_record_count}/${c.total_record_count}</span>
          </span>
        </div>`;
    })
    .join("");

  el.querySelectorAll("[data-cls]").forEach((card) => {
    card.onclick = () => toggleClass(card.dataset.cls);
  });
}


// ---------------------------------------------------------------------------
// Entries screen – detail pane
// ---------------------------------------------------------------------------

function renderNoEntryPlaceholder() {
  const el = $("#entry-detail");
  const selected = state.selected_classes;

  if (!selected.length) {
    el.innerHTML = `<div class="detail-empty">Select an entry to view details.</div>`;
    return;
  }

  if (selected.length === 1) {
    // Single class selected — detail will be filled by renderDetail once data arrives;
    // this is just the transient placeholder shown before the first load completes.
    el.innerHTML = `<div class="detail-empty">Select an entry to view details.</div>`;
    return;
  }

  // Multiple classes selected — show a pill summary
  const pills = selected.map((cn) => badge(cn, "badge-neutral")).join(" ");
  el.innerHTML = `
    <div class="detail-empty" style="flex-direction:column;gap:10px;align-items:flex-start;padding:16px">
      <span style="color:var(--muted);font-size:12px">${selected.length} classes selected</span>
      <div style="display:flex;flex-wrap:wrap;gap:6px">${pills}</div>
      <span style="color:var(--faint);font-size:11px">Select an entry to view details.</span>
    </div>`;
}

function renderDetail(detail) {
  const el = $("#entry-detail");

  if (!detail) {
    renderNoEntryPlaceholder();
    return;
  }

  const classCard      = buildClassCard(detail);
  const entryCard      = buildEntryCard(detail.selected_entry);
  const figureCard     = buildFigureCard(detail.selected_entry);
  const coOutputsCard  = buildCoOutputsCard(detail.selected_entry);
  const sameSpecCard   = buildSameSpecSiblingsCard(detail.selected_entry);
  const linkedCard     = buildLinkedEntriesCard(detail.selected_entry);
  const paramsCard     = buildParamsCard(detail.selected_entry);

  el.innerHTML = [figureCard, classCard, entryCard, coOutputsCard, sameSpecCard, linkedCard, paramsCard].join("");

  // Re-bind JSON explorer toggles (lost when using innerHTML/outerHTML)
  el.querySelectorAll(".jx-toggle").forEach(bindJxToggle);

  bindFileActions(el);
  el.querySelectorAll(".fig-main").forEach(bindZoom);

  // Delegated handler for class links and entry links inside the detail pane
  el.onclick = (e) => {
    const cls = e.target.closest(".jmp-cls");
    if (cls) { e.preventDefault(); toggleClass(cls.dataset.cls, { navigate: true }); return; }
    const entry = e.target.closest(".jmp-entry");
    if (entry) { e.preventDefault(); selectEntry(entry.dataset.entry, entry.dataset.cls || null); return; }
  };
}

function buildClassCard(detail) {
  function depLinks(names) {
    return (names ?? []).length
      ? (names).map((n) => `<a href="#" class="jmp-cls" data-cls="${esc(n)}">${esc(n)}</a>`).join(", ")
      : `<span class="kv-nil">None</span>`;
  }

  const srcTitle = detail.loaded
    ? (detail.source_stale
        ? "Live source — reflects current code on disk (changed since last run)"
        : "Live source — reflects current code on disk")
    : "Source from registry snapshot (class not loaded)";
  const graphTitle = detail.loaded
    ? (detail.source_stale || detail.deps_stale
        ? "Live dependency graph — reflects current code (changed since last run)"
        : "Live dependency graph — reflects current code")
    : "Graph from registry snapshot (class not loaded)";

  const actions = [
    ...(detail.source_available
      ? [detail.loaded
          ? `<button class="act-btn js-src" data-cls="${esc(detail.class_name)}" title="${esc(srcTitle)}">Source</button>`
          : `<button class="act-btn js-src" data-cls="${esc(detail.class_name)}" data-src-path="${esc(detail.class_source_path)}" title="${esc(srcTitle)}">Source</button>`]
      : []),
    ...(detail.graph_available
      ? [`<button class="act-btn js-graph" data-cls="${esc(detail.class_name)}" data-graph-path="${esc(detail.class_graph_path ?? '')}" title="${esc(graphTitle)}">Graph</button>`]
      : []),
    ...(detail.class_registry_path
      ? [`<button class="act-btn js-popup" data-path="${esc(detail.class_registry_path)}" title="Registry snapshot written at last run">Registry</button>`]
      : []),
  ].join("");

  const rows = [
    ["Call dependencies",         depLinks(detail.call_dependency_names)],
    ["Inheritance dependencies",  depLinks(detail.inheritance_dependency_names)],
  ].map(([label, links]) => `
    <div class="kv-dep-block">
      <div class="spec-kv-label">${label}</div>
      <div class="spec-kv-val">${links}</div>
    </div>`).join("");

  const statusBadges = [
    !detail.loaded      ? `<span class="badge badge--sm badge-cache"  title="Not loaded in Python registry — Source and Graph reflect registry snapshot">cache-only</span>` : "",
    detail.source_stale ? `<span class="badge badge--sm badge-warn" title="Source code changed since last run — Source and Graph reflect current code, not the registry snapshot">stale</span>` : "",
    (!detail.source_stale && detail.deps_stale)
      ? `<span class="badge badge--sm badge-deps" title="An upstream dependency changed since last run — Source and Graph reflect current code, not the registry snapshot">stale</span>` : "",
  ].filter(Boolean).join("");

  const typeBadge = detail.object_type ? `${badge(detail.object_type, "badge-neutral")}` : "";

  return `
    <div class="dcard">
      <div class="dcard-hd">
        <span class="dcard-hd-label">Class</span>
        <span class="dcard-hd-title">${esc(detail.class_name)}${typeBadge ? `<span class="dcard-hd-type">${typeBadge}</span>` : ""}</span>
        ${(statusBadges || actions) ? `<div class="dcard-hd-actions">${[statusBadges, actions].filter(Boolean).join('<span class="dcard-hd-sep"></span>')}</div>` : ""}
      </div>
      <div class="dcard-body">${rows}</div>
    </div>`;
}

function buildEntryCard(entry) {
  if (!entry) return "";

  const hash = entry.state_hash ?? "";
  const hashShort = hash.slice(0, 6);
  const spec = entry.spec ?? {};
  const bl = spec.bounds_latlon;

  // CRS — link to epsg.io if EPSG code
  const epsgUrl = spec.crs?.startsWith("EPSG:") ? `https://epsg.io/${spec.crs.slice(5)}` : null;
  const crsVal = spec.crs
    ? epsgUrl
      ? `<a class="spec-cell-link" href="${esc(epsgUrl)}" target="_blank" rel="noopener">${esc(spec.crs)}</a>`
      : `<span class="kv-val">${esc(spec.crs)}</span>`
    : null;

  // Spec cells rendered in a 2×2 grid
  const specCell = (label, content) => content
    ? `<div class="spec-kv-cell"><span class="spec-kv-label">${label}</span><span class="spec-kv-val">${content}</span></div>`
    : `<div class="spec-kv-cell spec-kv-cell--empty"></div>`;

  const specGrid = `
    <div class="spec-kv-grid">
      ${specCell("CRS",        crsVal ?? "")}
      ${specCell("Resolution", spec.resolution ? `<span class="kv-val">${esc(spec.resolution)}</span>` : "")}
      ${specCell("Shape",      spec.shape      ? `<span class="kv-val">${esc(spec.shape)}</span>`      : "")}
      ${specCell("Bounds",     bl ? `<button class="act-btn js-bounds-map" data-bounds="${esc(JSON.stringify(bl))}" data-crs="${esc(spec.crs ?? '')}">${esc(boundsLatLonText(bl))}</button>` : "")}
    </div>`;

  const fileRow = entry.primary_file
    ? `<div class="entry-file-row">
        <span class="entry-file-name">${esc(entry.primary_file.label)}</span>
        ${!_RASTER_EXTS.has(_fileExt(entry.primary_file.path))
          ? `<button class="act-btn js-open"   data-path="${esc(entry.primary_file.path)}">Open</button>`
          : ""}
        <button class="act-btn js-reveal" data-path="${esc(entry.primary_file.path)}">Reveal</button>
        <button class="act-btn js-copy"   data-path="${esc(entry.primary_file.path)}">Copy path</button>
      </div>`
    : "";

  // Header actions: internal file buttons + hash chip
  const hasVisibleParams = entry.params_path && entry.params_tree
    && Object.keys(entry.params_tree).some(k => !HIDDEN_JSON_KEYS.has(k));

  const internalBtns = [
    hasVisibleParams               ? `<button class="act-btn act-btn--ghost js-popup" data-path="${esc(entry.params_path)}">Params</button>` : "",
    entry.state_hash_path      ? `<button class="act-btn act-btn--ghost js-popup" data-path="${esc(entry.state_hash_path)}">Hash</button>` : "",
    entry.execution_graph_path ? `<button class="act-btn act-btn--ghost js-popup" data-path="${esc(entry.execution_graph_path)}">Graph</button>` : "",
  ].filter(Boolean).join("");

  const hashChip = hash
    ? `<button class="act-btn js-copy-hash" data-hash="${esc(hash)}" title="${esc(hash)}">${esc(hashShort)}</button>`
    : "";

  const warnings = entry.warnings ?? [];
  const warningsHtml = warnings.length
    ? `<div class="entry-warnings">${warnings.map((w) =>
        `<div class="entry-warning">⚠ ${esc(w)}</div>`
      ).join("")}</div>`
    : "";

  const staleBadge = entry.dep_hash_stale
    ? `<span class="badge badge--sm badge-warn" title="Dependencies changed since this entry was last computed — results may no longer match current code">stale</span>`
    : "";

  const entryActionParts = [staleBadge, internalBtns, hashChip].filter(Boolean);
  const entryActions = entryActionParts.length
    ? `<div class="dcard-hd-actions">${entryActionParts.join('<span class="dcard-hd-sep"></span>')}</div>`
    : "";

  return `
    <div class="dcard">
      <div class="dcard-hd">
        <span class="dcard-hd-label">Entry</span>
        ${entryActions}
      </div>
      <div class="dcard-body dcard-body--entry">
        ${specGrid}
        ${fileRow ? `<div class="dcard-sep"></div>${fileRow}` : ""}
      </div>
      ${warningsHtml}
    </div>`;
}

function buildFigureCard(entry) {
  if (!entry) return "";
  const previews = entry.figure_previews ?? [];
  if (!previews.length) return "";
  return `
    <div class="dcard dcard--figure">
      <div class="dcard-hd">
        <span class="dcard-hd-label">Preview</span>
        <div class="dcard-hd-actions">
          <button class="act-btn act-btn--ghost js-open" data-path="${esc(previews[0])}">Open</button>
        </div>
      </div>
      <div class="dcard-body dcard-body--fig">
        <div class="fig-preview">
          ${previews.map((p, i) =>
            `<img class="js-img ${i === 0 ? "fig-main" : "fig-thumb"}"
                  data-path="${esc(p)}" src="${fileURL(p)}" alt=""
                  loading="${i === 0 ? "eager" : "lazy"}">`
          ).join("")}
        </div>
      </div>
    </div>`;
}

function buildLinkedEntriesCard(entry) {
  if (!entry) return "";
  const links = (entry.linked_entries ?? []);
  if (!links.length) return "";

  const rows = links.map((lnk) => {
    const hash = lnk.state_hash ?? "";
    const hashShort = hash.slice(0, 6) || "—";
    const nameEl = hash
      ? `<a href="#" class="jmp-entry link-entry-name" data-entry="${esc(hash)}" data-cls="${esc(lnk.class_name)}">${esc(lnk.class_name)}</a>`
      : `<span class="link-entry-name">${esc(lnk.class_name)}</span>`;

    // Nested params as indented lines instead of inline summary
    const paramLines = Object.entries(lnk.params_summary ?? {})
      .map(([k, v]) => `<div class="link-param-line"><span class="link-param-key">${esc(k)}</span><span class="link-param-val">${v}</span></div>`)
      .join("");

    return `
      <div class="link-row">
        <div class="link-row-top">
          <span class="link-param-name">${esc(lnk.param_name || "—")}</span>
          ${nameEl}
          <span class="link-hash" title="${esc(hash)}">${esc(hashShort)}</span>
        </div>
        ${paramLines ? `<div class="link-params">${paramLines}</div>` : ""}
      </div>`;
  }).join("");

  return `
    <div class="dcard">
      <div class="dcard-hd"><span class="dcard-hd-label">Associated entries</span></div>
      <div class="dcard-body dcard-body--links">${rows}</div>
    </div>`;
}

function buildCoOutputsCard(entry) {
  if (!entry) return "";
  const outputs = entry.co_outputs ?? [];
  if (!outputs.length) return "";

  const cardId = "co-outputs-card";
  const hasDiff = outputs.some(e => (e.diff_rows ?? []).length > 0);
  const hasRows = outputs.some(e => (e.rows ?? []).length > 0);
  const showToggle = hasDiff && hasRows;

  function renderParamRows(paramRows) {
    if (!paramRows || !paramRows.length) return "";
    return `<div class="link-params co-params">` +
      paramRows.map(r =>
        `<div class="link-param-line">
          <span class="link-param-key">${esc(r.parameter)}</span>
          <span class="link-param-val">${esc(r.value)}</span>
        </div>`
      ).join("") +
      `</div>`;
  }

  const outputRows = outputs.map((e) => {
    const hashShort = e.state_hash ? e.state_hash.slice(0, 6) : "—";
    const fileRow = e.primary_file
      ? `<div class="entry-file-row" style="margin-top:4px">
           <span class="entry-file-name">${esc(e.primary_file.label)}</span>
           ${!_RASTER_EXTS.has(_fileExt(e.primary_file.path))
             ? `<button class="act-btn js-open" data-path="${esc(e.primary_file.path)}">Open</button>`
             : ""}
           <button class="act-btn js-reveal" data-path="${esc(e.primary_file.path)}">Reveal</button>
           <button class="act-btn js-copy"   data-path="${esc(e.primary_file.path)}">Copy path</button>
         </div>`
      : "";
    const diffHtml  = renderParamRows(e.diff_rows ?? []);
    const allHtml   = renderParamRows(e.rows ?? []);
    return `
      <div class="link-row">
        <div class="link-row-top">
          <a href="#" class="jmp-entry link-entry-name" data-entry="${esc(e.record_id)}" data-cls="${esc(e.class_name)}">${esc(e.class_name)}</a>
          <span class="link-hash" title="${esc(e.state_hash ?? "")}">${esc(hashShort)}</span>
        </div>
        ${fileRow}
        <div class="co-params-diff">${diffHtml}</div>
        <div class="co-params-all" style="display:none">${allHtml}</div>
      </div>`;
  }).join("");

  const toggleBtn = showToggle
    ? `<button class="toggle-btn co-params-toggle" data-card="${cardId}">Show all params</button>`
    : "";

  return `
    <div class="dcard" id="${cardId}">
      <div class="dcard-hd">
        <span class="dcard-hd-label">Co-outputs</span>
        <div class="dcard-hd-actions">${toggleBtn}</div>
      </div>
      <div class="dcard-body dcard-body--links">${outputRows}</div>
    </div>`;
}

function buildSameSpecSiblingsCard(entry) {
  if (!entry) return "";
  const siblings = entry.same_instance_runs ?? [];
  if (!siblings.length) return "";

  const rows = siblings.map((s) => {
    const hashShort = s.state_hash ? s.state_hash.slice(0, 6) : "—";
    const spec = s.spec ?? {};
    const specParts = [spec.crs, spec.resolution, spec.shape]
      .filter(Boolean)
      .map((v) => `<span class="tbl-spec-pill">${esc(v)}</span>`)
      .join("");
    const fileRow = s.primary_file
      ? `<div class="entry-file-row" style="margin-top:4px">
           <span class="entry-file-name">${esc(s.primary_file.label)}</span>
           ${!_RASTER_EXTS.has(_fileExt(s.primary_file.path))
             ? `<button class="act-btn js-open" data-path="${esc(s.primary_file.path)}">Open</button>`
             : ""}
           <button class="act-btn js-reveal" data-path="${esc(s.primary_file.path)}">Reveal</button>
           <button class="act-btn js-copy"   data-path="${esc(s.primary_file.path)}">Copy path</button>
         </div>`
      : "";
    return `
      <div class="link-row">
        <div class="link-row-top">
          <a href="#" class="jmp-entry link-entry-name" data-entry="${esc(s.record_id)}" data-cls="${esc(s.class_name)}">${esc(s.class_name)}</a>
          ${specParts}
          <span class="link-hash" title="${esc(s.state_hash ?? "")}">${esc(hashShort)}</span>
        </div>
        ${fileRow}
      </div>`;
  }).join("");

  return `
    <div class="dcard">
      <div class="dcard-hd"><span class="dcard-hd-label">Sibling entries</span></div>
      <div class="dcard-body dcard-body--links">${rows}</div>
    </div>`;
}

// ---------------------------------------------------------------------------
// JSON explorer — shared renderer for params card and JSON file popups
// ---------------------------------------------------------------------------

const HIDDEN_JSON_KEYS = new Set([
  "class_name", "object_type", "source_hash", "state_hash",
  "dependency_tree_hash", "tree", "call_dependencies",
  "inheritance_dependencies", "dependencies", "crs", "transform",
]);

function buildJsonExplorer(data, depth = 0, opts = { filterHidden: true }) {
  const ul = document.createElement("ul");
  ul.className = `jx-list jx-depth-${depth}`;

  if (Array.isArray(data)) {
    data.forEach((item, idx) => {
      ul.appendChild(_jxItem(String(idx), item, depth, opts));
    });
  } else if (data !== null && typeof data === "object") {
    Object.entries(data).forEach(([k, v]) => {
      if (opts.filterHidden && depth === 0 && HIDDEN_JSON_KEYS.has(k)) return;
      ul.appendChild(_jxItem(k, v, depth, opts));
    });
  }

  return ul;
}

function _jxClassLabel(obj) {
  const cn = obj?.class_name;
  return cn ? `<a href="#" class="jx-cls jmp-cls" data-cls="${esc(cn)}">${esc(cn)}</a>` : "";
}

function _jxItem(key, value, depth, opts = { filterHidden: true }) {
  const li = document.createElement("li");
  li.className = "jx-item";

  const isDataRef = value !== null && typeof value === "object" && !Array.isArray(value)
    && value.class_name && value.params;
  const isObject  = value !== null && typeof value === "object" && !isDataRef;
  const isArray   = Array.isArray(value);

  if (isDataRef) {
    const inner = value.params ?? {};
    const hasChildren = Object.keys(inner).length > 0;
    li.innerHTML = `
      <span class="jx-row">
        ${hasChildren ? `<button class="jx-toggle jx-open" aria-label="collapse"></button>` : `<span class="jx-leaf-dot"></span>`}
        <span class="jx-key">${esc(key)}</span>
        <span class="jx-sep">→</span>
        ${_jxClassLabel(value)}
      </span>`;
    if (hasChildren) {
      const children = buildJsonExplorer(inner, depth + 1, opts);
      li.appendChild(children);
      li.querySelector(".jx-toggle").addEventListener("click", _jxToggleHandler);
    }
  } else if (isArray) {
    const summary = `[${value.length}]`;
    li.innerHTML = `
      <span class="jx-row">
        ${value.length ? `<button class="jx-toggle jx-open" aria-label="collapse"></button>` : `<span class="jx-leaf-dot"></span>`}
        <span class="jx-key">${esc(key)}</span>
        <span class="jx-dim">${esc(summary)}</span>
      </span>`;
    if (value.length) {
      const children = buildJsonExplorer(value, depth + 1, opts);
      li.appendChild(children);
      li.querySelector(".jx-toggle").addEventListener("click", _jxToggleHandler);
    }
  } else if (isObject && value !== null) {
    const cn = value.class_name;
    const showLabel = cn && cn !== key;
    const childKeys = (opts.filterHidden
      ? Object.keys(value).filter(k => !HIDDEN_JSON_KEYS.has(k))
      : Object.keys(value)
    ).filter(k => !(cn && k === "class_name"));
    li.innerHTML = `
      <span class="jx-row">
        ${childKeys.length ? `<button class="jx-toggle jx-open" aria-label="collapse"></button>` : `<span class="jx-leaf-dot"></span>`}
        ${cn && !showLabel
          ? `<a href="#" class="jx-key jx-cls jmp-cls" data-cls="${esc(cn)}">${esc(key)}</a>`
          : `<span class="jx-key">${esc(key)}</span>`}
        ${showLabel ? _jxClassLabel(value) : ""}
      </span>`;
    if (childKeys.length) {
      const filtered = Object.fromEntries(childKeys.map(k => [k, value[k]]));
      const children = buildJsonExplorer(filtered, depth + 1, opts);
      li.appendChild(children);
      li.querySelector(".jx-toggle").addEventListener("click", _jxToggleHandler);
    }
  } else {
    const display = value === null ? "null" : String(value);
    li.innerHTML = `
      <span class="jx-row">
        <span class="jx-leaf-dot"></span>
        <span class="jx-key">${esc(key)}</span>
        <span class="jx-sep">:</span>
        <span class="jx-val">${esc(display)}</span>
      </span>`;
  }

  return li;
}

function _jxToggleHandler(e) {
  const btn = e.currentTarget;
  const li = btn.closest(".jx-item");
  const children = li.querySelector(":scope > ul");
  if (!children) return;
  const open = btn.classList.toggle("jx-open");
  children.style.display = open ? "" : "none";
}

function bindJxToggle(btn) {
  btn.addEventListener("click", _jxToggleHandler);
}

function buildParamsCard(entry) {
  if (!entry || !entry.params_tree) return "";
  // Check if anything is visible after hidden-key filtering
  const visibleKeys = Object.keys(entry.params_tree).filter(k => !HIDDEN_JSON_KEYS.has(k));
  if (!visibleKeys.length) return "";

  const explorer = buildJsonExplorer(entry.params_tree);
  const wrap = document.createElement("div");
  wrap.className = "dcard-body dcard-body--params";
  wrap.appendChild(explorer);

  return `
    <div class="dcard">
      <div class="dcard-hd"><span class="dcard-hd-label">Parameters</span></div>
      ${wrap.outerHTML}
    </div>`;
}



// ---------------------------------------------------------------------------
// Entries screen – load + render
// ---------------------------------------------------------------------------

async function loadDetail() {
  const data = await fetchDashboard({ ...buildPayload(), row_display: "none" });
  if (data.loading) { showLoadingOverlay(); return; }
  lastDashboard = { ...lastDashboard, ...data };
  state.selected_entry = data.selected_entry ?? null;
  renderDetail(data.detail);
}

// ---------------------------------------------------------------------------
// Loading overlay — polls /api/status until ready, then fires callback
// ---------------------------------------------------------------------------

let _loadingOverlayActive = false;

function showLoadingOverlay() {
  if (_loadingOverlayActive) return;
  _loadingOverlayActive = true;
  $("#loading-overlay").classList.remove("hidden");
  _pollUntilReady();
}

function hideLoadingOverlay() {
  _loadingOverlayActive = false;
  $("#loading-overlay").classList.add("hidden");
}

async function _pollUntilReady() {
  while (_loadingOverlayActive) {
    try {
      const status = await (await fetch("/api/status")).json();
      const p = status.progress ?? {};
      const done  = p.done  ?? 0;
      const total = p.total ?? 0;
      if (total > 0) {
        const pct = Math.round((done / total) * 100);
        $("#loading-bar-fill").style.width = `${pct}%`;
        $("#loading-label").textContent = "Loading…";
        $("#loading-sub").textContent = `${done} / ${total} entries`;
      } else {
        $("#loading-label").textContent = "Loading…";
        $("#loading-sub").textContent = "";
      }
      if (status.ready) {
        hideLoadingOverlay();
        loadEntriesOnly();
        return;
      }
    } catch {
      // server not up yet — keep polling
    }
    await new Promise((r) => setTimeout(r, 300));
  }
}

let _loadSeq = 0;

async function loadEntriesOnly() {
  const seq = ++_loadSeq;
  const data = await fetchDashboard(buildPayload());
  if (seq !== _loadSeq) return;  // superseded by a newer request
  if (data.loading) {
    showLoadingOverlay();
    return;
  }
  lastDashboard = data;

  state.selected_entry = data.selected_entry ?? null;
  _visibleEntryIds = data.visible_entry_ids ?? [];

  $("#topbar-counts").textContent =
    `${data.counts.entries} entries · ${data.counts.classes} classes`;
  $("#entry-count").textContent = `${data.counts.visible_entries} shown`;
  renderPills();
  renderClassList(data.class_cards);

  const opts = data.spec_options ?? {};
  renderSpecPills("#spec-crs",        opts.crs        ?? [], state.spec_filters.crs);
  renderSpecPills("#spec-resolution", opts.resolution ?? [], state.spec_filters.resolution);
  renderSpecPills("#spec-bounds",     opts.bounds     ?? [], state.spec_filters.bounds, (v) => {
    try { return boundsLatLonText(JSON.parse(v)) ?? v; } catch { return v; }
  });

  if (_viewMode === "compact") {
    renderEntryPills(data.table_rows);
  } else {
    renderTableHead();
    renderTableBody(data.table_rows);
  }
  renderDetail(data.detail);
}

async function loadEntries() {
  renderFilterRows();
  await loadEntriesOnly();
}




// ---------------------------------------------------------------------------
// Diagnostics modal
// ---------------------------------------------------------------------------

function showDiagnostics() {
  if (!lastDashboard) {
    toast("No data loaded yet");
    return;
  }

  const counts = lastDashboard.counts     ?? {};
  const diag   = lastDashboard.diagnostics ?? {};
  const opts   = lastDashboard.spec_options ?? {};

  const scanned        = diag.scanned_params_paths ?? 0;
  const created        = diag.created_entries      ?? 0;
  const resolverFails  = (diag.resolver_failures   ?? []).length;
  const missingHash    = (diag.missing_state_hash  ?? []).length;
  const derivedCls     = (diag.derived_class_name  ?? []).length;
  const hashCollisions = (diag.hash_collisions     ?? []).length;
  const tracedToClass  = created - derivedCls;
  const staleHidden    = diag.stale_hidden          ?? 0;

  const totalClasses   = counts.classes        ?? 0;
  const loadedClasses  = counts.classes_loaded ?? 0;
  const unloadedClasses = totalClasses - loadedClasses;

  function row(label, value, warn = false) {
    const cls = warn && value > 0 ? ' class="diag-warn"' : "";
    return `<tr${cls}>
      <td class="diag-label">${label}</td>
      <td class="diag-val">${value}</td>
    </tr>`;
  }

  openModal(
    "Registry diagnostics",
    `<table class="diag-table">
       <tbody>
         <tr class="diag-section-row"><td colspan="2">Classes</td></tr>
         ${row("Total",                           totalClasses)}
         ${row("In registry",                   loadedClasses)}
         ${row("Cache-only",                    unloadedClasses, true)}

         <tr class="diag-section-row"><td colspan="2">Scan</td></tr>
         ${row("Params files found",            scanned)}
         ${row("Entries created",               created)}
         ${row("Dropped (resolver error)",      resolverFails,  true)}
         ${row("Class from hash.json",          tracedToClass)}
         ${row("Class derived from filename",   derivedCls,     true)}

         <tr class="diag-section-row"><td colspan="2">Entry quality</td></tr>
         ${row("Missing state hash",            missingHash,    true)}
         ${row("Hash collisions (disambiguated)", hashCollisions, true)}
         ${staleHidden > 0 ? row("Stale entries hidden",       staleHidden,    false) : ""}

         <tr class="diag-section-row"><td colspan="2">Spec coverage (distinct values)</td></tr>
         ${row("CRS",        (opts.crs        ?? []).length)}
         ${row("Resolution", (opts.resolution ?? []).length)}
         ${row("Bounds",     (opts.bounds     ?? []).length)}
       </tbody>
     </table>`,
    "sm"
  );
}


// ---------------------------------------------------------------------------
// View mode (Compact / Detailed)
// ---------------------------------------------------------------------------

let _viewMode = "compact";

function applyViewMode(mode, reload = true) {
  _viewMode = mode;
  const layout = $(".app-layout");
  layout.classList.toggle("mode-compact",  mode === "compact");
  layout.classList.toggle("mode-detailed", mode === "detailed");

  // Restore saved detail pane width when entering detailed mode
  if (mode === "detailed") {
    const saved = localStorage.getItem("registry.detail-w");
    if (saved) $("#detail-pane").style.flex = `0 0 ${saved}px`;
  } else {
    $("#detail-pane").style.flex = "";
  }

  $$(".screen-tab").forEach((tab) =>
    tab.classList.toggle("active", tab.dataset.mode === mode)
  );

  // In compact mode, only header rows are needed (pills render from those).
  // In detailed mode, restore the display-seg selection.
  if (mode === "compact") {
    state.row_display = "none";
  } else {
    const active = $("#display-seg .seg-btn.active");
    state.row_display = active?.dataset.val ?? "all";
  }

  if (reload) loadEntriesOnly();
}


// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------

// Mode tabs (Compact / Detailed)
$$(".screen-tab").forEach((tab) => {
  tab.onclick = () => applyViewMode(tab.dataset.mode);
});

// Sidebar collapsible sections — accordion: at most one open at a time
$$(".sb-section-hd[data-target]").forEach((hd) => {
  hd.onclick = () => {
    const section = $("#" + hd.dataset.target);
    const isOpen = section.classList.contains("open");
    // Close all collapsible sections
    $$(".sb-section.collapsible").forEach((s) => s.classList.remove("open"));
    // If it wasn't open, open it
    if (!isOpen) section.classList.add("open");
  };
});


// Kind tabs
$$("#kind-tabs .kind-tab").forEach((tab) => {
  tab.onclick = () => {
    $$("#kind-tabs .kind-tab").forEach((t) =>
      t.classList.toggle("active", t === tab)
    );
    state.kind_filter = tab.dataset.kind;
    loadEntries();
  };
});

// Logic mode
$$("#logic-seg .seg-btn").forEach((btn) => {
  btn.onclick = () => {
    $$("#logic-seg .seg-btn").forEach((b) =>
      b.classList.toggle("active", b === btn)
    );
    state.logic_mode = btn.dataset.val;
    loadEntries();
  };
});

// Display mode (Selected params / All params — only active in Detailed view)
$$("#display-seg .seg-btn").forEach((btn) => {
  btn.onclick = () => {
    if (_viewMode === "compact") return;
    $$("#display-seg .seg-btn").forEach((b) =>
      b.classList.toggle("active", b === btn)
    );
    state.row_display = btn.dataset.val;
    loadEntries();
  };
});

// Initialise display state to "all" (Entries only is gone)
state.row_display = "all";

// Add filter
$("#add-filter").onclick = () => {
  state.filters.push({ target: "all", operator: "contains", value: "" });
  renderFilterRows();
  // focus the new row's value input
  const rows = $("#filter-rows").querySelectorAll(".fr-v");
  rows[rows.length - 1]?.focus();
};

// Spec pill clicks are wired in renderSpecPills() after each render.

// Diagnostics button
$("#btn-diag").onclick = showDiagnostics;

// Write source registry for all loaded classes

// Reload from disk
$("#btn-reload").onclick = async () => {
  try {
    await postRebuild();
    loadEntries();  // will poll until ready
  } catch (e) {
    toast(`Reload failed: ${e}`);
  }
};

// Show/hide zero-entry classes toggle
$("#btn-show-empty").onclick = (e) => {
  e.stopPropagation();
  _showEmptyClasses = !_showEmptyClasses;
  $("#btn-show-empty").classList.toggle("active", _showEmptyClasses);
  if (lastDashboard) renderClassList(lastDashboard.class_cards ?? []);
};

// Hide stale toggle
$("#btn-hide-stale").classList.toggle("active", state.hide_stale);
$("#btn-hide-stale").onclick = (e) => {
  e.stopPropagation();
  state.hide_stale = !state.hide_stale;
  localStorage.setItem("hide_stale", state.hide_stale);
  $("#btn-hide-stale").classList.toggle("active", state.hide_stale);
  loadEntries();
};

// Multi-select toggle — wired once to the static sidebar button
$("#btn-multi-select").onclick = (e) => {
  e.stopPropagation();
  _multiSelectEnabled = !_multiSelectEnabled;
  $("#btn-multi-select").classList.toggle("active", _multiSelectEnabled);
  // Collapsing to single-select: keep only the last selected class
  if (!_multiSelectEnabled && state.selected_classes.length > 1) {
    pushHistory(_viewMode);
    const last = state.selected_classes[state.selected_classes.length - 1];
    state.selected_classes = [last];
    state.selected_entry = null;
    loadEntries();
  }
};

function _syncUIAfterRestore(snap) {
  if (snap.view_mode && snap.view_mode !== _viewMode) applyViewMode(snap.view_mode, false);
  // Sync kind tabs
  $$("#kind-tabs .kind-tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.kind === state.kind_filter)
  );
  // Sync logic mode segment
  $$("#logic-seg .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.val === state.logic_mode)
  );
  // Sync display mode segment
  $$("#display-seg .seg-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.val === state.row_display)
  );
  loadEntries();
}

// Back / forward navigation
$("#btn-back").onclick = () => {
  const snap = navigateBack(_viewMode);
  if (snap) _syncUIAfterRestore(snap);
};

$("#btn-forward").onclick = () => {
  const snap = navigateForward(_viewMode);
  if (snap) _syncUIAfterRestore(snap);
};

// Clear all filters
$("#btn-clear-all").onclick = () => {
  pushHistory(_viewMode);
  state.selected_classes = [];
  state.spec_filters = { crs: [], resolution: [], bounds: [] };
  state.filters = [{ target: "all", operator: "contains", value: "" }];
  loadEntries();
};

// Detail pane resize
(function () {
  const handle = $("#resize-handle");
  const pane   = $("#detail-pane");
  const STORE_KEY = "registry.detail-w";

  const saved = localStorage.getItem(STORE_KEY);
  if (saved) pane.style.flex = `0 0 ${saved}px`;

  let startX, startW;

  handle.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    startW = pane.getBoundingClientRect().width;
    handle.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  });

  document.addEventListener("mousemove", (e) => {
    if (!handle.classList.contains("dragging")) return;
    const delta = startX - e.clientX;   // drag left = wider
    const newW  = Math.max(200, Math.min(window.innerWidth * 0.8, startW + delta));
    pane.style.flex = `0 0 ${newW}px`;
  });

  document.addEventListener("mouseup", () => {
    if (!handle.classList.contains("dragging")) return;
    handle.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem(STORE_KEY, pane.getBoundingClientRect().width);
  });
})();

// Incremental scroll: append more table rows when near the bottom
$("#table-scroll-wrap").addEventListener("scroll", () => {
  if (_viewMode !== "detailed" || _tableEntryCount === 0) return;
  const el = $("#table-scroll-wrap");
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 200) {
    _appendTableRows(PAGE_ENTRIES);
  }
}, { passive: true });

// ⌘[ / ⌘] — back / forward navigation
document.addEventListener("keydown", (e) => {
  if (!e.metaKey) return;
  if (e.key === "[") {
    e.preventDefault();
    const snap = navigateBack(_viewMode);
    if (snap) _syncUIAfterRestore(snap);
  } else if (e.key === "]") {
    e.preventDefault();
    const snap = navigateForward(_viewMode);
    if (snap) _syncUIAfterRestore(snap);
  }
});

// Arrow key navigation between entries
document.addEventListener("keydown", (e) => {
  if (!_visibleEntryIds.length) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();

  const ids = _visibleEntryIds;
  const cur = ids.indexOf(state.selected_entry);
  let next;
  if (e.key === "ArrowDown") next = cur === -1 ? 0 : Math.min(cur + 1, ids.length - 1);
  else                       next = cur === -1 ? ids.length - 1 : Math.max(cur - 1, 0);

  state.selected_entry = ids[next];
  applyTableSelection();

  // Scroll the active item into view — pill list in compact, table row in detailed
  if (_viewMode === "compact") {
    const activePill = $(`#entry-pill-list .entry-pill[data-entry="${CSS.escape(state.selected_entry)}"]`);
    activePill?.scrollIntoView({ block: "nearest" });
  } else {
    // Ensure the target entry is rendered before scrolling to it
    let activeRow = $(`#table-body tr.row-entry[data-entry="${CSS.escape(state.selected_entry)}"]`);
    if (!activeRow) {
      const targetIdx = _tableRows.findIndex(
        (r) => r.row_type === "header" && r.record_id === state.selected_entry
      );
      if (targetIdx !== -1) {
        // Count how many header rows precede it and render up to that point
        const headersNeeded = _tableRows.slice(0, targetIdx + 1).filter((r) => r.row_type === "header").length;
        while (_tableEntryCount < headersNeeded) _appendTableRows(PAGE_ENTRIES);
        applyTableSelection();
        activeRow = $(`#table-body tr.row-entry[data-entry="${CSS.escape(state.selected_entry)}"]`);
      }
    }
    activeRow?.scrollIntoView({ block: "nearest" });
  }

  loadDetail();
});


// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

// Apply initial layout classes without triggering a load (loadEntries does the first fetch)
applyViewMode("compact", false);
updateNavBtns();
loadEntries().catch((e) => toast(`Load error: ${e}`));
