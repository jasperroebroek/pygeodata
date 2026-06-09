/**
 * detail.js
 *
 * Entry detail pane + JSON explorer.
 */

import { $, esc, badge, boundsLatLonText, _RASTER_EXTS, _fileExt, buildBoundsMapUrl, toast, lastDashboard } from './utils.js';
import { state, BOOLEAN_TARGETS } from './state.js';
import { fileURL, fetchJsonPopup, fetchSourcePopup, fetchGraphPopup, openPath, revealPath } from './api.js';

// ---------------------------------------------------------------------------
// JSON explorer — shared renderer for params card and JSON file popups
// ---------------------------------------------------------------------------

export const HIDDEN_JSON_KEYS = new Set([
  "class_name", "object_type", "source_hash", "state_hash",
  "dependency_tree_hash", "tree", "call_dependencies",
  "inheritance_dependencies", "dependencies", "crs", "transform",
]);

export function buildJsonExplorer(data, depth = 0, opts = { filterHidden: true }, knownClasses = null) {
  const ul = document.createElement("ul");
  ul.className = `jx-list jx-depth-${depth}`;

  if (Array.isArray(data)) {
    data.forEach((item, idx) => {
      ul.appendChild(_jxItem(String(idx), item, depth, opts, knownClasses));
    });
  } else if (data !== null && typeof data === "object") {
    Object.entries(data).forEach(([k, v]) => {
      if (opts.filterHidden && depth === 0 && HIDDEN_JSON_KEYS.has(k)) return;
      ul.appendChild(_jxItem(k, v, depth, opts, knownClasses));
    });
  }

  return ul;
}

function _jxClassLabel(obj) {
  const cn = obj?.class_name;
  return cn ? `<a href="#" class="jx-cls jmp-cls" data-cls="${esc(cn)}">${esc(cn)}</a>` : "";
}

function _jxKeyEl(key, knownClasses) {
  // Render a key as a class link if it's a known class name, otherwise plain span.
  return knownClasses?.has(key)
    ? `<a href="#" class="jx-key jx-cls jmp-cls" data-cls="${esc(key)}">${esc(key)}</a>`
    : `<span class="jx-key">${esc(key)}</span>`;
}

function _jxItem(key, value, depth, opts = { filterHidden: true }, knownClasses = null) {
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
        ${_jxKeyEl(key, knownClasses)}
        <span class="jx-sep">→</span>
        ${_jxClassLabel(value)}
      </span>`;
    if (hasChildren) {
      const children = buildJsonExplorer(inner, depth + 1, opts, knownClasses);
      li.appendChild(children);
      li.querySelector(".jx-toggle").addEventListener("click", _jxToggleHandler);
    }
  } else if (isArray) {
    const summary = `[${value.length}]`;
    li.innerHTML = `
      <span class="jx-row">
        ${value.length ? `<button class="jx-toggle jx-open" aria-label="collapse"></button>` : `<span class="jx-leaf-dot"></span>`}
        ${_jxKeyEl(key, knownClasses)}
        <span class="jx-dim">${esc(summary)}</span>
      </span>`;
    if (value.length) {
      const children = buildJsonExplorer(value, depth + 1, opts, knownClasses);
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
          : _jxKeyEl(key, knownClasses)}
        ${showLabel ? _jxClassLabel(value) : ""}
      </span>`;
    if (childKeys.length) {
      const filtered = Object.fromEntries(childKeys.map(k => [k, value[k]]));
      const children = buildJsonExplorer(filtered, depth + 1, opts, knownClasses);
      li.appendChild(children);
      li.querySelector(".jx-toggle").addEventListener("click", _jxToggleHandler);
    }
  } else {
    const display = value === null ? "null" : String(value);
    li.innerHTML = `
      <span class="jx-row">
        <span class="jx-leaf-dot"></span>
        ${_jxKeyEl(key, knownClasses)}
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

export function bindJxToggle(btn) {
  btn.addEventListener("click", _jxToggleHandler);
}


// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

export function openModal(title, html, size = "", { codeNav = false } = {}) {
  const card = $("#modal").querySelector(".modal-card");
  card.classList.toggle("modal-card--sm", size === "sm");
  $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = html;
  $("#modal-body").dataset.codeNav = codeNav ? "1" : "";
  $("#modal").classList.add("open");
}

export function closeModal() {
  $("#modal").classList.remove("open");
  $("#modal-body").innerHTML = "";
  delete $("#modal-body").dataset.codeNav;
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
    _navigateToCodeClass(cn);
    return;
  }

  const jmpCls = e.target.closest(".jmp-cls");
  if (jmpCls) {
    e.preventDefault();
    const cn = jmpCls.dataset.cls;
    if (!cn) return;
    const isCodeNav = $("#modal-body").dataset.codeNav === "1";
    // Inherit dep_hash from the popup wrapper if present — routes to the correct
    // code version (same bracket the "Source" button on that entry would use).
    const wrap = jmpCls.closest(".jx-popup-wrap");
    const depHash = jmpCls.dataset.depHash || wrap?.dataset.depHash || null;
    closeModal();
    if (isCodeNav) _navigateToCodeClass(cn, depHash);
    else _toggleClass(cn, { navigate: true });
  }
});


// ---------------------------------------------------------------------------
// bindZoom
// ---------------------------------------------------------------------------

export function bindZoom(img) {
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
// File action buttons
// ---------------------------------------------------------------------------

export function bindFileActions(root) {
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
          const knownClasses = _knownClassSet();
          const el = document.createElement("div");
          el.className = "jx-popup-wrap";
          if (btn.dataset.depHash) el.dataset.depHash = btn.dataset.depHash;
          el.appendChild(buildJsonExplorer(data.json, 0, { filterHidden: false }, knownClasses));
          openModal(data.title, el.outerHTML, "", { codeNav: true });
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

  root.querySelectorAll(".js-nodes").forEach((btn) => {
    btn.onclick = async () => {
      const path = btn.dataset.path;
      try {
        const data = await fetchJsonPopup(path);
        const nodes = data.json?.nodes ?? data.json;
        const knownClasses = _knownClassSet();
        const el = document.createElement("div");
        el.className = "jx-popup-wrap";
        if (btn.dataset.depHash) el.dataset.depHash = btn.dataset.depHash;
        el.appendChild(buildJsonExplorer(nodes, 0, { filterHidden: false }, knownClasses));
        openModal(`${btn.dataset.cls} — Nodes`, el.outerHTML, "", { codeNav: true });
        $("#modal-body").querySelectorAll(".jx-toggle").forEach(bindJxToggle);
      } catch {
        toast("Nodes unavailable");
      }
    };
  });

  root.querySelectorAll(".js-tree").forEach((btn) => {
    btn.onclick = async () => {
      const path = btn.dataset.path;
      try {
        const data = await fetchJsonPopup(path);
        const tree = data.json?.tree ?? data.json;
        const knownClasses = _knownClassSet();
        const el = document.createElement("div");
        el.className = "jx-popup-wrap";
        if (btn.dataset.depHash) el.dataset.depHash = btn.dataset.depHash;
        el.appendChild(buildJsonExplorer(tree, 0, { filterHidden: false }, knownClasses));
        openModal(`${btn.dataset.cls} — Tree`, el.outerHTML, "", { codeNav: true });
        $("#modal-body").querySelectorAll(".jx-toggle").forEach(bindJxToggle);
      } catch {
        toast("Tree unavailable");
      }
    };
  });

  root.querySelectorAll(".js-src-nav").forEach((btn) => {
    btn.onclick = () => _navigateToCodeClass(btn.dataset.cls, btn.dataset.depHash || null);
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


// ---------------------------------------------------------------------------
// Action button group  (Open / Reveal / Copy / Popup)
// ---------------------------------------------------------------------------

export function actionsHTML(path) {
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
// Detail pane rendering
// ---------------------------------------------------------------------------

export function renderNoEntryPlaceholder() {
  const el = $("#entry-detail");
  const selected = state.selected_classes;

  if (!selected.length) {
    el.innerHTML = `<div class="detail-empty">Select an entry to view details.</div>`;
    return;
  }

  if (selected.length === 1) {
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

export function renderDetail(detail) {
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
    if (cls) { e.preventDefault(); _navigateToCodeClass(cls.dataset.cls, cls.dataset.depHash || null); return; }
    const entry = e.target.closest(".jmp-entry");
    if (entry) { e.preventDefault(); _selectEntry(entry.dataset.entry, entry.dataset.cls || null); return; }
  };
}

function buildClassCard(detail) {
  const entryDepHash = detail.selected_entry?.dep_hash ?? '';
  function depLinks(names) {
    return (names ?? []).length
      ? (names).map((n) => `<a href="#" class="jmp-cls" data-cls="${esc(n)}" data-dep-hash="${esc(entryDepHash)}">${esc(n)}</a>`).join(", ")
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
      ? [`<button class="act-btn js-src-nav" data-cls="${esc(detail.class_name)}" data-dep-hash="${esc(detail.selected_entry?.dep_hash ?? '')}" title="${esc(srcTitle)}">Source</button>`]
      : []),
    ...(detail.graph_available
      ? [`<button class="act-btn js-graph" data-cls="${esc(detail.class_name)}" data-graph-path="${esc(detail.class_graph_path ?? '')}" title="${esc(graphTitle)}">Graph</button>`]
      : []),
    ...(detail.class_tree_path
      ? [`<button class="act-btn js-tree" data-cls="${esc(detail.class_name)}" data-path="${esc(detail.class_tree_path)}" data-dep-hash="${esc(entryDepHash)}" title="View dependency tree JSON">Tree</button>`]
      : []),
    ...(detail.class_tree_path
      ? [`<button class="act-btn js-nodes" data-cls="${esc(detail.class_name)}" data-path="${esc(detail.class_tree_path)}" data-dep-hash="${esc(entryDepHash)}" title="View nodes JSON">Nodes</button>`]
      : []),
    ...(detail.class_registry_path
      ? [`<button class="act-btn js-popup" data-path="${esc(detail.class_registry_path)}" data-dep-hash="${esc(entryDepHash)}" title="View registry JSON">Registry</button>`]
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
        <span class="dcard-hd-title">${esc(detail.class_name)}${typeBadge ? `<span class="dcard-hd-type">${typeBadge}</span>` : ""}${statusBadges ? `<span class="dcard-hd-sep"></span>${statusBadges}` : ""}</span>
        ${actions ? `<span class="dcard-hd-spacer"></span><div class="dcard-hd-actions">${actions}</div>` : ""}
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
    ? `<span class="dcard-hd-spacer"></span><div class="dcard-hd-actions">${entryActionParts.join('<span class="dcard-hd-sep"></span>')}</div>`
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
        <span class="dcard-hd-spacer"></span>
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
    ? `<button class="act-btn co-params-toggle" data-card="${cardId}">Show all params</button>`
    : "";

  return `
    <div class="dcard" id="${cardId}">
      <div class="dcard-hd">
        <span class="dcard-hd-label">Co-outputs</span>
        <span class="dcard-hd-spacer"></span>
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
// Compact view – entry pill list
// ---------------------------------------------------------------------------

export function renderEntryPills(rows) {
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
          ${tinyHash ? `<span class="entry-pill-hash">${esc(tinyHash)}</span>` : ""}
          ${badge(r.object_type, "badge-neutral")}
          ${flags}
        </span>
      </div>`;
  }).join("");

  el.onclick = (e) => {
    const pill = e.target.closest(".entry-pill");
    if (pill) _selectEntry(pill.dataset.entry);
  };
}


// Build a Set of all known class names from the last loaded dashboard.
function _knownClassSet() {
  const cards = lastDashboard?.class_cards ?? [];
  return cards.length ? new Set(cards.map((c) => c.class_name)) : null;
}

// Lazy references — set by entries.js at init time.
let _navigateToCodeClass = () => {};
let _selectEntry = () => {};
let _toggleClass = () => {};

export function setDetailActions(navigateToCodeClass, selectEntry, toggleClass) {
  _navigateToCodeClass = navigateToCodeClass;
  _selectEntry = selectEntry;
  _toggleClass = toggleClass;
}
