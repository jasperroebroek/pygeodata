# Handover — pygeodata browser rebuild & SQLite catalog

This document hands over the pygeodata browser-rebuild effort. Buckets 1 and 2 are
complete and committed; the remaining buckets are specified but not started. Read the
whole file before continuing — the **decisions and contracts** sections are binding;
they came out of an explicit design interview and were verified at every checkpoint.

## Goal

The browser is used for **QC, figures, and code review** of big projects (e.g.
`~/Projects/apes`, ~10.7k cache entries). The rebuild separates the browser into its
own installable package (Svelte 5 frontend), backed by a **catalog SQLite DB**
(`.source/catalog.db`) that indexes processed features so import stops re-reading the
filesystem. The core data-processing codebase is untouched except for the single
catalog write hook.

## Decisions (binding)

1. **`state_hash` is the identity.** `state_hash = hash(instance_hash, spec)` where
   `instance_hash = hash(dep-tree, params, processor source)`. Presence of a
   well-formed `meta.json` = proof the entry was rendered from exactly those inputs.
   **No mtime-based validation anywhere.** `created_at` (meta.json mtime) is used for
   ordering only, never validity.
2. **Catalog is an index of proven proofs, never a source of truth.** Rows exist only
   because `meta.json` was written successfully. The disk JSON files stay the proof.
3. **DB lives at `.source/catalog.db`**, per project. No cross-project queries.
   Concurrency relies on the existing per-entry `process.lock` files — do not add new
   locking. WAL mode + `INSERT OR REPLACE` keyed by content hashes make shared-dir
   writers converge.
4. **One FORMAT_VERSION** (pygeodata's, `config.py`). The catalog stores it once in
   `meta`. No per-row format version. On mismatch → wipe → rebuild.
5. **No schema version numbers.** `store.schema_fingerprint()` hashes the compiled
   `CREATE TABLE` DDL; any model change changes the fingerprint → automatic
   drop/recreate → backfill rebuilds. The catalog is disposable, never migrated.
6. **Spec dimensions are fully normalized** into per-concern tables keyed by content
   hash: `crs` (hash of CRS string), `resolutions` (hash of transform + coefficients +
   res), `bounds` (hash of (transform, shape) **together** — identical bounds from
   different specs stay separate rows, per explicit decision), `shapes`.
7. **`entry_files` is one row per file** with a `FileRole` enum (`PRIMARY`/`SIDECAR`),
   PK `(state_hash, relpath)`. No sidecar JSON lists.
8. **No per-entry file reads at import.** Param rows (`flatten_params` output) and
   linked entries are stored on `entries` as JSON columns; files come from
   `entry_files`; spec display is reconstructed from the dimension tables (only
   distinct (crs, transform, shape) triples reproject, via memoized
   `SpecInfo.from_spec`). `parameters.json` is read only lazily for a single entry's
   detail popup (`_params_tree` in `payloads.py`).
9. **`rows`/`linked_entries` are keyed per entry** (`state_hash`), not by
   `params_hash` — `parameters.json` is spec-dependent because data-valued params
   embed their `state_hash` (`Artifact.format_as_json`).
10. **Browser startup:** default **blocks** until the first full state is ready
    (loading bar). `pygeodata browse --fast` serves the UI immediately from
    `/api/bootstrap` (latest entry per class, ~0.2s) with a slim progress bar pinned
    under the top row; full state swaps in when ready. A daemon thread reconciles
    (backfill + prune + optional Python reimport) every
    `--reconcile-interval` minutes (default 5), reimport on by default
    (`--no-reimport-on-reconcile` to disable). Manual reload buttons stay.

## Architecture (as implemented)

```
pygeodata/catalog/
  models.py       SQLAlchemy models — single source of truth for the schema
  dimensions.py   SpatialSpec → dimension rows + content hashes
  display.py      EntryDisplay derivation (rows/linked/files) + file finding
                  (moved here from entry_catalog; reused by both)
  store.py        Engine (WAL), fingerprint sync, upserts, incremental backfill,
                  prune, refresh, bootstrap_rows, index_entry (write hook target)
pygeodata/artifact.py
  write_cache_metadata  ends with index_entry(...) — data only (record, resolved
                   spec, params_json); catalog derives display internally
pygeodata/registries/registry.py
  EntryRegistry   hydrates from the catalog (one SELECT of identity columns);
                  falls back to a plain meta.json disk scan when no/empty catalog
pygeodata/catalog/entry_catalog.py
  discover_entries  assembles EntryInfo entirely from the DB; entry dirs are
                   resolved to absolute paths (send_file requirement); disk
                   fallback per entry only when the catalog is missing
~/Packages/pygeodata-browser/  (separate installable package, own git repo)
  pygeodata_browser/  the Flask browser (web, payloads, code/export services,
                   state, templates, static JS/CSS) + its tests; installs with
                   `pip install pygeodata-browser` or `pygeodata[browser]`
```

**Contract details:**
- Write hook: `store.index_entry(source_root, entry_dir, record, spec, params_json)`
  — warn-and-continue; a failed catalog write never fails a render.
- `store.refresh(source_root, cache_roots, format_version)` = incremental backfill +
  prune; the browser's load/reload threads and `catalog build` share it.
- Backfill is **incremental**: already-indexed state hashes are skipped (entry dirs
  are named by state hash, so the flat layout skips without reading `meta.json`).
- `store.schema_fingerprint()` + `meta.format_version` gate every engine open.
- Reimport's practical effect on the UI is just the live-class dict (class names +
  source/dep-tree hashes + object types from `TrackedObject._registry`); everything
  else the UI reads comes from `.source` on disk or the catalog. Svelte can consume a
  class-hash payload and reactively rerender — confirmed by audit.
- Inline (function-level) imports are **not allowed** except with a documented
  reason; the only two remaining are cycle-avoidance in `store.index_entry` /
  `store.backfill` (`display` imports the artifact families) and the pre-existing
  CLI/browser lazy imports.

## Performance (measured on ~/Projects/apes, 10,741 entries / 142 classes)

| Operation | Before | After |
|---|---|---|
| `catalog build` (full, one-time) | — | ~80s |
| Reconcile, nothing new | full re-read (~60s) | ~1.0s |
| `EntryRegistry()` | 4.0s (disk scan) | 0.38s |
| Full `build_state` | ~20s | ~6.7s |
| `/api/bootstrap` payload | — | ~0.19s |

The remaining ~6.7s is dominated by JSON-decoding the 32k stored param rows. The
structural fix is per-class/per-view queries — that is the Svelte app's job (bucket 4),
not more caching.

## Working agreements (follow exactly)

- Format/lint with base-env ruff: `/usr/local/Caskroom/mambaforge/base/bin/ruff format .`
  and `.../ruff check .`. Run after every change.
- Tests with the mamba `pygeodata` env:
  `/usr/local/Caskroom/mambaforge/base/envs/pygeodata/bin/python -m pytest -q --ignore=tests/processors`
  (include `tests/processors` only with reason / before final commits).
- 7 test failures are pre-existing and environmental: Graphviz `dot` is not on PATH
  (`ExecutableNotFound`). They fail identically on the parent commit. Do not chase
  them; do not count them as regressions.
- Be strict on comments/docstrings: only non-obvious intent, never restating code.
- No inline imports without a specific, documented reason.
- Don't reimplement existing helpers — before writing any derivation/scanning/serialization,
  check the codebase first (this bit us repeatedly; `SpatialSpec.*`, `CachePathResolver`,
  `EntryRecord.from_file`, `flatten_params`, conftest fixtures were all reused after review).
- The user reviews at bucket checkpoints; present a plan first, stop after the bucket.
- `apes` has `.source/catalog.db` built and current. Remaking it:
  `cd ~/Projects/apes && pygeodata catalog build` (or delete `catalog.db*` — it rebuilds
  on next open; schema changes auto-wipe via the fingerprint).

## Remaining buckets (in order)

### Bucket 3 — Package split (COMPLETE)
Done: browser lives in the sibling package `~/Packages/pygeodata-browser`
(module `pygeodata_browser`, deps `pygeodata>=0.3` + `flask`, own repo, tests
moved with it). Core `browse` CLI delegates to `pygeodata_browser` with an
install hint (`pygeodata[browser]` extra). The legacy `pygeodata/registry_browser`
and its tests are removed from core; the `dashboard` extra is gone. The
structured-diff helper tests (`_word_segments`, `_build_structured_hunks`) moved
to the browser package (`tests/test_code_diff_helpers.py`). Verified
screen-by-screen with Playwright against a synthetic fixture project AND the
real `apes` project (normal + fast mode, legacy vs new): all views identical
(screenshots pixel-equal, API payloads byte-equal modulo per-process version
UUIDs). One shared bug fixed in core during verification: catalog-hydrated
EntryInfo paths are now absolute (regression test in
`tests/catalog/test_entry_catalog.py`).

### Bucket 4 — Svelte 5 frontend
Sub-bucket roughly: (a) Vite+Svelte scaffold against the existing JSON API
(`/api/dashboard`, `/api/status`, `/api/bootstrap`, popups, file endpoints);
(b) class column; (c) table column **including: clicking a class on the left
auto-opens the first table row** (explicit user requirement, currently unimplemented);
(d) detail/figure/table viewers, code view, export view.
- Fast mode becomes truly reactive: bootstrap payload renders instantly, full state
  merges in when ready, top-row progress bar for background reconciles. The periodic
  reconcile already reimport-refreshes the live-class dict; Svelte just diffs the
  class-hash payload.
- The per-class queries should remove the remaining 6.7s eager build (serve class
  detail on demand from the catalog via SQLAlchemy).

### Bucket 5 — Docs & release
- `docs/source/registry_architecture.rst` and `cli.rst` are partially updated; finish
  the catalog section (schema, contracts, fingerprint story).
- whats_new entry, version bumps (core 0.3.0 → next), README mention of
  `pygeodata[browser]`.

## Known loose ends

- The legacy JS fast-mode implementation (now in `pygeodata-browser`) is pragmatic
  (bootstrap list into the class column, top bar, swap on ready). It is expected
  to be replaced by bucket 4; don't polish it.
- `catalog/filters.py`, `catalog/export.py`, `class_catalog.py` were not refactored;
  they operate on the same `EntryInfo` view and keep working.
- `tests/processors` was skipped throughout (per CLAUDE.md default); run it once
  before the final release commit.
- The 7 Graphviz-`dot` failures are the only red tests on this tree.
