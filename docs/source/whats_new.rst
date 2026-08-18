What's new
==========

Version 0.3.0
-------------

**New Table class**

- :class:`~pygeodata.table.Table` is a third artifact family, structurally a
  sibling of :class:`~pygeodata.figure.Figure` rather than
  :class:`~pygeodata.data.Data`: fixed ``ext = 'html'``, no ``load()``, no
  driver. A ``Table`` subclass's ``_process`` loads whatever ``Data`` it wants
  to summarise and writes the rendered HTML itself — typically
  ``df.to_html()`` — and that HTML file *is* the stored artifact.

- New ``path_tables`` config field (default ``Path('tables')``) — existing
  projects will see a new top-level ``tables/`` directory appear the first
  time a ``Table`` is processed. Repoint it with
  ``set_config(path_tables=...)`` like any other cache root, and add it to
  your ``.gitignore`` alongside ``data_processed/`` and ``figures/``.

- New :attr:`~pygeodata.config.Config.cache_roots` property — the single
  source of truth for "every root that holds cache entries," replacing four
  call sites that previously enumerated ``path_cache``/``path_figures`` by
  hand (or via a live-class walk). :func:`~pygeodata.cache.clean_cache`, the
  registry browser's file sandbox, and the entry-hash scanner all read it.

**Registry browser**

- The Entries kind-filter row gains a ``Table`` tab: ``All | Data | Table |
  Figure``. The Code view's kind-filter row gains a fifth ``Object`` tab for
  plain :class:`~pygeodata.tracked_object.TrackedObject` subclasses
  (processors, helpers) — Code-view only, since those classes have source
  history but never produce cache entries.
- ``object_type`` badges are now colour-coded per family (blue / green /
  mauve for Data / Table / Figure) everywhere they appear — the sidebar, the
  entries table, the entry detail card, and the Code view.
- The entry detail panel renders a ``Table``'s stored HTML directly, styled
  with the browser's own CSS (sticky header, row hover) rather than in an
  isolated frame, with a **Show** button to open it full-size in a modal —
  the same mechanism the existing figure preview uses for PNGs.
- **Fix:** an entry whose ``_process`` wrote a sidecar file beside its primary
  output (a CSV next to a rendered HTML table, say) had the sidecar silently
  discarded — only one output file was ever tracked per entry. Sidecar files
  are now surfaced in the detail panel beneath the primary file, which is
  shown in bold when sidecars are present.
- **Fix:** the co-outputs card's payload used the wrong dict key
  (``co_output_hashes`` instead of the ``co_outputs`` the frontend reads), so
  it has silently never rendered. Found while auditing the payload layer for
  an unrelated cleanup.

**CLI**

- ``pygeodata classes --type`` now accepts ``table`` alongside
  ``data``/``figure``/``trackedobject``; its help text lists all four values.
- ``pygeodata entry show`` lists sidecar files under the primary output.

No migration is needed for existing projects: ``Table`` is new, so no
existing entry changes location. Reclassifying an existing ``Data`` subclass
as a ``Table`` gets a new ``state_hash`` and a fresh computation into
``tables/``, leaving the old entry orphaned under ``data_processed/`` for
``clean-cache`` to collect — the normal behaviour for any class change.

Version 0.2.1
-------------

**Class-valued parameters**

- A :class:`~pygeodata.tracked_object.TrackedObject` subclass passed as a
  *parameter* is invisible to AST dependency discovery, so it never enters the
  entry's dependency tree. Its identity now travels through the params channel:
  ``format_json`` emits ``{class_name, dependency_tree_hash}`` for tracked
  classes, so the instance hash moves when the referenced class changes and the
  cache correctly recomputes.

- :func:`~pygeodata.cache.clean_cache` gained ``check_class_params``
  (CLI: ``--check-class-params``), which validates those class-valued params
  against ``parameters.json``. Without it the diagnostic reports such an entry
  as current while the runtime rebuilds it — the disagreement that this flag
  closes. Opt-in: it reads a second file per entry, and an entry whose
  ``parameters.json`` is missing or unreadable counts as stale.

  Entries written before this change store a bare dotted name for class-valued
  params, carrying no hash, and cannot be validated retroactively — one rebuild
  is required before the check is meaningful for them.

Version 0.2.0
-------------

**Instance identity moved to TrackedObject**

- :class:`~pygeodata.tracked_object.TrackedObject` now owns the instance
  parameter and identity machinery — ``get_params()``, ``get_params_hash()``,
  ``get_instance_hash()``, ``format_as_json()`` and ``_sort_params`` — which
  previously lived on :class:`~pygeodata.artifact.Artifact`.
  :class:`~pygeodata.artifact.Artifact` keeps the layer that produces a cached
  file: cache roots, extensions, and ``process()``.

  This lets a value type that produces no output file subclass
  :class:`~pygeodata.tracked_object.TrackedObject` directly and still get source
  hashing, identity hashes, and registry presence, without supplying a fictional
  ``get_cache_root``.

  Existing hashes are unchanged — framework classes are not part of any
  dependency tree, so relocating these methods does not affect cache keys.

- New ``get_instance_state()`` hook returns the dict hashed by
  ``get_instance_hash()``. :class:`~pygeodata.artifact.Artifact` overrides it to
  contribute the processor source hash, so
  :class:`~pygeodata.tracked_object.TrackedObject` carries no processor concept.

- The runtime execution graph is now written when a parameter is an
  :class:`~pygeodata.artifact.Artifact`, rather than any
  :class:`~pygeodata.tracked_object.TrackedObject`. Previously a parameter
  holding only non-artifact tracked values would trigger an edgeless graph
  render on every cache write.

**Drivers**

- :class:`~pygeodata.drivers.pandas.PandasDriver` for loading plain
  (non-geospatial) tables via CSV, Parquet, or Excel.

**Fixes**

- ``clean-cache`` no longer crashes on Python 3.10 and 3.11. Cache traversal
  used ``Path.walk``, which requires Python 3.12, while the package declares
  ``requires-python = ">=3.10"``.

Version 0.1.0
-------------

Initial release.

**Core framework**

- :class:`~pygeodata.data.Data` base class for defining cached geospatial
  loaders with automatic cache invalidation based on AST hashing.
- :class:`~pygeodata.figure.Figure` base class for cached plot outputs, stored
  separately under ``path_figures``.
- :class:`~pygeodata.artifact.Artifact` base class providing the shared hashing
  and parameter serialisation machinery.
- :class:`~pygeodata.spec.SpatialSpec` — immutable raster grid descriptor
  (CRS, affine transform, shape).
- :func:`~pygeodata.base.load` and :func:`~pygeodata.base.process` top-level
  functions.

**Processors**

- :class:`~pygeodata.processors.reprojection.Reprojector` — warp any raster to
  a target :class:`~pygeodata.spec.SpatialSpec` via ``rasterio.warp``. Supports
  scale/offset application, forced-read for multi-variable NetCDF, source CRS
  override, and all rasterio resampling algorithms.
- :class:`~pygeodata.processors.rasterizer.Rasterizer` — burn vector geometries
  to a raster. Supports constant values, column attributes, row-index burning,
  and dynamic vector loading via ``load_df``.

**Drivers**

- :class:`~pygeodata.drivers.RioXArrayDriver` — returns an
  :class:`xarray.DataArray` (default for raster loaders).
- :class:`~pygeodata.drivers.GeoPandasDriver` — returns a
  :class:`geopandas.GeoDataFrame` from any OGR-readable format.
- :class:`~pygeodata.drivers.GeoPandasParquetDriver` — returns a
  :class:`geopandas.GeoDataFrame` from GeoParquet.

**Caching and paths**

- Hash-based path layout: ``<path_cache>/<state_hash>/<name>.<ext>``.
- Operational parameters can be omitted from the cache key by storing them as
  underscore-prefixed attributes (``get_params`` skips them).
- Co-output pattern: ``_process`` can yield sibling loaders to write multiple
  outputs in a single run.
- :func:`~pygeodata.cache.clean_cache` and
  :func:`~pygeodata.cache.purge_unregistered_cache` for cache maintenance.

**Graphs and parallelism**

- :func:`~pygeodata.graphs.plot_compact_execution_graph` — render the runtime
  dependency graph of a loader instance.
- :func:`~pygeodata.graphs.plot_class_dependency_graph` — render the static
  class-level dependency graph.
- :func:`~pygeodata.parallel.build_dask_graph` — convert a loader graph to a
  Dask delayed graph for parallel or distributed execution.

**Registry browser**

- :func:`~pygeodata.registry_browser.open_registry_browser` — local Flask web
  UI for exploring cached entries, inspecting parameters and state hashes,
  viewing co-outputs, navigating dependency graphs, and opening output files.
