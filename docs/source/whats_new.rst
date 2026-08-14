What's new
==========

Version 0.1.2
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
