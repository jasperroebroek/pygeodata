What's new
==========

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
- :class:`~pygeodata.types.SpatialSpec` — immutable raster grid descriptor
  (CRS, affine transform, shape).
- :func:`~pygeodata.base.load` and :func:`~pygeodata.base.process` top-level
  functions.

**Processors**

- :class:`~pygeodata.processors.reprojection.Reprojector` — warp any raster to
  a target :class:`~pygeodata.types.SpatialSpec` via ``rasterio.warp``. Supports
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

- Hash-based path layout: ``<path_cache>/<ClassName>/<hash(spec + params)>/<name>.<ext>``.
- ``_exclude_params`` class attribute for omitting operational parameters from
  the cache key entirely.
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
