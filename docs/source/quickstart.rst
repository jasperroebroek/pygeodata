Quickstart
==========

Install the package and point it at a processed-data directory:

.. code-block:: python

   import pygeodata as pgd

   pgd.set_config(path_data_processed='/path/to/cache')

Define a loader by subclassing :class:`~pygeodata.data.Data`:

.. code-block:: python

   from pygeodata.data import Data
   from pygeodata.processors.reprojection import Reprojector

   @dataclass
   class MyRaster(Data):
       year: int

       @property
       def processor(self):
           return Reprojector(src_path=f'/data/{self.year}.tif')

Process and load in one call:

.. code-block:: python

   spec = pgd.SpatialSpec.from_raster_file('reference.tif')
   da = pgd.load(MyRaster(year=2020), spec)   # returns xarray.DataArray


Concepts
--------

Data and the caching model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every :class:`~pygeodata.data.Data` subclass represents a single
deterministic processing step. The framework caches each output as a file on
disk and skips reprocessing when the cache is still valid.

Cache validity is determined by a **state hash** that combines:

- A **source-hierarchy hash** — a SHA-256 of the AST dump of the class and all
  its call and inheritance dependencies. Reformatting code without changing
  logic does not invalidate the cache.
- A **params hash** — a stable serialization of every constructor parameter,
  including nested :class:`~pygeodata.data.Data` instances (represented
  by their own state hashes).

The hash is written alongside the output as a ``.hash.json`` file. On the next
:func:`~pygeodata.base.process` call the saved hash is compared with the live
hash; a mismatch triggers reprocessing.

SpatialSpec
~~~~~~~~~~~

:class:`~pygeodata.types.SpatialSpec` is a frozen dataclass that bundles the
three components of a raster grid:

- ``crs`` — a :class:`pyproj.CRS` (default EPSG:4326)
- ``transform`` — an :class:`affine.Affine` mapping pixel → world coordinates
- ``shape`` — ``(height, width)`` in pixels

A spec is *fully defined* when both ``transform`` and ``shape`` are set. Some
loaders can resolve an underdefined spec (CRS only) by inspecting their source
data via :meth:`~pygeodata.data.Data.resolve_spec`.

Processors and Drivers
~~~~~~~~~~~~~~~~~~~~~~

A **processor** is any callable matching the :class:`~pygeodata.types.Processor`
protocol — ``(dst_path, spec) -> None``. The two built-in processors are:

- :class:`~pygeodata.processors.reprojection.Reprojector` — warps a raster to a target
  grid using ``rasterio.warp``.
- :class:`~pygeodata.processors.rasterizer.Rasterizer` — burns vector geometries to a
  single-band raster using ``rasterio.features.rasterize``.

A **driver** is any callable matching the :class:`~pygeodata.types.Driver`
protocol — ``(path) -> T``. Built-in drivers are:

- :class:`~pygeodata.drivers.RioXArrayDriver` — loads a GeoTIFF as an
  :class:`xarray.DataArray` (default for raster loaders).
- :class:`~pygeodata.drivers.GeoPandasDriver` — loads a vector file as a
  :class:`geopandas.GeoDataFrame`.
- :class:`~pygeodata.drivers.GeoPandasParquetDriver` — loads a GeoParquet
  file as a :class:`geopandas.GeoDataFrame`.

Parameter exclusion
~~~~~~~~~~~~~~~~~~~

Three class-level tuples control which constructor parameters participate in
hashing, path generation, and serialization:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15

   * - Attribute
     - Hash
     - Path
     - params.json
   * - ``_exclude_params``
     - ✗
     - ✗
     - ✗
   * - ``_exclude_params_from_path``
     - ✗
     - ✗
     - ✓

Use ``_exclude_params`` for purely operational parameters (e.g. thread counts)
that do not affect output content. Use ``_exclude_params_from_path`` for
parameters that are used when overwriting meth:`Data.get_processed_path`.

Path generation
~~~~~~~~~~~~~~~

Output paths are built by :func:`~pygeodata.paths.generate_path` and follow
the structure::

   <base_dir>/<crs>/<grid>/<ClassName>/<param=value ...>/<name>.<ext>

When the number of parameters exceeds ``max_path_param_depth`` (default 5,
configurable via :class:`~pygeodata.config.Config`), the parameter segment is
replaced with a single SHA-256 digest to keep paths short. This can be adjusted
in the configuration object.

Parallel execution
~~~~~~~~~~~~~~~~~~

:func:`~pygeodata.parallel.build_dask_graph` converts a loader and its
transitive parameter dependencies into a `Dask <https://dask.org>`_ delayed
graph, enabling parallel or distributed execution:

.. code-block:: python

   from dask.distributed import Client
   from pygeodata.parallel import build_dask_graph

   client = Client()
   graph = build_dask_graph(my_loader, spec=spec)
   graph.compute()

Cache management
~~~~~~~~~~~~~~~~

:func:`~pygeodata.cache.clean_cache` scans the processed-data directory and
removes files whose ``source_hierarchy_hash`` no longer matches the live code,
keeping only outputs that are still valid:

.. code-block:: python

   pgd.clean_cache(dry_run=True)   # preview deletions
   pgd.clean_cache(dry_run=False)  # apply
