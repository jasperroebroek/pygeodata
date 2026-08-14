Quickstart
==========

Installation and configuration
-------------------------------

Install pygeodata and point it at a cache directory:

.. code-block:: python

   import pygeodata as pgd

   pgd.get_config().update(path_cache='data/processed')

Use :func:`~pygeodata.config.set_config` as a context manager when you only
want a setting to apply temporarily:

.. code-block:: python

   with pgd.set_config(num_threads=4):
       pgd.process(loader, spec)

Define a loader
---------------

Subclass :class:`~pygeodata.data.Data` and attach a ``processor``:

.. code-block:: python

   from dataclasses import dataclass
   from pygeodata import Data
   from pygeodata.processors.reprojection import Reprojector

   @dataclass
   class ElevationLoader(Data):
       src: str = 'data/elevation.tif'

       @property
       def processor(self):
           return Reprojector(src_path=self.src)

Process and load
----------------

:func:`~pygeodata.api.load` processes the data (if not already cached) and
returns it via the loader's driver (default: ``xarray.DataArray``):

.. code-block:: python

   spec = pgd.SpatialSpec.from_raster_file('reference.tif')
   da = pgd.load(ElevationLoader(), spec)

Call :func:`~pygeodata.api.process` explicitly if you only want to populate
the cache without loading the result into memory:

.. code-block:: python

   pgd.process(ElevationLoader(), spec)

Both calls are idempotent — they skip work when the cache is still valid.

Concepts
--------

Data and the caching model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Every :class:`~pygeodata.data.Data` subclass represents a deterministic
processing step. The framework caches each output as a file on disk and skips
reprocessing when the cache is still valid.

Cache validity is determined by a **state hash** that combines:

- A **dependency tree hash** — a SHA-256 of the AST of the class and all its
  call and inheritance dependencies. Reformatting code without changing logic
  does *not* invalidate the cache.
- The serialised parameter values — including nested
  :class:`~pygeodata.data.Data` instances, represented by their own state
  hashes.

The hash is written alongside the output in the ``meta.json`` file. On the next
:func:`~pygeodata.api.process` call the saved hash is compared with the live
hash; a mismatch triggers reprocessing.

SpatialSpec
~~~~~~~~~~~

:class:`~pygeodata.spec.SpatialSpec` is a frozen dataclass that bundles the
three components of a raster grid:

- ``crs`` — a :class:`pyproj.CRS` (required)
- ``transform`` — an :class:`affine.Affine` mapping pixel → world coordinates
  (optional)
- ``shape`` — ``(height, width)`` in pixels (optional)

A spec is *fully defined* when both ``transform`` and ``shape`` are set. Some
loaders can resolve an underdefined spec (CRS only) by inspecting their source
data via :meth:`~pygeodata.data.Data.resolve_spec`.

Processors and drivers
~~~~~~~~~~~~~~~~~~~~~~

A **processor** is any callable matching the :class:`~pygeodata.protocols.Processor`
protocol — ``(dst_path, spec) -> None``. The two built-in processors are:

- :class:`~pygeodata.processors.reprojection.Reprojector` — warps a raster to a
  target grid using ``rasterio.warp``.
- :class:`~pygeodata.processors.rasterizer.Rasterizer` — burns vector geometries
  to a single-band raster using ``rasterio.features.rasterize``.

Override ``_process(self, spec)`` directly for arbitrary logic.

A **driver** is any callable matching the :class:`~pygeodata.protocols.Driver`
protocol — ``(path) -> T``. Built-in drivers:

- :class:`~pygeodata.drivers.RioXArrayDriver` — loads a GeoTIFF as an
  :class:`xarray.DataArray` (default for raster loaders).
- :class:`~pygeodata.drivers.GeoPandasDriver` — loads a vector file as a
  :class:`geopandas.GeoDataFrame`.
- :class:`~pygeodata.drivers.GeoPandasParquetDriver` — loads a GeoParquet
  file as a :class:`geopandas.GeoDataFrame`.

Override ``_load(self, path)`` to return any Python object from a cached file.

Excluding parameters
~~~~~~~~~~~~~~~~~~~~

:meth:`~pygeodata.tracked_object.TrackedObject.get_params` discovers parameters from
``vars(self)`` and skips any attribute whose name starts with an underscore.
Store purely operational values (thread counts, verbosity flags) that must not
affect output content as underscore-prefixed attributes, and they are excluded
from the cache key entirely — the output path, the state hash, and the
``parameters.json`` file:

.. code-block:: python

   @dataclass
   class ElevationLoader(Data):
       src: str = 'data/elevation.tif'
       _n_jobs: int = 4

       @property
       def processor(self):
           return Reprojector(src_path=self.src)

   # Setting a different _n_jobs does not create a new cache entry.

Because dataclass fields with a leading underscore are awkward to pass
positionally, a common alternative is to assign the operational value in
``__post_init__`` (e.g. ``self._n_jobs = n_jobs``) or read it from
:func:`~pygeodata.config.get_config` instead of storing it as a parameter.

Path generation
~~~~~~~~~~~~~~~

Output paths follow the structure::

   <path_cache>/<state_hash>/<name>.<ext>

where ``state_hash`` is the SHA-256 over ``(instance_hash, spec)`` and ``name``
is the class name in snake_case. Use the registry browser to navigate cached
entries without needing human-readable paths — see
:func:`~pygeodata.registry_browser.serve.open_registry_browser`.

Parallel execution
~~~~~~~~~~~~~~~~~~

:func:`~pygeodata.parallel.build_dask_graph` converts a loader and its
transitive dependencies into a `Dask <https://dask.org>`_ delayed graph,
enabling parallel or distributed execution:

.. code-block:: python

   from dask.distributed import Client
   from pygeodata.parallel import build_dask_graph

   client = Client()
   graph = build_dask_graph(my_loader, spec=spec)
   graph.compute()

Cache management
~~~~~~~~~~~~~~~~

:func:`~pygeodata.cache.clean_cache` scans the cache directory and removes
files whose state hash no longer matches the live code:

.. code-block:: python

   pgd.clean_cache(dry_run=True)    # preview deletions
   pgd.clean_cache(dry_run=False)   # apply

Pass ``delete_unregistered=True`` to :func:`~pygeodata.cache.clean_cache` to
also remove directories that do not belong to any currently imported loader class.
