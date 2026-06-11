.. _05_drivers:

Drivers
=======

A driver is the read half of a ``Data`` loader. After ``_process``
writes an output file, the driver reads it and returns a Python object.
This tutorial covers:

- The ``Driver`` protocol
- ``RioXArrayDriver`` — returns an ``xarray.DataArray``
- ``GeoPandasDriver`` — returns a ``GeoDataFrame``
- Writing a custom driver
- How the ``driver`` class attribute and ``processor.default_driver``
  interact

The Driver protocol
-------------------

A driver is any callable with the signature:

.. code-block:: python

   def __call__(self, path: Path) -> T:
       ...

It also conventionally exposes a ``default_ext`` class attribute that
tells the framework what file extension to expect.

Drivers are attached to a ``Data`` subclass in two ways:

1. **``driver`` class attribute** — explicit, always takes precedence
2. **``processor.default_driver``** — fallback when no ``driver`` is set
   on the class

``Data.load()`` calls ``self.driver(self.get_processed_path(spec))`` and
returns the result.

RioXArrayDriver
---------------

``RioXArrayDriver`` reads a GeoTIFF (or any rasterio-compatible file)
and returns an ``xarray.DataArray`` with CRS and spatial reference
metadata attached via ``rioxarray``.

Key behaviours:

- ``mask_and_scale=True`` (default): nodata pixels become ``NaN``, and
  any ``SCALE``/``OFFSET`` tags embedded by ``Reprojector`` are applied
  automatically. The returned array contains physical values.
- ``flatten=True`` (default): a single-band raster with shape
  ``(1, H, W)`` is squeezed to ``(H, W)``, dropping the ``band``
  dimension.
- Raises ``TooManyDimensions`` if the file contains multiple variables
  (e.g. a multi-variable NetCDF). Use the ``netcdf:file:var`` path
  syntax in ``Reprojector`` to select a variable before caching.

Overriding the driver on a class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``driver`` as a class attribute to override
``processor.default_driver``:

GeoPandasDriver
---------------

``GeoPandasDriver`` reads a vector file (GeoJSON, Shapefile, GPKG, etc.)
and returns a ``geopandas.GeoDataFrame``. Its ``default_ext`` is
``'geojson'``.

``GeoPandasParquetDriver`` is also available for Parquet-backed
GeoDataFrames:

Writing a custom driver
-----------------------

Any callable that accepts a ``Path`` and returns a value is a valid
driver. The simplest form is a plain function; for parameterised drivers
use a dataclass with ``__call__``.

Overriding \_load
~~~~~~~~~~~~~~~~~

For more control, override ``_load(self, path)`` directly on the
``Data`` class instead of providing a driver object. This is equivalent
but keeps the read logic co-located with the class:

When ``_load`` is overridden, the ``driver`` property is never called.
This pattern also works when the output is not a file at all — for
example if ``_process`` writes nothing and ``_load`` computes the result
on the fly from upstream cached files.

Driver resolution order
-----------------------

When ``Data.load()`` needs a driver, resolution proceeds in this order:

1. ``_load`` overridden on the class? → use it directly
2. ``driver`` class attribute set? → use it
3. ``processor.default_driver`` exists? → use it
4. None of the above → ``NotImplementedError``

Extension resolution (for ``get_processed_path``) follows the same
precedence:

1. ``ext`` class attribute set? → use it
2. ``processor.ext`` exists? → use it
3. ``driver.default_ext`` exists? → use it
4. None → ``ValueError``
