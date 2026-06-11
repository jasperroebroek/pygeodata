.. _03_reprojection:

Reprojection in Depth
=====================

``Reprojector`` wraps ``rasterio.warp.reproject`` with automatic output
path management, atomic writes, and spec resolution. It is the standard
processor for any workflow that starts from an existing raster file.

This tutorial covers:

- All constructor parameters
- Partial spec resolution: how ``Reprojector.resolve_spec()`` infers
  transform and shape
- ``scales`` and ``offsets``: applying value transforms during the warp
- Handling NetCDF subdatasets with ``forced_read``
- Overriding source CRS for files without embedded projection
- The temp-file + atomic rename pattern

Constructor reference
---------------------

.. code-block:: python

   Reprojector(
       src_path,           # str | Path — source raster or 'netcdf:file.nc:variable'
       bands=None,         # int | Sequence[int] — bands to read (1-indexed); None = all
       src_crs=None,       # rasterio.CRS — override embedded CRS
       src_nodata=None,    # float — override embedded nodata
       resampling=Resampling.nearest,  # rasterio.enums.Resampling
       dst_dtype=None,     # numpy dtype — output dtype; None = same as source
       dst_nodata=None,    # float — output nodata; None = same as source
       nbits=None,         # int — NBITS profile option (e.g. 1 for binary masks)
       scales=None,        # float | Sequence[float] — per-band scale factors
       offsets=None,       # float | Sequence[float] — per-band offsets
       raster_creation_options=None,  # RasterCreationOptions for GeoTIFF profile
       forced_read=False,  # bool — read via src.read() instead of rio.band()
       warp_mem_limit=None,  # int MB — overrides config.warp_mem_limit
       num_threads=None,   # int — overrides config.num_threads
   )

The reprojector has two class-level attributes that ``Data`` uses
automatically:

- ``Reprojector.ext = 'tif'`` — sets the output extension
- ``Reprojector.default_driver = RioXArrayDriver()`` — sets the read
  driver for ``load()``

Basic usage
-----------

Partial spec resolution
-----------------------

When the target ``SpatialSpec`` has only a CRS but no transform or
shape, ``Reprojector.resolve_spec()`` reads the source file and computes
the optimal output transform using
``rasterio.warp.calculate_default_transform``. This preserves the
original resolution as closely as possible while reprojecting to the new
CRS.

This is useful when you want to change the coordinate system without
imposing a specific grid — let rasterio decide the resolution and
extent.

``Artifact.resolve_spec()`` — called internally by ``process()`` —
delegates to ``processor.resolve_spec()`` when the spec is partial. The
resolved spec is then used for both the output path and the warp. You
can call ``resolve_spec`` manually to inspect what will be produced
before committing to a full ``process()`` run.

Selecting bands
---------------

``bands`` accepts a single 1-indexed integer or a sequence. If not set,
all bands are reprojected.

scales and offsets
------------------

``scales`` and ``offsets`` apply the transform
``y = scale * x + offset`` to pixel values **after** reprojection, by
writing GeoTIFF ``SCALE`` and ``OFFSET`` metadata tags via rasterio's
``_set_all_scales`` / ``_set_all_offsets``. This is the standard
mechanism for storing integer rasters with physical unit conversions —
for example, LAI stored as ``uint16`` raw integers where the physical
value is ``raw * 0.1``.

The scale/offset are embedded in the file itself. When
``RioXArrayDriver`` reads the file with ``mask_and_scale=True`` (the
default), rioxarray applies them automatically so you get the physical
values without a separate processing step.

For per-band transforms, pass a sequence with one value per output band:

.. code-block:: python

   Reprojector(
       src_path='data/raw/multi.tif',
       bands=(1, 2, 3),
       scales=(0.1, 0.01, 0.001),  # different scale per band
       offsets=(0.0, 0.0, -273.15),
   )

If ``scales`` or ``offsets`` is ``None``, the values are read from the
source file's embedded tags. This preserves original scale/offset
metadata through the warp.

NetCDF subdatasets
------------------

Use the ``netcdf:path.nc:variable`` path syntax to address a specific
variable in a NetCDF file. Set ``forced_read=True`` to avoid an
empty-raster bug that occurs when ``rasterio.band()`` is used with
multi-variable datasets. Without it, ``Reprojector`` emits a
``RuntimeWarning``.

When ``forced_read=True``, the source data is read via
``src.read(bands)`` into a NumPy array before warping, bypassing the
``rio.band()`` lazy reference. This is slightly more memory-intensive
but required for correct results with NetCDF subdatasets.

Overriding source CRS
---------------------

Some legacy GeoTIFFs and NetCDF files lack embedded CRS metadata.
Provide ``src_crs`` to tell the reprojector what projection to assume
for the source.

Similarly, use ``src_nodata`` to override a missing or incorrect nodata
value in the source file.

Atomic writes
-------------

``Reprojector.__call__`` writes via a temporary file and then renames
atomically:

1. Create a temp file in a ``TemporaryDirectory`` with a random UUID
   name
2. Write the warped output to the temp file
3. Apply scales/offsets tags
4. ``shutil.move(temp_path, dst_path)`` — on POSIX systems this is an
   atomic ``os.rename``

This guarantees that the destination file is either absent or fully
written. A crashed or interrupted ``process()`` call never leaves a
partial output at the final path, which would confuse the hash check.
Combined with the ``filelock.FileLock`` in ``Artifact.process()``, this
makes concurrent processing safe: two processes targeting the same
output will not corrupt each other's files.

GeoTIFF creation options
------------------------

Control compression, tiling, and other GeoTIFF profile settings via
``RasterCreationOptions``. If not set, the Reprojector uses the global
``get_config().raster_creation_options``.
