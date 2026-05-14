None

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../../source/notebooks/01_getting_started.ipynb>`_.

.. _01_getting_started:

Getting Started with pygeodata
==============================

This notebook introduces the core concepts of ``pygeodata``:

- Configuring the library
- Defining a ``SpatialSpec``
- Writing a minimal ``DataLoader``
- Processing and loading data

.. code:: python

    import os
    os.chdir("../../..")

.. code:: python

    from pygeodata import DataLoader, SpatialSpec, get_config, set_config, process, load
    from pygeodata.processors.reprojection import Reprojector
    from pyproj import CRS
    from affine import Affine
    from pathlib import Path
    from dataclasses import dataclass
    from dataclasses import fields

1. Configure paths
------------------

``set_config`` lets you point ``pygeodata`` at your data directories and
set default processing options.

.. code:: python

    set_config(
        path_data_processed=Path("./data/processed"),  # where outputs are stored
        warp_mem_limit=256,                             # MB of memory for warping
        num_threads=4,                                  # threads for reprojection
    )
    config = get_config()
    for field in fields(config):
        print(f'{field.name}={getattr(config, field.name)}')


.. parsed-literal::

    path_data_processed=data_processed
    num_threads=1
    warp_mem_limit=0
    spec=None
    raster_creation_options=RasterCreationOptions(compress=None, tiled=None, blockxsize=None, blockysize=None, interleave=None, photometric=None, bigtiff=None, sparse_ok=None, kwargs=None)
    max_path_param_depth=5


2. Define a ``SpatialSpec``
---------------------------

A ``SpatialSpec`` describes the *target* coordinate system, transform,
and shape. You can construct one manually or derive it from an existing
raster file.

.. code:: python

    # Option A — derive from an existing reference raster
    # spec = SpatialSpec.from_raster_file("reference.tif")
    
    # Option B — define a 0.5° global grid in WGS-84
    res = 0.5  # degrees per pixel
    spec = SpatialSpec(
        crs=CRS.from_epsg(4326),
        transform=Affine(res, 0, -180, 0, -res, 90),
        shape=(360, 720),  # (height, width)
    )
    print(spec)
    print("Resolution:", spec.resolution)
    print("Bounds:    ", spec.bounds)
    print("Fully defined:", spec.is_fully_defined)


.. parsed-literal::

    SpatialSpec(crs=EPSG:4326, transform=Affine(0.50, 0.00, -180.00, 0.00, -0.50, 90.00), shape=(360, 720))
    Resolution: (0.5, 0.5)
    Bounds:     BoundingBox(left=-180.0, bottom=-90.0, right=180.0, top=90.0)
    Fully defined: True


3. A minimal ``DataLoader``
---------------------------

Subclass ``DataLoader`` and set the ``processor`` property. Here we use
``Reprojector`` which reprojects a source GeoTIFF to the target
``SpatialSpec``. Note that the actual class is imported, as DataLoaders
cannot be defined in a REPL or jupyter notebook.

.. code:: python

    # @dataclass
    # class ElevationLoader(DataLoader):
    #     """Reproject a raw DEM to the target spec."""
    #     src: str = "data/raw/dem.tif"
    
    #     @property
    #     def processor(self):
    #         return Reprojector(src_path=self.src)
    
    
    from utils.docs_loaders import ElevationLoader
    
    
    loader = ElevationLoader()
    print(loader)
    print("Output path:", loader.get_processed_path(spec))


.. parsed-literal::

    ElevationLoader(src='data/elevation.tif')
    Output path: data_processed/EPSG_4326/affine_0.5000_0.0000_-180.0000_0.0000_-0.5000_90.0000_shape_360_720/ElevationLoader/src=data_elevation.tif/elevation_loader.tif


4. Process and load
-------------------

``process()`` runs the pipeline (skips if output is already cached).
``load()`` processes and returns the data (an ``xarray.DataArray`` via
``RioXArrayDriver``).

.. code:: python

    # process() is idempotent — safe to call multiple times
    resolved_spec = process(loader, spec)
    print("Resolved spec:", resolved_spec)
    
    # load() calls process() internally
    da = load(loader, spec)
    print(da)


.. parsed-literal::

    Resolved spec: SpatialSpec(crs=EPSG:4326, transform=Affine(0.50, 0.00, -180.00, 0.00, -0.50, 90.00), shape=(360, 720))
    <xarray.DataArray (y: 360, x: 720)> Size: 1MB
    [259200 values with dtype=float32]
    Coordinates:
      * y            (y) float64 3kB 89.75 89.25 88.75 ... -88.75 -89.25 -89.75
      * x            (x) float64 6kB -179.8 -179.2 -178.8 ... 178.8 179.2 179.8
        spatial_ref  int64 8B 0
    Attributes:
        AREA_OR_POINT:  Area


5. Inspect the cache state
--------------------------

The cache key combines a hash of the **class source code** (AST) and the
**parameter values**. Changing either invalidates the cache.

.. code:: python

    print("Is processed:     ", loader.is_processed(spec))
    print("Cache valid:      ", loader.is_cache_valid(spec))
    print("State hash:       ", loader.get_state_hash())
    print("Source hash:      ", loader.get_source_hash())
    print("Hierarchy hash:   ", loader.get_source_hierarchy_hash())


.. parsed-literal::

    Is processed:      True
    Cache valid:       True
    State hash:        3c71364e72609edf57b6faf330f589884878e893843ea6356984f1f2ed042c01
    Source hash:       f757c1dca26c8d6795d2e59d634169d221728242aa845d758029bfb2e44f9ecf
    Hierarchy hash:    a0a7be62f82f473087184c8aedfddad806ec244ce80595d3cf232e7750e453eb

