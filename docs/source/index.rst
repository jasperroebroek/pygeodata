pygeodata
=========

**pygeodata** is a framework for deterministic, cached geospatial data processing.
It provides a :class:`~pygeodata.data.Data` base class whose subclasses
encapsulate rasterization, reprojection, and arbitrary processing steps. Outputs
are cached on disk and invalidated automatically when source code or parameters
change, enabling reproducible, incremental pipelines.

.. code-block:: python

   import pygeodata as pgd

   pgd.get_config().update(path_cache='data/processed')

   spec = pgd.SpatialSpec.from_raster_file('reference.tif')
   data = pgd.load(MyLoader(year=2020), spec)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   quickstart
   cli
   api
   whats_new

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/01_getting_started
   tutorials/02_building_a_pipeline
   tutorials/03_reprojection
   tutorials/04_custom_processing
   tutorials/05_drivers
   tutorials/06_parallel_processing
   tutorials/07_cache_management
   tutorials/08_registry_browser

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/01_pipeline_and_cache
   examples/02_data_as_parameters