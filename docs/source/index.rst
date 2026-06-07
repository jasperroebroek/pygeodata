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
   api
   whats_new

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/01_getting_started
   notebooks/02_building_a_pipeline
   notebooks/03_reprojection
   notebooks/04_rasterization
   notebooks/05_custom_processing
   notebooks/06_custom_drivers
   notebooks/07_parallel_processing
   notebooks/08_visualisation_and_debugging
   notebooks/09_registry_browser
