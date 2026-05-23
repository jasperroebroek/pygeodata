pygeodata
=========

**pygeodata** is a framework for deterministic, cached geospatial data processing.
It provides a :class:`~pygeodata.data.Data` base class whose subclasses
encapsulate rasterization, reprojection, and arbitrary processing steps. Outputs
are cached on disk and invalidated automatically when source code or parameters
change, enabling reproducible, incremental pipelines.

.. code-block:: python

   import pygeodata as pgd

   pgd.set_config(path_data_processed='data_processed')

   spec = pgd.SpatialSpec.from_raster_file('reference.tif')
   data = pgd.load(MyLoader(year=2020), spec)

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   api
   whats_new
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   notebooks/01_getting_started
   notebooks/02_building_a_pipeline
