Installation
============

Requirements
------------

pygeodata requires Python 3.10 or later. The core dependencies are installed
automatically:

- `rioxarray <https://corteva.github.io/rioxarray/>`_ — raster I/O via rasterio
- `geopandas <https://geopandas.org/>`_ — vector data
- `zarr <https://zarr.readthedocs.io/>`_ — Zarr array support
- `fiona <https://fiona.readthedocs.io/>`_ — vector file I/O backend
- `pyarrow <https://arrow.apache.org/docs/python/>`_ — GeoParquet support
- `filelock <https://py-filelock.readthedocs.io/>`_ — safe concurrent cache writes

Installing
----------

.. code-block:: bash

   pip install pygeodata

Optional extras
---------------

Install additional optional dependencies by specifying extras:

.. code-block:: bash

   pip install pygeodata[viz]           # dependency graph plots
   pip install pygeodata[parallel]      # Dask integration
   pip install pygeodata[dashboard]     # registry browser
   pip install pygeodata[test]          # test suite
   pip install pygeodata[documentation] # docs build

Or all at once:

.. code-block:: bash

   pip install pygeodata[all]

The extras provide:

- **viz** — ``graphviz`` for rendering class and runtime dependency graphs with
  :func:`~pygeodata.graphs.plot_compact_execution_graph` and
  :func:`~pygeodata.graphs.plot_class_dependency_graph`.

- **parallel** — ``dask`` and ``distributed`` for converting loader graphs to
  Dask delayed graphs via :func:`~pygeodata.parallel.build_dask_graph`.

- **dashboard** — ``flask`` for the local registry browser launched by
  :func:`~pygeodata.registry_browser.open_registry_browser`.

- **test** — ``pytest`` and ``pytest-mock`` for running the test suite.

- **documentation** — ``sphinx``, ``sphinx_rtd_theme``, ``numpydoc``,
  ``jupyter``, and ``matplotlib`` for building these docs.

Development install
-------------------

To install from source with all extras:

.. code-block:: bash

   git clone https://github.com/jasper-roebroek/pygeodata
   cd pygeodata
   pip install -e ".[all]"
