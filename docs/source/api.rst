#############
API reference
#############

This page documents the full public API of pygeodata. All symbols listed under
:ref:`top-level` are importable directly from the ``pygeodata`` namespace.

.. contents:: Sections
   :local:
   :depth: 1

.. _top-level:

Top-level functions
===================

These two functions are the primary entry points for running a pipeline.

.. autosummary::
   :toctree: generated/

   pygeodata.api.process
   pygeodata.api.load

.. autofunction:: pygeodata.api.process
   :no-index:

.. autofunction:: pygeodata.api.load
   :no-index:


Core classes
============

.. autosummary::
   :toctree: generated/

   pygeodata.data.Data
   pygeodata.figure.Figure
   pygeodata.artifact.Artifact
   pygeodata.tracked_object.TrackedObject

Data
----

.. autoclass:: pygeodata.data.Data
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Figure
------

.. autoclass:: pygeodata.figure.Figure
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Artifact
--------

.. autoclass:: pygeodata.artifact.Artifact
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

TrackedObject
-------------

.. autoclass:: pygeodata.tracked_object.TrackedObject
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


SpatialSpec
===========

.. autosummary::
   :toctree: generated/

   pygeodata.types.SpatialSpec

.. autoclass:: pygeodata.types.SpatialSpec
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Config
======

.. autosummary::
   :toctree: generated/

   pygeodata.config.Config
   pygeodata.config.get_config
   pygeodata.config.set_config

.. autoclass:: pygeodata.config.Config
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autofunction:: pygeodata.config.get_config
   :no-index:

.. autofunction:: pygeodata.config.set_config
   :no-index:

The module-level constant ``FORMAT_VERSION`` records the integer cache format version.
Cache entries written with a different ``FORMAT_VERSION`` are treated as stale.

.. autodata:: pygeodata.config.FORMAT_VERSION
   :no-index:


Processors
==========

Processors implement the :class:`~pygeodata.types.Processor` protocol: they are
callables that accept ``(dst_path, spec)`` and write the output file. Built-in
processors can be attached to any :class:`~pygeodata.data.Data` or
:class:`~pygeodata.figure.Figure` subclass via the ``processor`` class attribute.

.. autosummary::
   :toctree: generated/

   pygeodata.processors.Reprojector
   pygeodata.processors.Rasterizer

Reprojector
-----------

.. autoclass:: pygeodata.processors.Reprojector
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Rasterizer
----------

.. autoclass:: pygeodata.processors.Rasterizer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Drivers
=======

Drivers implement the :class:`~pygeodata.types.Driver` protocol: they are
callables that accept a file path and return loaded data. A driver is attached to
a :class:`~pygeodata.data.Data` subclass via the ``driver`` property or is
provided as ``default_driver`` on a processor.

.. autosummary::
   :toctree: generated/

   pygeodata.drivers.RioXArrayDriver
   pygeodata.drivers.GeoPandasDriver
   pygeodata.drivers.GeoPandasParquetDriver

.. autoclass:: pygeodata.drivers.RioXArrayDriver
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: pygeodata.drivers.GeoPandasDriver
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: pygeodata.drivers.GeoPandasParquetDriver
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Cache management
================

.. autosummary::
   :toctree: generated/

   pygeodata.cache.clean_cache
   pygeodata.cache.clean_registry
   pygeodata.cache.rebuild_registry

.. autofunction:: pygeodata.cache.clean_cache
   :no-index:

.. autofunction:: pygeodata.cache.clean_registry
   :no-index:

.. autofunction:: pygeodata.cache.rebuild_registry
   :no-index:


Path resolvers
==============

Path resolver dataclasses centralise the derivation of all on-disk paths for a
given cache entry or registry entry.

.. autosummary::
   :toctree: generated/

   pygeodata.paths.CachePathResolver
   pygeodata.paths.CodeRegistryResolver
   pygeodata.paths.TreeRegistryResolver

.. autoclass:: pygeodata.paths.CachePathResolver
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: pygeodata.paths.CodeRegistryResolver
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: pygeodata.paths.TreeRegistryResolver
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Hashing utilities
=================

These functions underpin the four-level hash chain
(*source_hash → dep_tree_hash → instance_hash → state_hash*).

.. autosummary::
   :toctree: generated/

   pygeodata.hash.calculate_cls_source_hash
   pygeodata.hash.calculate_dict_hash

.. autofunction:: pygeodata.hash.calculate_cls_source_hash
   :no-index:

.. autofunction:: pygeodata.hash.calculate_dict_hash
   :no-index:


Parallel processing
===================

Optional Dask-based parallel execution. Requires ``pygeodata[parallel]``.

.. autosummary::
   :toctree: generated/

   pygeodata.parallel.build_dask_graph

.. autofunction:: pygeodata.parallel.build_dask_graph
   :no-index:


Types and protocols
===================

Protocol classes and lightweight dataclasses used throughout the framework.
``Processor`` and ``Driver`` are structural protocols — any object that matches
the required signature qualifies, without explicit subclassing.

.. autosummary::
   :toctree: generated/

   pygeodata.types.Processor
   pygeodata.types.Driver
   pygeodata.types.AllowsFormatting
   pygeodata.types.HasParameters

.. autoclass:: pygeodata.types.Processor
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: pygeodata.types.Driver
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: pygeodata.types.AllowsFormatting
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: pygeodata.types.HasParameters
   :members:
   :undoc-members:
   :no-index:
