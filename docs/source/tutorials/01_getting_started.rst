.. _01_getting_started:

Getting Started
===============

pygeodata is a declarative geospatial pipeline framework. You describe
*what* you want — a ``Data`` subclass with parameters — set a
``SpatialSpec`` once, and the framework handles caching, dependency
tracking, and invalidation. The design is analogous to SQL: describe the
desired output, the orchestrator figures out whether to compute or
retrieve it from cache.

This tutorial covers:

- Installing and configuring pygeodata
- Defining a ``SpatialSpec``
- Writing a minimal ``Data`` subclass backed by ``Reprojector``
- ``process()`` and ``load()`` — what they do and why the second call is
  a no-op
- What lands on disk: the content-addressed cache layout

Installation
------------

.. code-block:: bash

   pip install pygeodata

For parallel processing support:

.. code-block:: bash

   pip install pygeodata[parallel]

Configuration
-------------

``get_config()`` returns the single global ``Config`` instance. Call
``.update()`` on it to set paths and processing options. All keys are
validated — typos raise ``ValueError`` immediately.

The ``set_config`` function is also available as a context manager for
temporary overrides:

.. code-block:: python

   from pygeodata import set_config

   with set_config(path_cache=Path('data/test')):
       result = load(my_loader, spec)  # uses the temporary path
   # original path_cache restored here

SpatialSpec
-----------

A ``SpatialSpec`` describes the target coordinate reference system,
affine transform, and raster shape. It is the key that binds a ``Data``
instance to a specific output: the same loader with two different specs
produces two independently cached files.

From a reference raster
~~~~~~~~~~~~~~~~~~~~~~~

The most common pattern is to derive the spec from an existing file:

Explicit construction
~~~~~~~~~~~~~~~~~~~~~

You can also build a spec from scratch. An affine transform follows the
rasterio convention: ``Affine(x_res, 0, x_origin, 0, -y_res, y_origin)``
where the origin is the upper-left corner.

Partial specs
~~~~~~~~~~~~~

A spec with only a CRS (no transform or shape) is *partial*. Partial
specs are valid inputs to ``process()`` and ``load()`` when the loader's
processor implements ``resolve_spec()`` — ``Reprojector`` does this by
reading the source file and computing the optimal transform. See
:doc:`03_reprojection` for details.

Defining a Data subclass
------------------------

Subclass ``Data`` and implement either:

- ``processor`` property — for reprojection, rasterization, or any
  callable ``(dst_path, spec) -> None``
- ``_process(self, spec)`` — for arbitrary custom logic

Use ``@dataclass`` to declare parameters as fields. pygeodata reads
parameters via ``vars(self)``, so any instance attribute is included in
the cache key.

   **Important:** ``Data`` subclasses must be defined in importable
   modules, not interactively in a notebook or REPL. pygeodata reads
   class source via ``inspect.getsource()`` for AST-based hashing;
   interactive definitions have no inspectable source.

The ``Reprojector`` processor handles the actual warp via
``rasterio.warp.reproject``. Because a ``processor`` is declared,
pygeodata also knows:

- the output extension (``tif``, from ``Reprojector.ext``)
- the default driver (``RioXArrayDriver``, from
  ``Reprojector.default_driver``)

You do not need to set ``ext`` or ``driver`` explicitly when using
``Reprojector``.

process() and load()
--------------------

``process(artifact, spec)`` is the primary entry point. It:

1. Resolves the spec (fills in transform/shape if partial)
2. Computes the state hash for ``(artifact, spec)``
3. Checks whether a valid cached output already exists
4. Runs ``_process`` only if the cache is missing or stale
5. Writes the hash file, params file, and spec file alongside the output

``load(artifact, spec)`` calls ``process()`` then reads the output via
the artifact's driver, returning the data directly.

What lands on disk
------------------

pygeodata uses a **content-addressed cache**. The output directory path
is derived from the state hash — a SHA-256 digest over the class
dependency tree, instance parameters, and spec. No two distinct
``(loader, spec)`` combinations can collide.

::

   data/processed/
   └── {state_hash}/
       ├── elevation_loader.tif        # the output file
       ├── .elevation_loader.hash.json # state hash + dependency_tree_hash
       ├── .elevation_loader.params.json
       └── .elevation_loader.spec.json

The four files in every cache directory:

+--------------------------+--------------------------------------------+
| File                     | Contents                                   |
+==========================+============================================+
| ``{stem}.tif``           | The processed output                       |
+--------------------------+--------------------------------------------+
| ``.{stem}.hash.json``    | ``state_hash``, ``dependency_tree_hash``,  |
|                          | ``instance_hash``, ``class_name``,         |
|                          | ``format_version``                         |
+--------------------------+--------------------------------------------+
| ``.{stem}.params.json``  | JSON-serialized parameter dict, including  |
|                          | nested ``Data`` params                     |
+--------------------------+--------------------------------------------+
| ``.{stem}.spec.json``    | Full ``SpatialSpec`` dict: CRS WKT,        |
|                          | transform coefficients, shape              |
+--------------------------+--------------------------------------------+

On subsequent ``process()`` calls, the framework reads ``.hash.json``,
computes the live hash, and compares. Match → skip. Mismatch →
reprocess.

The **instance hash** is spec-independent: it identifies the
``(class code + params)`` combination. The **state hash** adds the spec
on top, making it unique per ``(artifact, spec)`` pair.

The **dependency tree hash** captures the AST of the entire transitive
class graph. If you edit ``ElevationLoader`` — even just reformat it —
the hash changes and downstream caches are invalidated. pygeodata uses
AST comparison, not raw text, so pure reformatting does *not* change the
hash.

The .source/ registry
---------------------

Alongside the cache, pygeodata maintains a ``.source/`` directory
(configurable via ``path_registry``). It is written automatically
whenever ``process()`` runs.

::

   .source/
   ├── code/
   │   └── {source_hash}/
   │       ├── source.py       # raw source of the class at that hash
   │       └── source.json     # class_name, source_hash, registered_at timestamp
   └── snapshots/
       └── {dep_tree_hash}/
           ├── tree.json       # full {nodes, tree} dependency topology
           └── graph.pdf       # rendered dependency graph (if deps exist)

This store is **append-only**: entries are never overwritten. It is the
version history used by the registry browser's Code view (see
:doc:`08_registry_browser`).
