.. _02_building_a_pipeline:

Building a Pipeline
===================

A pygeodata pipeline is a directed acyclic graph of ``Data`` instances.
Each node describes one output; edges are dependencies. This tutorial covers:

- Chaining loaders: one ``Data`` calling ``load()`` on another inside
  ``_process``
- Call dependency detection via AST parsing
- How changing upstream code invalidates downstream caches
- Data instances as constructor parameters — injecting dependencies
  explicitly
- The dependency graph written alongside outputs

Chaining loaders
----------------

The simplest way to express a dependency is to call ``load()`` on an
upstream loader inside ``_process``. Here ``SlopeLoader`` depends on
``ElevationLoader``: it calls ``load(ElevationLoader(), spec)`` to get
the elevation array, then computes slope.

Call dependency detection
-------------------------

When ``SlopeLoader._process`` calls
``load(ElevationLoader(...), spec)``, pygeodata registers
``ElevationLoader`` as a **call dependency** of ``SlopeLoader``. This
detection happens at import time via AST parsing of the class source:
pygeodata finds all names referenced in the class body that resolve to
registered ``TrackedObject`` subclasses.

You never declare dependencies manually. If your code calls a class, it
is a dependency. This means:

- Unused imports do not create spurious dependencies
- Conditional calls (``if some_flag: load(OtherLoader(), spec)``) are
  still detected — AST analysis is conservative

The ``nodes`` dict is a flat index: one entry per reachable class, keyed
by class name, with its current ``source_hash`` and ``object_type``. The
``tree`` encodes the full topology with ``call_dependencies`` and
``inheritance_dependencies`` at every level.

The **dependency tree hash** is SHA-256 over the entire serialized tree.
Any change to ``ElevationLoader`` changes its ``source_hash`` entry in
the nodes dict, which changes the tree hash.

Upstream change invalidates downstream cache
--------------------------------------------

Consider what happens when you edit ``ElevationLoader`` — for example,
switching the resampling algorithm:

The cascade works through the full transitive closure: if ``A → B → C``
and you edit ``C``, then both ``B`` and ``A`` are invalidated on the
next ``process()`` call.

Data instances as parameters
----------------------------

An alternative to call dependencies is to accept ``Data`` instances as
constructor arguments. This is **dependency injection**: the caller
decides which upstream loaders to use, rather than having the downstream
class hard-code them.

How injected Data instances affect the hash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a ``Data`` instance is passed as a parameter, its **instance hash**
is included in the parent's parameter dict during hashing. This means:

- Changing ``WaterTableDepthLoader``\ 's code changes its instance hash
- That changed instance hash propagates into ``LandWaterTableDepth``\ 's
  params hash
- ``LandWaterTableDepth``\ 's state hash changes → its cache is
  invalidated

The invalidation propagates correctly even when the upstream is injected
rather than hard-coded.

Call deps vs injection: comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------+-----------------------+-----------------------+
|                       | Call dependencies     | Injected parameters   |
+=======================+=======================+=======================+
| Upstream wired by     | Class body            | Constructor argument  |
|                       | (hard-coded)          |                       |
+-----------------------+-----------------------+-----------------------+
| Switching upstream    | Edit class source     | Pass different        |
|                       |                       | instance              |
+-----------------------+-----------------------+-----------------------+
| Dep tree hash         | Includes all call     | Only the injected     |
|                       | deps transitively     | instance's hash       |
+-----------------------+-----------------------+-----------------------+
| Use when              | Dependency is fixed   | Caller needs to vary  |
|                       | by design             | upstream              |
+-----------------------+-----------------------+-----------------------+

Both patterns are fully tracked. Mixed usage (some hard-coded, some
injected) is common and correct.

The dependency graph
--------------------

When a ``Data`` instance has any injected ``Data`` parameters and
``process()`` is called, pygeodata writes a ``.graph.pdf`` file
alongside the output. This is a rendered directed graph showing the full
runtime dependency structure — node per ``Data`` instance, labeled with
class name and parameter values.

::

   data/processed/{state_hash}/
   ├── land_water_table_depth.tif
   ├── .land_water_table_depth.hash.json
   ├── .land_water_table_depth.params.json
   ├── .land_water_table_depth.spec.json
   └── .land_water_table_depth.graph.pdf   ← rendered dependency graph

The graph is also accessible programmatically:
