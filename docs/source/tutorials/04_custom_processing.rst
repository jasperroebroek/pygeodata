.. _04_custom_processing:

Custom Processing
=================

When neither ``Reprojector`` nor ``Rasterizer`` fits your use case,
override ``_process(self, spec)`` directly. This gives you full control
over what gets written and how.

This tutorial covers:

- ``_process(self, spec)`` — the override contract
- ``ensure_processed_path`` and ``get_processed_path`` — getting the
  output path
- ``resolve_spec`` — when and why to call it from within ``_process``
- Co-outputs: yielding sibling artifacts from a single ``_process`` run
- ``Figure``: same pattern, stores to ``path_figures`` instead of
  ``path_cache``

Overriding \_process
--------------------

The contract for ``_process(self, spec)`` is simple:

1. Write the output to ``self.ensure_processed_path(spec)`` (or
   ``self.get_processed_path(spec)`` if you handle ``mkdir`` yourself)
2. Return ``None`` for single-output cases, or ``yield`` sibling
   artifacts for co-outputs

You must declare ``ext`` and ``driver`` as class attributes if you are
not using a ``processor`` that provides defaults.

ensure_processed_path vs get_processed_path
-------------------------------------------

+-------------+-------------------------------------+-----------------+
| Method      | Creates parent dirs?                | Use when        |
+=============+=====================================+=================+
| ``ensure_   | Yes                                 | Inside          |
| processed_p |                                     | ``_process`` —  |
| ath(spec)`` |                                     | always safe     |
+-------------+-------------------------------------+-----------------+
| ``get_      | No                                  | Inspection /    |
| processed_p |                                     | pre-check —     |
| ath(spec)`` |                                     | does not modify |
|             |                                     | filesystem      |
+-------------+-------------------------------------+-----------------+

Both return the same ``Path``. The output path follows the
content-addressed layout:

::

   {path_cache}/{state_hash}/{stem}.{ext}

The ``state_hash`` encodes the class dependency tree, all parameter
values, and the spec — so the same loader with different params or spec
always lands in a different directory.

resolve_spec inside \_process
-----------------------------

``Artifact.process()`` calls ``self.resolve_spec(spec)`` before invoking
``_process``. The resolved spec is what gets passed into your method. In
most ``_process`` implementations you do not need to call
``resolve_spec`` again.

The one case where you do need it is when you are constructing sibling
loader paths manually inside ``_process`` and need the resolved spec to
address them correctly:

Co-outputs
----------

Sometimes a single expensive computation naturally produces multiple
related outputs. Rather than running ``_process`` once per output (and
paying the cost N times), you can **yield sibling artifacts** from
``_process``. pygeodata will write the state hash, params, and spec
files for every yielded artifact in the same run.

A co-output pattern is identified by ``_process`` being a generator (it
contains ``yield``). The yielded artifacts must be instances of the same
class with different parameters — the same single ``_process`` call
services all of them.

Example: regression loader that yields beta, se, and p-value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calling ``process()`` on *any* sibling triggers ``_process`` once and
caches all three. Subsequent calls on any sibling are no-ops.

The registry browser groups co-output siblings together in the entry
detail panel (see :doc:`08_registry_browser`).

Figure
------

``Figure`` is a sibling base class to ``Data``. It follows identical
``_process`` / co-output semantics but stores outputs in
``path_figures`` instead of ``path_cache``, and has ``ext = 'png'`` as
its default.

==================== ======================== ==========================
\                    ``Data``                 ``Figure``
==================== ======================== ==========================
Cache root           ``path_cache``           ``path_figures``
Default ``ext``      None (must set)          ``'png'``
``load()`` available Yes                      Yes (returns path content)
Typical use          Rasters, vectors, arrays Plots, maps, diagrams
==================== ======================== ==========================

Use ``Figure`` for any visualisation output that should be cached
alongside data outputs. The registry browser shows ``Figure`` entries
alongside ``Data`` entries.

Figure co-outputs
~~~~~~~~~~~~~~~~~

The co-output pattern works identically for ``Figure``. For example, a
single rendering pass that produces both a global map and a regional
inset can yield both loaders from one ``_process``.
