.. _06_parallel_processing:

Parallel Processing
===================

pygeodata integrates with `Dask <https://dask.org/>`__ for parallel
processing. ``build_dask_graph`` constructs a lazy computation graph
that respects data dependencies, allowing Dask to schedule work across
cores or a distributed cluster.

This tutorial covers:

- ``build_dask_graph()`` — building a Dask delayed graph from a loader
- Processing multiple specs in parallel
- Nested dependency graphs: upstream loaders wired automatically
- The file-lock-then-recheck pattern that makes concurrent processing
  safe
- When to use parallel vs sequential ``process()``

build_dask_graph
----------------

``build_dask_graph(artifact, spec)`` wraps a ``process()`` call as a
Dask ``delayed`` node and returns it. Calling ``dask.compute(task)``
executes it — if the cache is already valid, the node completes
immediately without reprocessing.

.. code-block:: python

   from pygeodata.parallel import build_dask_graph

   task = build_dask_graph(my_loader, spec=my_spec)
   task.compute()

The Dask task name is ``"{ClassName}-{hash[:8]}"``, which appears in the
Dask dashboard for easy identification.

Processing one loader at multiple specs in parallel
---------------------------------------------------

A common use case is producing the same output at several resolutions or
extents. Build one task per spec and compute them all together.

Nested DAGs: upstream dependencies wired automatically
------------------------------------------------------

``build_dask_graph`` inspects the loader's parameters recursively. Any
embedded ``Data`` instance found in ``get_params()`` is treated as an
upstream node and wired as a Dask dependency. You do not need to build
tasks for upstream loaders separately.

Distributed cluster
-------------------

Connect a Dask ``Client`` before calling ``compute()`` to distribute
work across many workers. The API is identical to the threaded case —
only the scheduler changes.

Thread safety: the file-lock-then-recheck pattern
-------------------------------------------------

``Artifact.process()`` — called by each Dask task — uses the following
pattern to guarantee correctness under concurrent access:

.. code-block:: python

   def process(self, spec):
       spec = self.resolve_spec(spec)

       if self.is_processed(spec):      # fast pre-check (no lock)
           return

       lock_path = self.get_processed_path(spec).parent / 'process.lock'

       with FileLock(lock_path, timeout=3600):   # acquire per-directory lock
           if self.is_cache_valid(spec):         # recheck inside lock
               return                            # another worker already finished
           self._process(spec)                   # only one worker runs this
           ...

The double-check (pre-lock check + post-lock recheck) means:

- If the cache is already valid when the task starts, it returns
  immediately without contention
- If two tasks race to process the same output, one acquires the lock
  first, finishes, and writes the hash file. The second acquires the
  lock, finds the hash file valid, and returns without reprocessing
- ``Reprojector`` uses ``shutil.move`` (atomic rename) so the output is
  never visible in a partial state

When to use parallel vs sequential
----------------------------------

**Use ``process_parallel`` / ``build_dask_graph`` when:**

- Processing the same loader for many independent specs (resolutions,
  extents, years)
- A pipeline has multiple independent branches that can run concurrently
- Reprojection is the bottleneck and you have multiple cores available
  (each ``Reprojector`` call is GIL-releasing via GDAL/C)

**Use sequential ``process()`` when:**

- Loaders have strict linear dependencies — parallelism offers no
  benefit
- The pipeline is already mostly cached and re-runs are fast
- You are debugging: sequential tracebacks are much easier to follow

Dask is an optional dependency. Install it with:

.. code-block:: bash

   pip install pygeodata[parallel]
