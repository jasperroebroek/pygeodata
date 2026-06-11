.. _07_cache_management:

Cache Management
================

pygeodata is designed to be append-only during normal operation: outputs
accumulate in ``path_cache`` and code snapshots accumulate in
``path_registry``. This tutorial explains the maintenance tools that let
you clean up stale entries.

This tutorial covers:

- ``clean_cache(dry_run=True/False)`` — removing stale or invalid cache
  entries
- ``clean_registry(dry_run=True/False)`` — removing outdated
  ``.source/`` entries
- ``rebuild_registry()`` — rebuilding the registry from scratch
- The ``.source/`` layout in detail
- Why ``.source/`` is append-only

clean_cache
-----------

``clean_cache`` walks ``path_cache`` (and ``path_figures`` for
``Figure`` outputs) and removes any directory whose cached hash no
longer matches the live state. This happens when:

- The class or any of its dependencies was modified after the output was
  written
- The format version changed between pygeodata releases
- The hash file is missing entirely (incomplete previous run)

Always run with ``dry_run=True`` first to review what would be deleted.

What clean_cache walks
~~~~~~~~~~~~~~~~~~~~~~

``clean_cache`` iterates over all ``Artifact`` subclasses (``Data`` and
``Figure``) and walks their respective cache roots. For each leaf
directory (a directory that contains data files but no subdirectories),
it:

1. Looks for a ``.{stem}.hash.json`` file
2. If missing → deletes the directory (label: ``Hash missing``)
3. If present → checks ``format_version`` and the
   ``dependency_tree_hash`` against the live class
4. If stale → deletes the directory (label: ``Hash wrong``)
5. If the class is no longer registered in ``TrackedObject._registry`` →
   prompts for confirmation

After deletion, empty parent directories are pruned automatically.

Unregistered classes
~~~~~~~~~~~~~~~~~~~~

If a cached output belongs to a class that no longer exists in the
current Python environment (renamed, deleted, or not imported),
``clean_cache`` cannot compute the live hash to compare against. With
``delete_unregistered=True`` (the default), it prompts you
interactively:

::

   OldLoader not found in registry. Delete? [y/N]

Set ``delete_unregistered=False`` to skip these entries silently.

clean_registry
--------------

``clean_registry`` scans the ``.source/`` directory and removes entries
written by a different ``FORMAT_VERSION``. This is necessary after major
pygeodata updates that change the on-disk schema.

It walks ``code/*/source.json`` and ``snapshots/*/tree.json``, checks
the ``format_version`` field, and deletes any directory whose metadata
either lacks the field or carries an old version number.

rebuild_registry
----------------

``rebuild_registry()`` deletes the entire ``.source/`` directory and
re-writes it from scratch for all currently registered classes. Use this
when:

- The registry is corrupted or in an unknown state
- You want to remove all historical code snapshots and start fresh
- You have renamed ``path_registry`` and want to repopulate the new
  location

..

   **Warning:** ``rebuild_registry()`` discards the version history used
   by the registry browser's Code view. After this call, only the
   current version of each class will be visible in the code browser —
   all previous snapshots are gone.

The .source/ layout
-------------------

The ``.source/`` directory (controlled by ``path_registry``) has two
subdirectories:

::

   .source/
   ├── code/
   │   └── {source_hash}/          ← one dir per unique class source text
   │       ├── source.py           ← raw Python source of the class at that hash
   │       └── source.json         ← metadata: class_name, source_hash, registered_at
   └── snapshots/
       └── {dep_tree_hash}/        ← one dir per unique dependency tree state
           ├── tree.json           ← full {nodes, tree} topology dict
           └── graph.pdf           ← rendered class dependency graph (if deps exist)

code/ — per-class source snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each time a class is first seen at a new ``source_hash``,
``_write_code_registry()`` creates a new directory under ``code/`` and
writes:

- ``source.py``: the exact source text of the class as returned by
  ``inspect.getsource()``
- ``source.json``: metadata including ``class_name``, ``source_hash``,
  ``object_type``, and ``registered_at`` (ISO timestamp)

The ``registered_at`` timestamp is refreshed whenever this hash becomes
active again after a different hash was active — for example, when you
revert a commit. This keeps the most-recent-use ordering accurate for
the Code browser.

snapshots/ — dependency tree snapshots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each unique ``dep_tree_hash`` (SHA-256 of the entire class graph
including all transitive dependencies) gets one directory under
``snapshots/`` with:

- ``tree.json``: the ``{nodes, tree}`` dict representing the full
  dependency topology
- ``graph.pdf``: a rendered class dependency graph (only written when
  the class has dependencies)

The ``nodes`` section maps class names to their source hashes and object
types. The ``tree`` section encodes the full topology.

Why .source/ is append-only
---------------------------

The ``.source/`` store is designed to be **append-only**: once a
``code/{source_hash}/`` or ``snapshots/{dep_tree_hash}/`` directory
exists, it is never overwritten.

This is intentional. The registry browser's Code view shows the full
version history of every class — every unique source text that was ever
active, with timestamps. This history is only preserved if old entries
are never deleted during normal operation.

The only time ``.source/`` should shrink is:

1. ``clean_registry()`` — removes entries from a different
   ``format_version``
2. ``rebuild_registry()`` — complete reset

Entries added by different computers or collaborators can be safely
merged because the content-addressed layout means no two entries ever
conflict: same hash = same content, different hash = different
directory.

Checking registry validity
--------------------------

Before processing, you can check whether the registry is up to date for
a class without triggering any writes:

``is_registry_valid()`` returns ``True`` when all expected files exist
for the current ``source_hash`` and ``dep_tree_hash``. It returns
``False`` when any file is missing — typically because the class was
modified since the last write.
