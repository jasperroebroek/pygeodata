Registry architecture
=====================

pygeodata maintains four in-memory registries, each owning a distinct slice of
the ``.source/`` directory and the artifact cache.

.. code-block:: text

    SourceRegistry   — .source/code/{source_hash}/          one CodeState per class snapshot
    TreeRegistry     — .source/snapshots/{dep_tree_hash}/   one TreeSnapshot per dep-tree version
    VersionRegistry  — composes SourceRegistry + TreeRegistry; owns version-group logic
    EntryRegistry    — {cache_root}/{state_hash}/            one EntryRecord per processed artifact

``VersionRegistry`` is the composition point: its constructor builds and owns a
``SourceRegistry`` and a ``TreeRegistry``, exposing them via ``.source_registry``
and ``.tree_registry``.  Callers that need version-group information should
construct one ``VersionRegistry`` and access the others through it.

The neutral view-model layer lives in ``pygeodata/catalog/``: ``class_catalog.py``
and ``entry_catalog.py`` build ``ClassInfo`` / ``EntryInfo`` dicts from the
registries without any Flask dependency.  Both the CLI and the browser import
from ``pygeodata.catalog``; only the browser imports from
``pygeodata.registry_browser``.

Construction pattern
--------------------

* **Browser** — builds one ``VersionRegistry`` + one ``EntryRegistry`` at startup,
  stores both on ``AppState``, and reuses them across all routes.
* **CLI** — constructs fresh registries per subcommand.  This is correct for a
  one-shot process and avoids shared state.

Write path
----------

Writes to ``.source/`` happen in ``TrackedObject._write_code_registry()`` and
``_write_tree_registry()``, called automatically by ``write_registry()`` during
every ``process()`` run.  The registries themselves are read-only: they scan
``.source/`` on construction and never write to it.