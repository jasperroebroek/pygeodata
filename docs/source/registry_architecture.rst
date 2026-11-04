Registry architecture
=====================

pygeodata maintains four in-memory registries, each owning a distinct slice of
the ``.source/`` directory and the artifact cache.

.. code-block:: text

    SourceRegistry   — .source/code/{source_hash}/          one CodeState per class snapshot
    TreeRegistry     — .source/snapshots/{dep_tree_hash}/   one TreeSnapshot per dep-tree version
    VersionRegistry  — composes SourceRegistry + TreeRegistry; owns version-group logic
    EntryRegistry    — .catalog.json over {cache_root}/{state_hash}/   one CatalogEntry per processed artifact

``VersionRegistry`` is the composition point: its constructor builds and owns a
``SourceRegistry`` and a ``TreeRegistry``, exposing them via ``.source_registry``
and ``.tree_registry``.  Callers that need version-group information should
construct one ``VersionRegistry`` and access the others through it.

Every registry is built once and never changes; build a new one to see newer data.

The neutral view layer lives in ``pygeodata/views/``: ``classes.py`` builds
``ClassInfo`` from the registries, and ``entries.py`` derives what a display needs
beyond the stored ``CatalogEntry`` (staleness, co-outputs, spec options), without
any Flask dependency.  Both the CLI and the browser (the ``pygeodata-browser``
package) import from ``pygeodata.views``. ``pygeodata/archive.py`` exports and
imports entries with their code snapshots.
``pygeodata/catalog/`` holds the catalog index file, its stored types and the
write path. ``pygeodata/records.py`` holds the per-entry file contracts
(``EntryRecord`` for ``meta.json``, ``SpecRecord`` for ``spec.json``).

Construction pattern
--------------------

* **Browser** — builds one ``VersionRegistry`` + one ``EntryRegistry`` at startup,
  stores both on ``AppState``, and reuses them across all routes.
* **CLI** — constructs fresh registries per subcommand.  This is correct for a
  one-shot process and avoids shared state.

Write path
----------

Writes to ``.source/`` happen in ``pygeodata/registries/writer.py``
(``write_registry()``), called during every ``process()`` run; cache entries are
written by ``pygeodata/cache_entry.py``, which also indexes them in the catalog.
``SourceRegistry``, ``TreeRegistry`` and ``VersionRegistry`` only scan ``.source/``.

``EntryRegistry`` reads the catalog through ``catalog.store.open_catalog()``, which
compares it with one directory listing per cache root (entry dirs are named by
their state hash). A catalog written from this project root is brought up to
date in one locked write: new entry dirs indexed, deleted ones pruned unless their
root is missing or empty. A catalog written from another root (a synced mirror of
a remote project) is read-only: it is used as it is, its difference reported, and
no entry dir is ever scanned.