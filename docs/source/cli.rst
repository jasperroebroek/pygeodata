CLI reference
=============

The ``pygeodata`` command-line interface gives you direct access to the code
registry, dep-tree snapshots, cache entries, and version history without
starting the browser.

All commands that accept a hash value also accept a **truncated prefix** —
any unique prefix is resolved automatically.  An ambiguous prefix produces a
clear error listing the candidates.

.. code-block:: text

    pygeodata
    │
    ├── browse
    │     Launch the registry browser web UI.
    │     --port INTEGER          Port to listen on (0 = random free port)
    │     --import-all            Import all .py files from cwd before starting
    │     --verbose-import        Print each module imported (with --import-all)
    │
    ├── classes
    │     Show staleness for all loaded and known classes, grouped by type.
    │     --import-all            Import all .py files from cwd first
    │     --verbose-import        Print each module imported (with --import-all)
    │     --type TEXT             Filter to one object type: DATA or FIGURE
    │     --hide-current          Omit classes with no staleness
    │     --full-hash             Print full source hashes
    │     --registry TEXT         Path to .source/ registry root
    │
    ├── clean-cache
    │     Remove stale or invalid cache entries.
    │     --no-dry-run            Actually delete files (default is a dry run)
    │     --no-delete-unregistered  Keep entries whose class is no longer in the registry
    │
    ├── clean-source
    │     Remove orphaned code snapshots and dependency trees from .source/.
    │     Keeps the latest snapshot per class and anything referenced by a live
    │     cache entry.  Runs as a dry run by default.
    │     --no-dry-run            Actually delete files (default is a dry run)
    │
    ├── import ARCHIVE
    │     Import a pygeodata .tar.gz export archive into the current project.
    │
    ├── versions
    │     Show the full version timeline: every group with its changed classes
    │     and associated dep-tree snapshots, newest-first.
    │     With --class, show that class's full snapshot history instead
    │     (mirrors the browser's per-class Versions card).
    │     --registry TEXT         Path to .source/ (defaults to project config)
    │     --full-hash             Print full hashes instead of 12-char prefixes
    │     --class TEXT            Show one class's full snapshot history
    │
    ├── code
    │   Inspect the .source/ code registry.
    │   All subcommands accept --registry TEXT.
    │   │
    │   ├── list
    │   │     List all tracked classes grouped by object type with snapshot counts.
    │   │     --full-hash
    │   │     -v / --verbose      Also print each snapshot's hash and registration date
    │   │
    │   ├── show
    │   │     Show snapshot metadata for a class or source hash.
    │   │     --full-hash
    │   │     --class TEXT        List all snapshots for this class
    │   │     --hash TEXT         Show metadata for this source hash (prefix ok)
    │   │
    │   ├── source
    │   │     Print or diff source code snapshots.
    │   │     --full-hash         Affects diff header labels
    │   │     --class TEXT        Show or diff the latest snapshot for this class
    │   │     --hash TEXT         Hash to show (prefix ok); repeat twice to diff two hashes
    │   │     --diff              Diff latest vs previous snapshot (with --class or single --hash)
    │   │     --expand            Show full file context in diffs (no truncation)
    │   │     --no-color          Disable ANSI diff colours
    │   │
    │   └── versions
    │         Look up which version group contains a given class or source hash.
    │         Unlike the top-level ``versions`` command, this is a scoped lookup
    │         tool — it does not print the full timeline.
    │         --class TEXT        List groups that include this class
    │         --hash TEXT         Show which group owns this source hash (prefix ok)
    │
    ├── snapshot
    │   Inspect the .source/ dep-tree snapshot registry.
    │   All subcommands accept --registry TEXT.
    │   │
    │   ├── list
    │   │     List all dep-tree snapshots with their root class and node count.
    │   │     --full-hash
    │   │
    │   ├── show
    │   │     Show contents of a dep-tree snapshot.
    │   │     --full-hash
    │   │     --hash TEXT         Dep-tree hash to inspect (prefix ok)  [required]
    │   │     --json              Print raw tree.json instead of formatted output
    │   │
    │   └── version
    │         Show which version group a dep-tree snapshot belongs to.
    │         --hash TEXT         Dep-tree hash to look up (prefix ok)  [required]
    │
    └── entry
        Inspect cache entries.  Options on the group itself apply to all subcommands.
        --import-all            Import all .py files from cwd before scanning
                                (required to resolve staleness when classes have deps)
        --verbose-import        Print each module imported (with --import-all)
        --full-hash             Print full hashes instead of 12-char prefixes
        │
        ├── list
        │     List all cache entries with staleness indicator, CRS, resolution, bounds,
        │     and hash.  Columns are aligned; use grep to filter further.
        │     --class TEXT        Show only entries of this class
        │     --hide-stale        Omit entries with any staleness indicator
        │
        └── show HASH_PREFIX
              Show full detail for one cache entry: identity, staleness, spatial spec,
              co-outputs, file paths, and params JSON.  Hash prefix ok.
              --no-params         Omit params JSON from output


Staleness indicators
--------------------

Shown in the first column of ``entry list`` and in the ``Staleness`` line of
``entry show``:

+------------+----------------------------------------------------+
| Indicator  | Meaning                                            |
+============+====================================================+
| ``S``      | Dependency-tree hash has changed — entry is stale  |
+------------+----------------------------------------------------+
| ``F``      | Cache format version mismatch                      |
+------------+----------------------------------------------------+
| ``N``      | Class not loaded in this process — staleness unknown |
+------------+------------------------------------------------------+
| *(blank)*  | Up to date                                           |
+------------+------------------------------------------------------+

Pass ``--import-all`` to the ``entry`` group to import project modules and
resolve ``N`` into ``S`` or blank.


Example output
--------------

``entry list`` (with ``--import-all`` so staleness is resolved):

.. code-block:: text

    S AspectLoader                             Data    EPSG:3035  1000m  29.9° N, 19.9° W → 63.8° N, 69.0° E  cc71cf42816b
      AspectLoader                             Data    EPSG:3035  1000m  29.9° N, 19.9° W → 63.8° N, 69.0° E  e8d7f06bb04c
      BenchmarkSlopeLoader                     Data    EPSG:3035  1000m  29.9° N, 19.9° W → 63.8° N, 69.0° E  171fb6572c5d
      BioClimaticVariablesLoader               Data    EPSG:3035  1000m  29.9° N, 19.9° W → 63.8° N, 69.0° E  7480992da0e6

``entry show`` — a stale entry:

.. code-block:: text

    Class        AspectLoader
    Type         Data
    State hash   cc71cf42816b
    Instance     3c444c0eface
    Dep hash     ab48e7305e06
    Staleness    dep stale

    CRS          EPSG:3035
    Resolution   1000m
    Shape        [4358, 5379]
    Bounds       29.9° N, 19.9° W → 63.8° N, 69.0° E

    Params path  data_processed/cc71cf42816b.../.aspect_loader.params.json
    Hash path    data_processed/cc71cf42816b.../.aspect_loader.hash.json
    Spec path    data_processed/cc71cf42816b.../.aspect_loader.spec.json

``entry show`` — a fresh entry with params:

.. code-block:: text

    Class        BioClimaticVariablesLoader
    Type         Data
    State hash   ff862faa1d94
    Instance     90cdd0320676
    Dep hash     f8a005796723
    Staleness    ok

    CRS          EPSG:3035
    Resolution   1000m
    Shape        [4358, 5379]
    Bounds       29.9° N, 19.9° W → 63.8° N, 69.0° E

    Params path  data_processed/ff862faa1d94.../.bio_climatic_variables_loader.params.json
    Hash path    data_processed/ff862faa1d94.../.bio_climatic_variables_loader.hash.json
    Spec path    data_processed/ff862faa1d94.../.bio_climatic_variables_loader.spec.json

    Params
    {
      "variable": 4
    }
