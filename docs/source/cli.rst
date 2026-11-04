CLI reference
=============

The ``pygeodata`` command-line interface gives you direct access to the code
registry, dep-tree snapshots, cache entries, and version history without
starting the browser.

Before running any command, the CLI imports the settings modules and prints
where each was loaded from:  the file given by the global ``--settings`` option
(``pygeodata --settings <path> <command>``), plus ``pygeodata_settings.py`` and
``src/pygeodata_settings.py`` in the current directory — all of them, so
configuration can be split across files.  The settings modules are plain Python
and typically call ``get_config().update(...)``, so the CLI sees exactly the
configuration your Python code uses.  Without a settings module the built-in
defaults apply; a module that fails to import aborts the command, and so do two
modules that set the same config key — an ambiguous configuration is never
resolved silently.

All commands that accept a hash value also accept a **truncated prefix** —
any unique prefix is resolved automatically.  An ambiguous prefix produces a
clear error listing the candidates.

Where classes come from
-----------------------

Staleness compares each cache entry with the project's *current* classes.  By
default the CLI reads them **from source**: it parses the project's ``.py``
files — those in the working directory and in its packages (directories with an
``__init__.py``; ``tests``, virtual environments, build and data directories are
skipped) — and derives every class's source hash and dependency tree without
running any project code.  This is fast, needs none of the project's heavy
dependencies, and works on a read-only mirror of a project.

``--import-all`` imports those modules instead and uses the imported classes.
It also warns about every class where the imports and the source disagree —
for example a class created dynamically, which source reading cannot see.

Source hashes do not depend on the Python version, so a mirror may run a
different Python than the machine that wrote the cache.

``clean-cache`` always imports the project: it deletes entries, so it refuses a
real run when any project module fails to import, since that module's entries
would otherwise look unregistered.


Commands
--------

.. code-block:: text

    pygeodata
    │
    ├── analyse
    │     Analyse the project's classes: facts, file and line, dependency tree
    │     hash and problems.  Reads the source by default.  The settings
    │     modules' output goes to stderr for this command.
    │     --import-all            Import the project's modules instead of
    │                             reading them from source
    │     --json                  Print the result as JSON and nothing else,
    │                             whatever the project prints at import
    │
    ├── browse
    │     Launch the registry browser web UI. Blocks until the first full
    │     state is ready by default.  Reads the project's classes from
    │     source; the server never imports project code.
    │     --port INTEGER          Port to listen on (0 = random free port)
    │     --fast / --no-fast      Serve the UI immediately from the catalog
    │                             bootstrap while the full state builds
    │                             (default: block until ready)
    │     --reconcile-interval FLOAT
    │                             Minutes between automatic catalog checks
    │                             and, with --import-all, project imports
    │                             (0 disables; default: 5)
    │     --import-all            Read the classes by importing the project in
    │                             a subprocess instead of from source.
    │                             Imports EVERY .py file at the project root
    │                             and in its packages: a root-level script
    │                             without an if __name__ == "__main__" guard
    │                             RUNS, on every import
    │     --read-only / --writable
    │                             Never write into the project (a synced
    │                             mirror), or allow it.  Default: read-only
    │                             when the catalog was written from another
    │                             project root or catalog_enabled is False
    │
    ├── catalog
    │   Inspect and rebuild the .catalog.json cache catalog.
    │   │
    │   ├── build
    │   │     Index every cache entry into the catalog JSON.
    │   │     --cache-root PATH       Cache root to index (repeatable); defaults to
    │   │                             the configured cache roots
    │   │     --prune / --no-prune    Also drop entries whose cache entry no longer
    │   │                             exists (default: prune)
    │   │     --import-all            Import the project's modules first
    │   │     --verbose-import        Print each module imported (with --import-all)
    │   │
    │   └── info
    │         Show where the catalog was written (root, host, pygeodata and
    │         Python version), whether this machine may write it, its entry
    │         counts per type, and how it differs from disk.
    │
    ├── classes
    │     Show staleness for the project's classes and those with cache
    │     entries, grouped by type.  Exits 1 when anything is stale.
    │     --import-all            Import the project's modules instead of
    │                             reading them from source
    │     --verbose-import        Print each module imported (with --import-all)
    │     --type TEXT             Filter to one object type: Data, Table,
    │                             Figure, or TrackedObject (case-insensitive)
    │     --hide-current          Omit classes with no staleness
    │     --full-hash             Print full source hashes
    │     --registry TEXT         Path to .source/ registry root
    │
    ├── clean-cache
    │     Remove stale or invalid cache entries.  Always imports the project,
    │     and refuses a real run when any project module fails to import.
    │     --verbose-import          Print each module imported
    │     --dry-run / --no-dry-run  Actually delete files with --no-dry-run
    │                               (default is a dry run)
    │     --delete-unregistered / --no-delete-unregistered
    │                               Keep entries whose class is no longer in the
    │                               registry with --no-delete-unregistered
    │     --check-class-params      Also validate TrackedObject subclasses held as
    │                               parameters (see below).  Off by default.
    │
    ├── clean-locks
    │     Remove process.lock files left behind by terminated runs.  Locks
    │     held by a running process are skipped; a dry run lists every lock
    │     file found, including any in use.
    │     --dry-run / --no-dry-run  Actually delete files with --no-dry-run
    │                               (default is a dry run)
    │
    ├── clean-source
    │     Remove orphaned code snapshots and dependency trees from .source/.
    │     Keeps the latest snapshot per class and anything referenced by a live
    │     cache entry.  Runs as a dry run by default.
    │     --dry-run / --no-dry-run  Actually delete files with --no-dry-run
    │                               (default is a dry run)
    │
    ├── export ARCHIVE [HASH_PREFIXES ...]
    │     Export cache entries to an uncompressed pygeodata archive.  Pass one
    │     selector: hash prefixes, --class, --type, or --all.
    │     --class TEXT            Export all entries of this class
    │     --type TEXT             Export all entries of one object type: Data,
    │                             Table, or Figure (case-insensitive)
    │     --all                   Export every cache entry
    │     --snapshots / --no-snapshots
    │                             Include the code and dep-tree snapshots that
    │                             reproduce the entries (default: included)
    │
    ├── import ARCHIVE
    │     Import a pygeodata export archive (compressed or not) into the
    │     current project.
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
        Inspect cache entries.
        --full-hash             Print full hashes instead of 12-char prefixes
        │
        ├── list
        │     One row per cache entry: staleness, class, type, CRS, resolution,
        │     bounds and state hash.  Paged in a terminal.
        │     --import-all        Import the project's modules instead of
        │                         reading them from source
        │     --verbose-import    Print each module imported (with --import-all)
        │     --class TEXT        Show only entries of this class
        │     --type [data|table|figure]
        │                         Show only entries of one object type
        │     --hide-stale        Omit entries with any staleness marker
        │     --params            Show the parsed parameters under each entry
        │     --filter EXPR       Filter entries (repeatable, ANDed); see below
        │     --no-pager          Print directly to stdout
        │
        └── show HASH_PREFIX
              Full detail for one cache entry: identity, staleness, spatial
              spec, co-outputs, file paths and params JSON.
              --import-all        As for list
              --verbose-import    As for list
              --params / --no-params
                                  Omit the params JSON with --no-params


``--filter`` takes ``[target:]value``, optionally prefixed with ``!`` (NOT) or
joined with ``|`` (OR).  Targets: ``class``, ``crs``, ``key``, ``value``,
``path``, ``key_group`` (default: all of them).  For example
``--filter "value:RF|value:KMEANS"`` or ``--filter "!ElevationLoader"``.


Staleness markers
-----------------

``entry list``, ``entry show`` and ``classes`` share one set of markers, and
print a legend for the ones that occur:

+------------+----------------------------------------------------------+
| Marker     | Meaning                                                  |
+============+==========================================================+
| ``F``      | Entry written by another pygeodata cache format          |
+------------+----------------------------------------------------------+
| ``S``      | Entry is stale: its dependency tree changed              |
+------------+----------------------------------------------------------+
| ``C``      | Class's source changed since it last ran                 |
+------------+----------------------------------------------------------+
| ``D``      | Class's dependencies changed since it last ran           |
+------------+----------------------------------------------------------+
| ``?``      | Unknown: the class is not found in the project           |
+------------+----------------------------------------------------------+
| *(blank)*  | Up to date                                               |
+------------+----------------------------------------------------------+

Entries use ``F``, ``S`` and ``?``; classes use ``C``, ``D`` and ``?``.

An entry or class not found in the project is never shown as current: the
``.source/`` registry can still show it stale, but it only knows the code that
last ran, not the code on disk.


Example output
--------------

A project with ``RegionTable`` edited since its entries were written,
``SummaryTable`` depending on it, and cache entries of ``OldTable``, a class
that no longer exists.

``entry list``:

.. code-block:: text

    ? OldTable      Table  EPSG:3035  1000m  33.5° N, 9.6° W → 58.2° N, 74.0° E  0f3c9a1b2c3d
    S RegionTable   Table  EPSG:3035  1000m  33.5° N, 9.6° W → 58.2° N, 74.0° E  92db8dbf65ab
    S RegionTable   Table  EPSG:3035  1000m  33.5° N, 9.6° W → 58.2° N, 74.0° E  d69e5318415e
    S SummaryTable  Table  EPSG:3035  1000m  33.5° N, 9.6° W → 58.2° N, 74.0° E  e731659fdfd5

    S stale: its dependency tree changed  ? unknown: class not found in the project

``classes``:

.. code-block:: text

    ── Table (3) ──
      ? OldTable      src:—  deps:—
      C RegionTable   src:2fc288140504  deps:—
      D SummaryTable  src:aae5ee56b09b  deps:[RegionTable]

    C source changed since it last ran  D dependencies changed since it last ran  ? unknown: class not found in the project

``entry show e731``:

.. code-block:: text

    Class        SummaryTable
    Type         Table
    State hash   e731659fdfd5
    Instance     04c57a022268
    Dep hash     20203262c88e
    Staleness    stale: its dependency tree changed

    CRS          EPSG:3035
    Resolution   1000m
    Shape        (4100, 5000)
    Bounds       33.5° N, 9.6° W → 58.2° N, 74.0° E

    Output       tables/e731659fdfd5.../summary_table.html
    Params path  tables/e731659fdfd5.../parameters.json
    Hash path    tables/e731659fdfd5.../meta.json
    Spec path    tables/e731659fdfd5.../spec.json

``catalog info``:

.. code-block:: text

    Catalog      .catalog.json
    ID           021031be84f944a89ff66ec45663f962  (version 1)
    Written      2026-11-04 14:06 by pygeodata 0.5.0 (Python 3.14.4) on laptop.local
    From root    /home/user/project
    Mode         writable
    Entries      4  (Table: 4)
