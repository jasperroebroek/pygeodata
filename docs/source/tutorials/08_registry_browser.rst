.. _08_registry_browser:

The Registry Browser
====================

The registry browser is a local web UI for exploring the pygeodata cache
and source registry. It shows every class that has ever been processed,
every cached output with its parameters and spec, and the full version
history of each class's source code.

This tutorial covers:

- Launching the browser with ``pygeodata browse``
- The Classes view: staleness indicators
- The Entries view: parameters, cache validity, co-outputs
- The Code view: browsing which source code produced a specific result

Launching the browser
---------------------

From a terminal in your project root:

.. code-block:: bash

   pygeodata browse

This starts a local HTTP server on a random free port and opens the URL
in your default browser. The URL is printed before the browser opens:

::

   http://127.0.0.1:52341

Before starting the server, ``pygeodata browse`` auto-imports all
``.py`` files in the current directory (recursively, skipping ``venv``,
``tests``, and build directories). This populates
``TrackedObject._registry`` so the browser can resolve class names from
hash files.

Options:

.. code-block:: bash

   pygeodata browse --port 8080    # use a specific port
   pygeodata browse --no-import    # skip auto-importing .py files

From Python:

.. code-block:: python

   from pygeodata.registry_browser.serve import open_registry_browser
   open_registry_browser(port=8080)  # blocks

The browser reads ``path_cache``, ``path_figures``, and
``path_registry`` as configured in the current session. No data leaves
your machine.

Classes view
------------

The **Classes** tab lists every loader class that has at least one entry
in ``.source/code/``. Each class card shows:

- The class docstring
- The object type (``Data`` or ``Figure``)
- Count of cached entries: valid / total
- A staleness indicator: green dot if all entries are valid, yellow if
  some are stale, red if all are stale

Staleness indicators
~~~~~~~~~~~~~~~~~~~~

There are two independent dimensions of staleness:

**Source stale** — the class's own source code has changed since an
entry was written. The ``source_hash`` in the ``meta.json`` file no
longer matches the live class. Because ``source_hash`` is computed from
the class **AST**, reformatting or editing comments and docstrings does
*not* change it — only semantically meaningful edits do. Source
staleness alone does not necessarily invalidate the cache: what matters
for validity is ``dependency_tree_hash``.

**Deps stale** — the ``dependency_tree_hash`` stored in ``meta.json``
no longer matches the live hash. This means the class itself or one of
its transitive dependencies has changed in a semantically meaningful
way. Cache entries with stale ``dependency_tree_hash`` will be
reprocessed on the next ``process()`` call.

The registry browser shows both indicators independently. A class can be
source-stale (its own code changed) but deps-valid, or source-valid but
deps-stale (an upstream dependency changed).

Entries view
------------

The **Entries** tab is the main landing page. It shows every
``(class × params × spec)`` combination that has ever been processed —
one row per cached output file.

Columns:

+--------------------------+-------------------------------------------+
| Column                   | Description                               |
+==========================+===========================================+
| Class                    | Loader class name (links to the class     |
|                          | card)                                     |
+--------------------------+-------------------------------------------+
| Spec                     | CRS, resolution, shape, bounding box      |
+--------------------------+-------------------------------------------+
| Parameters               | Key–value parameter pairs for this entry  |
+--------------------------+-------------------------------------------+
| Valid                    | Green check if ``dependency_tree_hash``   |
|                          | matches live; red X if stale              |
+--------------------------+-------------------------------------------+
| File                     | Cache file path — Reveal (open in Finder) |
|                          | or Copy buttons                           |
+--------------------------+-------------------------------------------+

Filtering and search
~~~~~~~~~~~~~~~~~~~~

The search box filters by class name or parameter value. The sidebar on
the left lets you show only entries from a specific class. Entries are
rendered incrementally so large caches load without blocking the UI.

Entry detail panel
------------------

Click any row to open the **entry detail panel** on the right.

Params card
~~~~~~~~~~~

The full parameter dict as stored in ``parameters.json``. Nested ``Data``
instances appear as expandable sub-trees showing the upstream class
name, its own parameters, and its instance hash. This gives complete
provenance for any cached file without re-running the loader.

Spec card
~~~~~~~~~

CRS, affine transform, shape, resolution, and bounding box for this
specific output. Derived from ``spec.json``.

Co-outputs card
~~~~~~~~~~~~~~~

When an entry was produced by a co-output ``_process`` (see
:doc:`04_custom_processing`), the Co-outputs card lists all siblings
written in the same run. Each sibling is shown with its distinguishing
parameter (e.g. ``stat=mean`` vs ``stat=std``) and links to its own
entry detail. Each sibling has its own state hash (they differ by
parameters); every sibling's ``meta.json`` records the others in its
``co_output_hashes`` list.

Dependency graph
~~~~~~~~~~~~~~~~

The graph icon in the entry header opens the class dependency graph —
the same PDF written to ``.source/snapshots/{dep_tree_hash}/graph.pdf``.
Nodes represent classes; solid arrows are call dependencies, dashed
arrows are inheritance. Clicking a node highlights it.

File actions
~~~~~~~~~~~~

- **Reveal** — opens the cache directory in Finder/Explorer
- **Copy path** — copies the full file path to the clipboard

These are the fastest way to open a specific output in QGIS, Panoply, or
any external tool.

Code view
---------

The **Code** tab is accessible from the class card and from the entry
detail panel. It shows the full version history of a class: every unique
source text that was ever active, ordered by ``registered_at`` timestamp
(most recent first).

Each version card shows:

- The ``source_hash`` (8-character prefix)
- ``registered_at`` timestamp
- The full rendered source code

Connecting outputs to source versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``meta.json`` file stores the ``source_hash`` that was active
when the output was written. The Code view lets you cross-reference:
select a cached entry in the Entries view, note its ``source_hash`` from
the entry detail, and find the matching version card in the Code view to
see exactly what code produced it.

This is the first tool to reach for when debugging a suspicious result:
you can confirm whether the output was produced by the current code or
by a previous version.

When the registry browser is useful
-----------------------------------

**Debugging invalid caches** — the valid/invalid indicator pinpoints
which entries are stale after a code change. The source popup shows
exactly what changed.

**Auditing outputs** — the Params and Spec cards give a full provenance
record for any cached file without needing to re-run the code that
produced it.

**Understanding a pipeline** — the graph view and Classes tab give a
quick overview of the project structure, even for code you did not
write.

**Finding co-outputs** — when one ``_process`` produces multiple files,
the Co-outputs card links them all. No manual path hunting required.

**Opening files directly** — the Reveal button opens the output
directory in your file manager, saving several steps when you want to
inspect a raster in QGIS.

**Connecting results to source** — the Code view shows what code
produced a specific cached output, enabling reproducibility audits.
