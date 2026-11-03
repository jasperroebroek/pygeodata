# pygeodata — working agreement

## Formatting

Format/lint with ruff, which lives in the mamba `base` env, not the project's own `pygeodata` env:

```
/usr/local/Caskroom/mambaforge/base/bin/ruff format .
/usr/local/Caskroom/mambaforge/base/bin/ruff check .
```

Run this after making code changes, before considering a change done.

## Tests

Run the test suite with the project's own env:

```
python -m pytest -q --ignore=tests/processors
```

Skip `tests/processors` by default — it's slow. Only include it when there's a good reason (e.g. the change touches a processor/driver) or right before committing, when the full suite should run once:

```
python -m pytest -q
```

## CLI import time

`import pygeodata.cli` and `pygeodata --help` must stay fast. Neither may load numpy, affine, graphviz, filelock or any other heavy package. `tests/test_cli_import.py` guards this in a fresh interpreter, and it runs with the normal suite. To investigate a regression:

```
python -X importtime -c "import pygeodata.cli" 2>&1 | sort -t'|' -k2 -n | tail -15
time pygeodata --help
```

Baseline (2026-11-03, macOS, Python 3.14, warm cache): `import pygeodata.cli` takes about 45 ms in-process, and `pygeodata --help` about 0.1 s wall time. For comparison, `python -c "import click"` takes about 0.08 s. Before this work it was about 300 ms in-process.

How it stays light:
- `pygeodata/__init__.py` resolves its public API lazily via `lazy_loader`. The names are declared once, in `pygeodata/__init__.pyi`.
- `pygeodata/cli/__init__.py` imports only click. Each command lives in its own module under `pygeodata/cli/` and is registered in the `COMMANDS` table (`'module:attribute'` plus its short help) instead of being imported. A test keeps the short help equal to the command's docstring summary.
- An import that a module needs only on a code path the CLI doesn't take goes inside the function, with a one-line comment giving the reason. Otherwise imports stay at module level.

## Collaboration

Ask before assuming. Prefer pausing to check direction — on ambiguous requirements, design choices with more than one reasonable answer, or before a change with a wide blast radius — over pushing ahead on a guess. It's fine to ask multiple clarifying questions rather than resolve them silently.
