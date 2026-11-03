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

`import pygeodata.cli` must stay fast, and the goal is to make it faster still. The CLI must not load numpy, affine, graphviz or other heavy packages. Check after any change that touches imports:

```
python -X importtime -c "import pygeodata.cli" 2>&1 | sort -t'|' -k2 -n | tail -15
python -c "import sys, pygeodata.cli; print(sorted(m for m in ('numpy', 'affine', 'graphviz', 'filelock', 'pandas', 'xarray', 'rasterio', 'pyproj', 'geopandas', 'shapely', 'matplotlib') if m in sys.modules))"
time python -c "import pygeodata.cli"
```

Baseline (2026-11-03, macOS, Python 3.14, warm cache): about 290 ms in-process, or about 390 ms of process wall time. For comparison, `python -c "import click"` takes about 70 ms. Heavy modules loaded today: numpy, affine, graphviz and filelock (filelock ≥ 3.29 imports asyncio, about 100 ms).

An import a module needs only on a code path the CLI doesn't use goes inside the function, with a one-line comment giving the reason. Otherwise imports stay at module level.

## Collaboration

Ask before assuming. Prefer pausing to check direction — on ambiguous requirements, design choices with more than one reasonable answer, or before a change with a wide blast radius — over pushing ahead on a guess. It's fine to ask multiple clarifying questions rather than resolve them silently.
