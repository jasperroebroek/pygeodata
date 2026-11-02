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

## Collaboration

Ask before assuming. Prefer pausing to check direction — on ambiguous requirements, design choices with more than one reasonable answer, or before a change with a wide blast radius — over pushing ahead on a guess. It's fine to ask multiple clarifying questions rather than resolve them silently.
