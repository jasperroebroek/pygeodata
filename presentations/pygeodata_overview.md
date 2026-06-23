# pygeodata
### Reproducible geospatial pipelines, locally

---

## Slide 1 — The Problem

**Geospatial analysis pipelines should be declarative.**

SQL follows the **declarative programming model**: you describe *what* you want, not *how* to produce it. `SELECT ... WHERE ...` tells the engine what you need; it decides how to retrieve it. The same idea applies to geospatial pipelines: you should be able to declare a spatial spec (CRS, resolution, extent) once, and have the pipeline figure out reprojection, tiling, resampling, and intermediate files on your behalf.

In practice, local geospatial analysis is the opposite:
- Manual reprojection boilerplate for every dataset
- Ad-hoc caching ("did I already run this?")
- No record of *which version of your code* produced a given file
- Re-running everything when one upstream input changes

**pygeodata brings the declarative model to local Python.** You describe your datasets and figures as classes. You set the spec once. The framework is the orchestrator — it decides whether to compute or return from cache, resolves partial specs, propagates dependency invalidation, and records what happened.

```
Imperative (traditional)          Declarative (pygeodata)
────────────────────────          ───────────────────────
reproject(src, crs, res)          load(ElevationLoader())
if not exists(output):            # spec injected, cache checked,
    run_slope(dem, output)        # slope triggers elevation automatically
load_raster(output)
```

```
┌─────────────────────────────────────────────────────────┐
│                    You define once                      │
│                                                         │
│   SPEC = SpatialSpec(crs=EPSG:3035,                     │
│                      resolution=1000m,                  │
│                      extent=Europe)                     │
│                                                         │
│   load(ElevationLoader(), SPEC)     ← just works        │
│   load(SlopeLoader(), SPEC)         ← depends on above  │
│   load(LAILoader('mean'), SPEC)     ← independent       │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 2 — How It Works

**Every dataset is a class. You declare what it is; the framework decides what to do.**

```python
@dataclass
class LAILoader(Data):
    moment: str              # ← parameter becomes part of cache key

    @property
    def processor(self):
        return Reprojector(   # ← processor receives (output_path, spec)
            PATH_DATA / f'LAI_{self.moment}.vrt',
            resampling=Resampling.average,
        )
```

When you call `load(LAILoader('mean'), spec)`:

```
┌──────────────────────────────────────────────────────────────────┐
│  Hash = SHA256(AST of class code + params + dependency tree)     │
│                           │                                      │
│            ┌──────────────┴──────────────┐                       │
│            ▼                             ▼                       │
│    Hash matches on disk?         Compute & cache                 │
│    → return cached file          → write file + hash metadata    │
└──────────────────────────────────────────────────────────────────┘
```

The hash is computed from the **AST** of your class — not just parameters. If you change your code, the hash changes. Stale caches are detected automatically.

Everything lands in a content-addressed store:

```
data_processed/
  {state_hash}/          ← hash(class code + params + spec)
    elevation_loader.tif
    meta.json
    parameters.json
    spec.json

.source/
  code/{source_hash}/    ← every version of every class, forever
    source.py
    source.json
  snapshots/{dep_tree_hash}/
    tree.json            ← full dependency graph at time of run
    graph.pdf
```

---

## Slide 3 — The Registry Browser

**A local dashboard that shows everything that was ever computed.**

```
┌──────────────────────────────────────────────────────────────┐
│  Registry Browser                                            │
│  ─────────────────                                           │
│  Classes           │  Entries (instances × specs)           │
│  ─────────         │  ────────────────────────────          │
│  ElevationLoader   │  LAILoader(moment='mean')               │
│  SlopeLoader       │    spec: EPSG:3035, 1000m, Europe       │
│  LAILoader         │    state: ✓ current                     │
│  FigureSlope       │    file: elevation_loader.tif           │
│  ...               │    params / source / dep-graph          │
│                    │                                         │
│                    │  LAILoader(moment='max')                 │
│                    │    ...                                   │
└──────────────────────────────────────────────────────────────┘
```

- **What ran**, with what parameters, at what spec
- **Whether the cache is stale** (code or dependency changed)
- **Full version history** — browse old source code for any class
- **Dependency graphs** — which loaders depend on which

`pygeodata browse` — one command, runs in the browser.
