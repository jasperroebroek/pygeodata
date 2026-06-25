# pygeodata — Deep Dive
### Architecture, internals, and design decisions

---

## Slide 1 — Motivation: Declarative Geospatial Pipelines

Geospatial analysis should be declarative: you describe *what* you want, and the framework figures out *how* to produce it. You declare a spatial specification once — a CRS, a resolution, a bounding box — and every operation adapts automatically: reprojection, resampling, tiling. Nothing is hard-coded to a particular grid.

In practice, local Python pipelines force you to abandon that model: manual reprojection calls, hand-rolled caching, no record of which version of your code produced a given file.

**pygeodata restores the declarative model.** You declare a `SpatialSpec` once. Your whole pipeline — data loaders, figures, derived products — receives that spec and knows how to adapt. The framework handles the rest: deterministic caching, code versioning, dependency tracking, and a browser UI to inspect everything.

```
# You set SPEC once, at project startup
SPEC = SpatialSpec(
    crs=CRS.from_string('EPSG:3035'),
    transform=Affine.translation(left, top) * Affine.scale(1000, -1000),
    shape=(dst_height, dst_width),
)
get_config().update(spec=SPEC)

# Then just ask for things
elevation = load(ElevationLoader())           # spec injected from config
slope     = load(SlopeLoader())               # depends on elevation internally
lai       = load(LAILoader('mean'))           # independent
```

---

## Slide 2 — Declarative Programming

pygeodata is built around the **declarative paradigm**: you describe your desired outputs, and the framework is the orchestrator that figures out how to produce them.

This is the same contract that SQL, React, and GEE offer in their own domains:

| Domain | You declare | Orchestrator decides |
|--------|-------------|----------------------|
| SQL | `SELECT name FROM users WHERE active` | query plan, index usage, join order |
| React | component tree + state | DOM diffing, re-render scheduling |
| GEE | `image.reproject(crs, scale)` | tile splitting, server-side compute |
| **pygeodata** | `load(SlopeLoader(), spec)` | cache check, dependency resolution, reprojection |

In practice this means:

```
┌─────────────────────────────────────────────────────────────────┐
│  What you write               What the framework does           │
│  ──────────────               ────────────────────────          │
│  class SlopeLoader(Data):     - parse AST, compute source hash  │
│      def _process(self, s):   - track ElevationLoader as dep    │
│          process(             - check if cache is still valid   │
│              ElevationLoader())  - if not: recompute upstream   │
│          gdal.DEMProcessing(…)   - acquire lock, run _process   │
│                               - write hash, params, spec files  │
│  load(SlopeLoader())          - return xarray.DataArray         │
└─────────────────────────────────────────────────────────────────┘
```

You never write: "check if the elevation file exists, if not reproject it, then check if the slope file is stale relative to the elevation file, if so recompute it…". You write `SlopeLoader` and call `load`. The framework owns the *how*.

---

## Slide 2 — The Two Base Classes

Everything in pygeodata is a `Data` or a `Figure`, both subclassing `Artifact → TrackedObject`.

```
TrackedObject          ← registry, hashing, dependency tracking
    └── Artifact       ← caching, processing, file paths
            ├── Data   ← geospatial datasets; has a .load() method
            └── Figure ← output plots; ext='png' by default
```

| Class | Cache root | Has `.load()`? | Default ext |
|-------|-----------|----------------|-------------|
| `Data` | `data_processed/` | Yes | from processor |
| `Figure` | `figures/` | No | `png` |

You define a subclass, set parameters as instance attributes (dataclasses work perfectly), and implement either `_process()` or assign a `processor`. That's it.

```python
# Minimal: delegate processing to a built-in Reprojector
@dataclass
class LAILoader(Data):
    moment: str

    @property
    def processor(self):
        return Reprojector(
            PATH_DATA / f'LAI_{self.moment}.vrt',
            resampling=Resampling.average,
        )

# Custom: full control over processing
class SlopeLoader(Data):
    driver = RioXArrayDriver()

    def _process(self, spec: SpatialSpec) -> None:
        process(ElevationLoader(), spec)          # trigger dependency
        path_dem = ElevationLoader().ensure_processed_path(spec=spec)
        path_dst = self.ensure_processed_path(spec=spec)
        gdal.DEMProcessing(path_dst, path_dem, 'slope', ...)
```

---

## Slide 3 — The SpatialSpec

`SpatialSpec` is the single parameter that flows through the entire pipeline. It is a frozen dataclass holding:

| Field | Meaning |
|-------|---------|
| `crs` | Coordinate reference system (pyproj CRS) |
| `transform` | Affine transform — origin + pixel size |
| `shape` | (height, width) in pixels |

From these three, you can derive resolution, bounds, and extent. The spec can be **partially defined** (just a CRS, no transform or shape). Processors can resolve it themselves — for example, `Reprojector.resolve_spec()` reads the source file and computes the appropriate transform for the target CRS.

```
Partially defined spec           Processor resolves it
SpatialSpec(crs=EPSG:4326)  →   reads src raster → calculates target shape
                                 returns fully defined SpatialSpec
```

The spec participates in the **state hash**: the same class + same params processed at a different spec produce a different cache entry. You can trivially run the same analysis at multiple resolutions or extents without any conflict.

---

## Slide 4 — Hashing: Four Levels

pygeodata maintains four nested levels of identity.

```
┌──────────────────────────────────────────────────────────────────────┐
│  source_hash   = SHA256(AST dump of the class)                       │
│                 changes when code changes                            │
│                                                                      │
│  dep_tree_hash = SHA256(source_hash of all transitive dependencies)  │
│                 changes when any upstream class changes              │
│                                                                      │
│  instance_hash = SHA256(dep_tree_hash + params)                      │
│                 spec-independent; stable identifier for an instance  │
│                                                                      │
│  state_hash    = SHA256(instance_hash + spec)                        │
│                 unique per (class + params + spec) triple            │
└──────────────────────────────────────────────────────────────────────┘
```

The **source hash** uses the AST, not raw text. Whitespace changes, comments, and docstring edits do not invalidate the cache. Only semantic changes do.

The **dependency tree hash** propagates changes upward: if `ElevationLoader` changes, `SlopeLoader`'s dep_tree_hash changes even if `SlopeLoader` itself is untouched — because its output now depends on different code.

The **state hash** is the cache key: it is written alongside every output file and compared on every run.

---

## Slide 5 — The Dependency Tree

pygeodata tracks two kinds of dependency between classes:

**Call dependencies** — discovered by parsing the class AST and finding references to other `TrackedObject` subclasses within the method bodies. If `SlopeLoader._process` calls `ElevationLoader()`, that is a call dependency.

**Inheritance dependencies** — `cls.__mro__` filtered to `TrackedObject` subclasses. If `FigureVariance` extends `MapPanelFigureBase`, that is an inheritance dependency.

Both are combined into a `{nodes, tree}` JSON structure that is stored in `.source/snapshots/`:

```json
{
  "nodes": {
    "SlopeLoader":     {"hash": "abc...", "object_type": "Data"},
    "ElevationLoader": {"hash": "def...", "object_type": "Data"}
  },
  "tree": {
    "SlopeLoader": {
      "call_dependencies": {
        "ElevationLoader": {"call_dependencies": {}, "inheritance_dependencies": {}}
      },
      "inheritance_dependencies": {}
    }
  }
}
```

Nodes are deduplicated (flat dict). The tree is fully expanded (shared deps appear at every occurrence). This separation makes both human inspection and hash computation straightforward.

---

## Slide 6 — The Cache Layout

Every processed output lands in a **content-addressed** directory named by its state hash.

```
data_processed/
  {state_hash}/
    elevation_loader.tif      ← the actual output
    meta.json                 ← metadata + state_hash for validation
    parameters.json           ← params at time of processing
    spec.json                 ← spatial spec at time of processing
    graph.pdf                 ← runtime dependency graph (if applicable)

figures/
  {state_hash}/
    figure_slope.png
    meta.json
    ...
```

The `meta.json` file is the source of truth for cache validity. It stores:

```json
{
  "format_version": 1,
  "class_name": "ElevationLoader",
  "object_type": "Data",
  "source_hash": "abc...",
  "dependency_tree_hash": "def...",
  "instance_hash": "ghi...",
  "state_hash": "jkl...",
  "co_outputs": []
}
```

On the next run, `is_cache_valid()` reads this file and compares `state_hash` to the live computed value. If they match, the file is returned immediately. If they differ, processing runs again.

---

## Slide 7 — The Source Registry

Alongside the cache, pygeodata maintains `.source/` — a **permanent, append-only record** of every version of every class that was ever executed.

```
.source/
  code/
    {source_hash}/
      source.py       ← the class source code at that hash
      source.json     ← metadata: class_name, registered_at, object_type
  snapshots/
    {dep_tree_hash}/
      tree.json       ← full dependency tree at that hash
      graph.pdf       ← rendered dependency graph
```

`source.py` is written once and never overwritten — it is content-addressed. `source.json` has its `registered_at` timestamp updated whenever a previously-seen hash re-activates after a different hash was active (i.e., after a revert). This keeps the mtime meaningful for browsing version history.

`write_registry()` is called automatically during every `process()` run.

---

## Slide 8 — The Process Lifecycle

Here is the full sequence when you call `artifact.process(spec)` or `load(artifact, spec)`:

```
process(LAILoader('mean'), spec)
    │
    ├─ resolve_spec(spec)              ← fill in shape/transform if partial
    ├─ is_processed(spec)?
    │   ├─ is_cache_valid(spec)?       ← compare state_hash on disk vs live
    │   └─ processed_path_exists()?
    │   └─ YES → return immediately
    │
    ├─ acquire file lock               ← safe for parallel runs
    ├─ _process(spec)                  ← user code or Processor.__call__
    │
    └─ for each produced artifact:
        ├─ update_registry()           ← write .source/ entries
        ├─ write_parameters(spec)      ← parameters.json
        ├─ write_spec(spec)            ← spec.json
        └─ write_cache_metadata(spec)  ← meta.json
```

The file lock (`filelock`) ensures that parallel workers don't race to write the same output. After locking, `is_processed` is checked again inside the lock.

---

## Slide 9 — The Co-Output Pattern

Sometimes a single expensive computation produces multiple output files — for example, fitting a regression model yields beta coefficients, standard errors, and p-values simultaneously. It would be wasteful (and incorrect) to re-run the model for each.

**pygeodata supports co-outputs**: `_process()` can `yield` sibling artifacts instead of writing only its own output.

```python
@dataclass
class SFTRegressionLoader(Data):
    method: SpaceForTimeMethod
    size: int
    regression_param: RegressionParameter   # BETA, BETA_SE, or P

    def _process(self, spec):
        # All three are produced together
        results = run_regression(self.method, self.size, spec)
        write_array(SFTRegressionLoader(self.method, self.size, RegressionParameter.BETA), results.beta)
        write_array(SFTRegressionLoader(self.method, self.size, RegressionParameter.BETA_SE), results.se)
        write_array(SFTRegressionLoader(self.method, self.size, RegressionParameter.P), results.p)

        yield SFTRegressionLoader(self.method, self.size, RegressionParameter.BETA)
        yield SFTRegressionLoader(self.method, self.size, RegressionParameter.BETA_SE)
        yield SFTRegressionLoader(self.method, self.size, RegressionParameter.P)
```

`process()` deduplicates by state hash and writes hash metadata for all yielded artifacts. Each one is independently cacheable from that point on.

---

## Slide 10 — Data as Parameters

`Data` instances can be parameters of other `Data` instances. pygeodata handles this naturally: the `instance_hash` is computed recursively, and the runtime dependency graph is built by walking `get_params()`.

```python
@dataclass
class FeatureCorrelationLoader(Data):
    method: SpaceForTimeMethod
    size: int
    feature: Data         # ← another Data instance as parameter
    variable: Data        # ← another one

    def _process(self, spec):
        feat_data = load(self.feature, spec)
        var_data  = load(self.variable, spec)
        ...
```

When `feature=ElevationLoader()` changes its source hash, `FeatureCorrelationLoader`'s `instance_hash` changes automatically — because the instance hash includes the full dependency tree hash of all nested artifacts.

The runtime dependency graph (stored as `.graph.pdf`) visualises this nesting at a per-run level, showing exactly which instances (not just which classes) were involved.

---

## Slide 11 — Processors and Drivers

For common geospatial operations, pygeodata provides **Processors** — callable objects that take `(output_path, spec)` and write a file.

The built-in `Reprojector` covers the most common case: warp any raster source to the target spec using rasterio.

```python
# Processor protocol
class Processor(Protocol):
    def __call__(self, dst_path: Path, spec: SpatialSpec) -> None: ...

# Built-in Reprojector
Reprojector(
    src_path='data/LAI_mean.vrt',
    resampling=Resampling.average,
    dst_dtype=np.float32,
    scales=0.1,
)
```

**Drivers** are the read side — callables that take a path and return data:

```python
class Driver(Protocol):
    default_ext: str
    def __call__(self, path: Path) -> Any: ...

# Built-ins
RioXArrayDriver()    # reads GeoTIFF → xarray.DataArray with spatial metadata
GeoPandasDriver()    # reads vector files → geopandas.GeoDataFrame
```

Processors can also implement `resolve_spec()` to fill in a partial spec from the source file.

---

## Slide 12 — Parallel Processing

The `parallel` module builds a Dask delayed computation graph so artifacts and their dependencies can be scheduled concurrently:

```python
from pygeodata.parallel import build_dask_graph

graph = build_dask_graph(LAILoader('mean'), spec=SPEC)
graph.compute()   # execute with dask.distributed or the default scheduler
```

Each artifact call acquires its own file lock on the output directory. The file-lock-then-recheck pattern ensures idempotency: if two workers race, the second finds the cache valid after acquiring the lock and exits without re-running.

---

## Slide 13 — The Registry Browser

`pygeodata browse` launches a local web server exposing a dashboard that shows the full state of a project's cache and source registry.

**Classes view** — all `TrackedObject` subclasses found in the registry, with staleness indicators:

```
Class          | Type   | Source stale? | Deps stale? | Dependencies
ElevationLoader| Data   | ✓ current     | ✓ current   | —
SlopeLoader    | Data   | ✓ current     | ✓ current   | ElevationLoader
LAILoader      | Data   | ✓ current     | ✓ current   | —
FigureSlope    | Figure | ⚠ changed     | ⚠ changed   | SFTRegressionLoader, ...
```

**Entries view** — every (class × params × spec) triple that was ever processed:

- Parameters rendered as a searchable table (nested dicts, linked artifact references)
- Spatial spec: CRS, resolution, shape, bounds (reprojected to lat/lon for readability)
- Direct link to the output file
- Co-output siblings listed inline
- Cache validity indicator

**Code view** — browse the full version history of any class. Every saved `source.py` is shown with timestamps, allowing you to see exactly what code produced each result.

---

## Slide 14 — Cache Management

Four maintenance operations keep the cache and source registry healthy over time.

**`clean_cache(dry_run=True)`** — walks `data_processed/` and `figures/`, and deletes any directory whose state hash no longer matches the live hash of the corresponding class. Handles:
- Missing hash files
- Format version mismatches
- Classes no longer in the registry (optionally, with confirmation)
- Multiple hash files in one directory (conflict resolution)

**`clean_source_registry(dry_run=True)`** — walks `.source/` and removes orphaned snapshots. Keeps the latest snapshot per class and anything referenced by a live cache entry. Also available as `pygeodata clean-source` on the CLI.

**`clean_registry(dry_run=True)`** — walks `.source/` and removes entries written by a different `FORMAT_VERSION`.

**`rebuild_registry()`** — wipes and rewrites all `.source/` entries from the currently-loaded classes.

**`load_from_hash(state_hash, filename=None, params=None)`** — loads a cached output by its (possibly truncated) state hash without re-running `process`. Useful for scripted access to specific cached outputs:

```python
from pygeodata import load_from_hash

da = load_from_hash('cc71cf42816b')
```

---

## Slide 15 — Real-World Example: space-for-time

The `space-for-time` project analyses the LAI–LST relationship across Europe using the space-for-time substitution approach. It has ~20 `Data` classes and ~15 `Figure` classes.

**One spec, declared once:**

```python
SPEC = SpatialSpec(
    crs=CRS.from_string('EPSG:3035'),
    transform=Affine.translation(left, top) * Affine.scale(1000, -1000),
    shape=(dst_height, dst_width),     # ~6000 × 7000 pixels over Europe
)
get_config().update(spec=SPEC, path_cache=PATH_DATA_PROCESSED, ...)
```

**The full pipeline in ~30 lines:**

```python
process(BenchmarkSlopeLoader(min_samples_leaf=25))

for size in [25, 81, 625, 2500, 10000, 0]:
    for method in SpaceForTimeMethod:
        process(SFTRegressionLoader(method=method, size=size, ...))
        for feature in [BioClimaticVariablesLoader(1), ElevationLoader(), ...]:
            process(FeatureCorrelationLoader(method=method, size=size, feature=feature, ...))
            process(VarianceLoader(method=method, size=size, feature=feature))
```

**What you get for free:**
- Change `BenchmarkSlopeLoader` → only that entry and its dependents recompute
- Change the spec resolution from 1000m to 500m → entirely separate cache, both coexist
- Open the registry browser → see every result, when it was computed, with what code
- `FigureVariance(feature=ElevationLoader())` — the figure's hash includes ElevationLoader's code hash, so changing the DEM source invalidates the figure automatically
