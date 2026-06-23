# pygeodata

**Reproducible geospatial pipelines, locally.**

pygeodata is built around the declarative programming principle: you should be able to describe *what* data you want, not *how* to produce it. You declare a spatial spec — a CRS, transform and shape — once. Every dataset and figure in your project receives that spec and knows how to adapt. The framework is the orchestrator: it decides whether to compute or return from cache, resolves reprojection, propagates invalidation when code changes, and keeps a permanent record of what ran and when.

This is the declarative model that SQL offers for databases, applied to local geospatial analysis in Python.

```python
from pygeodata import Data, SpatialSpec, get_config, load
from pygeodata.processors import Reprojector
from dataclasses import dataclass

get_config().update(spec=SpatialSpec.from_raster_file('reference.tif'))

@dataclass
class ElevationLoader(Data):
    @property
    def processor(self):
        return Reprojector('data/elevation.tif')

@dataclass
class SlopeLoader(Data):
    def _process(self, spec):
        process(ElevationLoader(), spec)
        gdal.DEMProcessing(self.ensure_processed_path(spec), ElevationLoader().get_processed_path(spec), 'slope')

# Both lines check the cache, recompute only what's stale, and return the result.
# Change ElevationLoader's code → both caches are invalidated automatically.
elevation = load(ElevationLoader())
slope     = load(SlopeLoader())
```

## How it works

Every `Data` or `Figure` class is fingerprinted by hashing the AST of its source code, the ASTs of all its transitive dependencies, and its parameter values. That fingerprint, combined with the spatial spec, is the cache key. If it matches what's on disk, the cached file is returned. If not, the class is reprocessed and the new fingerprint is written.

The hash is AST-based, not text-based: reformatting your code or editing a comment never triggers a rerun.

```
source_hash      SHA256(AST of the class)
dep_tree_hash    SHA256(source_hash of all transitive dependencies)
instance_hash    SHA256(dep_tree_hash + params)           
state_hash       SHA256(instance_hash + spec)             
```

Everything lands in a content-addressed store:

```
data_processed/
  {state_hash}/
    elevation_loader.tif
    meta.json                       ← cache key + metadata
    parameters.json                 ← params at time of run
    spec.json                       ← spec at time of run

.source/
  code/{source_hash}/
    source.py                       ← every version of every class
    source.json
  snapshots/{dep_tree_hash}/
    tree.json                       ← full dependency tree at time of run
    graph.pdf
```

## Defining loaders

Subclass `Data` for datasets, `Figure` for plots. Use a `processor` property for standard reprojection/rasterization, or override `_process` directly for anything else. Parameters are plain instance attributes — dataclasses work perfectly.

```python
# Delegate to the built-in Reprojector
@dataclass
class LAILoader(Data):
    moment: str   # 'min', 'mean', 'max', ...

    @property
    def processor(self):
        return Reprojector(
            f'data/lai/LAI_{self.moment}.vrt',
            resampling=Resampling.average,
            scales=0.1,
        )

# Full control
class SlopeLoader(Data):
    driver = RioXArrayDriver()

    def _process(self, spec):
        process(ElevationLoader(), spec)
        gdal.DEMProcessing(
            self.ensure_processed_path(spec),
            ElevationLoader().get_processed_path(spec),
            'slope',
        )
```

### The SpatialSpec

`SpatialSpec` holds the CRS, an affine transform, and a pixel shape. It can be partially defined — just a CRS, no transform — and processors can resolve it from the source file. The spec flows through the pipeline automatically; you set it once in config and never pass it manually unless you need to.

```python
# From an existing raster
spec = SpatialSpec.from_raster_file('reference.tif')

# Or build it explicitly
spec = SpatialSpec(
    crs=CRS.from_string('EPSG:3035'),
    transform=Affine.translation(left, top) * Affine.scale(1000, -1000),
    shape=(height, width),
)

get_config().update(spec=spec)

# After that, just ask for things
load(ElevationLoader())   # spec injected automatically
load(LAILoader('mean'))

# or ask for a specific spec
load(ElevationLoader(), spec)
load(LAILoader('mean'), spec)
```

### Composing pipelines

`Data` instances can be parameters of other `Data` instances. The full dependency graph is tracked — changing any upstream class invalidates all downstream caches automatically.

```python
@dataclass
class FeatureCorrelationLoader(Data):
    feature: Data    # e.g. ElevationLoader()
    variable: Data

    def _process(self, spec):
        feat = load(self.feature, spec)
        var  = load(self.variable, spec)
        # ... compute correlation ...
```

### Co-outputs

When one computation produces multiple outputs — a regression that yields a slope, standard error, and p-value simultaneously — `_process` can yield sibling artifacts instead of returning. Each yielded artifact gets its own cache entry and is independently re-usable.

```python
@dataclass
class RegressionLoader(Data):
    param: Literal['beta', 'se', 'p']

    def _process(self, spec):
        beta, se, p = run_regression(spec)
        write(RegressionLoader('beta'), beta)
        write(RegressionLoader('se'), se)
        write(RegressionLoader('p'), p)
        yield RegressionLoader('beta')
        yield RegressionLoader('se')
        yield RegressionLoader('p')
```

### Figures

`Figure` works identically to `Data` — same caching, same hashing, same registry — but outputs go to `figures/` and the default extension is `png`. There is no `load()` method; figures are outputs only.

```python
@dataclass
class FigureElevation(Figure):
    def _process(self, spec):
        da = load(ElevationLoader(), spec)
        da.plot()
        plt.savefig(self.ensure_processed_path(spec), dpi=300)
        plt.close()
```

## The registry browser

`pygeodata browse` opens a local web UI that shows the full state of your project — classes, cached results, versions of every class that run.

- **Classes** — all tracked subclasses, with staleness indicators when code or dependencies have changed since the last run
- **Entries** — every (class × params × spec) combination processed, with parameter tables, spatial spec details, and links to output files
- **Code view** — the full version history of any class; browse what the code looked like when a specific result was produced

## Cache management

```python
from pygeodata import clean_cache

clean_cache(dry_run=True)    # preview what would be deleted
clean_cache(dry_run=False)   # delete stale entries
```

`clean_cache` walks `data_processed/` and `figures/` and removes any directory whose state hash no longer matches the live hash of the corresponding class. Classes that have been renamed or removed from the codebase are flagged for manual confirmation.

```python
from pygeodata import rebuild_registry

rebuild_registry()   # wipe and rewrite all .source/ entries from current classes
```

## Installation

```bash
pip install pygeodata
```

Optional extras:

```bash
pip install pygeodata[viz]        # dependency graph plots (graphviz)
pip install pygeodata[dashboard]  # registry browser (Flask)
```

## License

MIT
