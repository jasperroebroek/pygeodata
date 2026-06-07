# pygeodata

**Deterministic, cached geospatial data processing for Python**

`pygeodata` lets you define geospatial data loaders as plain Python classes. Each loader describes one processing step — reprojection, rasterization, or arbitrary computation. Outputs are cached on disk and automatically invalidated when source code or parameters change, so rerunning a pipeline only does the work that is actually needed.

```python
from dataclasses import dataclass
from pygeodata import Data, SpatialSpec, get_config, load
from pygeodata.processors.reprojection import Reprojector

get_config().update(path_cache='data/processed')

@dataclass
class ElevationLoader(Data):
    src: str = 'data/elevation.tif'

    @property
    def processor(self):
        return Reprojector(src_path=self.src)

spec = SpatialSpec.from_raster_file('reference.tif')
da = load(ElevationLoader(), spec)   # returns xarray.DataArray
```

Running the same call again is a no-op — the cached output is returned immediately. Change the class body or a parameter and the cache is invalidated automatically.

## Features

- **Automatic cache invalidation** — hashes the AST of your class and all its dependencies; reformatting code never triggers a rerun
- **Parameter-driven paths** — each unique combination of constructor parameters gets its own cache directory; no collisions, no manual naming
- **Pipeline composition** — pass loader instances as parameters to other loaders; the full dependency graph is tracked
- **Built-in processors** — `Reprojector` (rasterio warp) and `Rasterizer` (rasterio features) cover the common cases
- **Custom processing** — override `_process(self, spec)` for arbitrary logic; override `_load(self, path)` for non-standard formats
- **Co-outputs** — yield sibling loaders from `_process` to write multiple outputs in one run
- **Parallel execution** — convert any loader graph to a Dask delayed graph with `build_dask_graph`
- **Registry browser** — local web UI for exploring cached entries, inspecting parameters and hashes, and opening output files

## Installation

```bash
pip install pygeodata
```

Optional extras:

```bash
pip install pygeodata[viz]        # dependency graph plots (graphviz)
pip install pygeodata[parallel]   # Dask integration
pip install pygeodata[dashboard]  # registry browser (Flask)
```

## Core concepts

### `SpatialSpec`

Describes a target raster grid: CRS, affine transform, and pixel dimensions. Derive one from any existing raster:

```python
spec = SpatialSpec.from_raster_file('reference.tif')
```

### `Data` loaders

Subclass `Data` and either set a `processor` property or override `_process` directly:

```python
@dataclass
class LandMask(Data):
    """Rasterize country polygons to a binary land/sea mask."""

    @property
    def processor(self):
        return Rasterizer(src_path='countries.shp', values=1, fill_value=0)
```

Constructor parameters automatically drive the cache path and hash. Two instances with different parameters never share a cache entry.

### Pipelines

Pass loaders as constructor parameters to compose multi-step pipelines:

```python
@dataclass
class LandElevation(Data):
    elevation: ElevationLoader
    mask: LandMask

    ext = 'tif'
    driver = RioXArrayDriver()

    def _process(self, spec):
        da = load(self.elevation, spec)
        mask = load(self.mask, spec)
        da.where(mask == 1).rio.to_raster(self.get_processed_path(spec))
```

Changing code or parameters in any upstream loader automatically invalidates `LandElevation`'s cache.

### Cache management

```python
from pygeodata import clean_cache

clean_cache(dry_run=True)   # preview stale entries
clean_cache(dry_run=False)  # delete them
```

### Registry browser

```python
from pygeodata.registry_browser import open_registry_browser
open_registry_browser()     # opens a local web UI
```

## Documentation

Full documentation including tutorials is available in `docs/`. Build with:

```bash
cd docs && make html
```

## License

MIT
