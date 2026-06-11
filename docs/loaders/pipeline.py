"""
Loader definitions used by the pygeodata tutorial notebooks.

These live in a separate importable module because pygeodata reads class source
code via inspect.getsource() for cache invalidation — classes defined
interactively in a notebook cannot be inspected this way.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import CRS
from rasterio.enums import Resampling

from pygeodata.api import load
from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.processors.reprojection import Reprojector
from pygeodata.processors.rasterizer import Rasterizer
from pygeodata.spec import SpatialSpec


# ---------------------------------------------------------------------------
# Notebook 01 – Getting Started
# ---------------------------------------------------------------------------

@dataclass
class ElevationLoader(Data):
    """Reproject the bundled DEM (EPSG:3035, 1 km) to the target spec."""

    src: str = 'data/elevation.tif'

    @property
    def processor(self):
        return Reprojector(src_path=self.src)


# ---------------------------------------------------------------------------
# Notebook 02 – Building a Pipeline
# ---------------------------------------------------------------------------

@dataclass
class WaterTableDepthLoader(Data):
    """Reproject the water-table depth raster to the target spec."""

    src: str = 'data/wtd.tif'

    @property
    def processor(self):
        return Reprojector(src_path=self.src)


class CountryMaskLoader(Data):
    """Rasterize country polygons to a binary land mask (1 = land, 0 = sea)."""

    @property
    def processor(self):
        return Rasterizer(
            src_path=Path('data/countries/ne_110m_admin_0_map_units.shp'),
            values=1,
            fill_value=0,
        )


@dataclass
class LandWaterTableDepth(Data):
    """Water-table depth restricted to land pixels only."""

    wtd: WaterTableDepthLoader
    mask: CountryMaskLoader

    ext = 'tif'
    driver = RioXArrayDriver()

    def _process(self, spec: SpatialSpec) -> None:
        import rioxarray  # noqa: F401

        wtd_da = load(self.wtd, spec)
        mask_da = load(self.mask, spec)
        wtd_da.where(mask_da == 1).rio.to_raster(self.get_processed_path(spec))


# ---------------------------------------------------------------------------
# Notebook 03 – Reprojection
# ---------------------------------------------------------------------------

class LUH2C3AnnLoader(Data):
    """C3 annual crops fraction from LUH2 (no embedded CRS → supply EPSG:4326)."""

    @property
    def processor(self):
        return Reprojector(
            src_path='netcdf:data/luh2.nc:c3ann',
            src_crs=CRS.from_epsg(4326),
            forced_read=True,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )


@dataclass
class ElevationFeetLoader(Data):
    """Elevation reprojected from metres to feet via a scale factor."""

    src: str = 'data/elevation.tif'

    @property
    def processor(self):
        return Reprojector(
            src_path=self.src,
            scales=3.28084,
            resampling=Resampling.bilinear,
        )


# ---------------------------------------------------------------------------
# Notebook 04 – Rasterization
# ---------------------------------------------------------------------------

class CountryIndexLoader(Data):
    """Rasterize countries, burning the integer row index into each pixel."""

    @property
    def processor(self):
        return Rasterizer(
            src_path=Path('data/countries/ne_110m_admin_0_map_units.shp'),
            values='index',
            fill_value=-1,
        )


class CountryPopEstLoader(Data):
    """Rasterize country polygons, burning the POP_EST attribute."""

    @property
    def processor(self):
        return Rasterizer(
            src_path=Path('data/countries/ne_110m_admin_0_map_units.shp'),
            values='POP_EST',
            fill_value=np.nan,
        )


class ClippedCountryMaskLoader(Data):
    """Land mask using load_df to clip features to the target spec bounds."""

    @property
    def processor(self):
        import geopandas as gpd

        def _load_clipped(spec: SpatialSpec) -> gpd.GeoDataFrame:
            b = spec.bounds
            gdf = gpd.read_file(
                'data/countries/ne_110m_admin_0_map_units.shp'
            ).to_crs(spec.crs)
            return gdf.cx[b.left:b.right, b.bottom:b.top]

        return Rasterizer(load_df=_load_clipped, values=1, fill_value=0)


# ---------------------------------------------------------------------------
# Notebook 05 – Custom Processing
# ---------------------------------------------------------------------------

@dataclass
class MeanStdLoader(Data):
    """
    Co-output example: one _process call writes both the mean and std rasters.

    Yielding sibling loaders from _process tells pygeodata that a single run
    produced multiple outputs and all should have their hashes written.
    """

    wtd: WaterTableDepthLoader
    mask: CountryMaskLoader
    stat: str  # 'mean' or 'std'

    ext = 'tif'
    driver = RioXArrayDriver()

    def _process(self, spec: SpatialSpec):
        import rioxarray  # noqa: F401

        wtd_da = load(self.wtd, spec).where(load(self.mask, spec) == 1)

        mean_loader = MeanStdLoader(wtd=self.wtd, mask=self.mask, stat='mean')
        std_loader  = MeanStdLoader(wtd=self.wtd, mask=self.mask, stat='std')

        mean_path = mean_loader.get_processed_path(spec)
        std_path  = std_loader.get_processed_path(spec)
        mean_path.parent.mkdir(parents=True, exist_ok=True)
        std_path.parent.mkdir(parents=True, exist_ok=True)

        mean_val = float(wtd_da.mean())
        std_val  = float(wtd_da.std())
        import xarray as xr
        for val, path in ((mean_val, mean_path), (std_val, std_path)):
            da_out = xr.full_like(wtd_da, fill_value=val)
            da_out = da_out.rio.write_crs(wtd_da.rio.crs)
            da_out.rio.to_raster(path)

        yield mean_loader
        yield std_loader


class WTDFigure(Data):
    """
    Figure subclass: caches a matplotlib map of water-table depth.
    Stored in path_figures rather than path_cache.
    """

    wtd: WaterTableDepthLoader = None
    mask: CountryMaskLoader = None
    ext = 'png'

    def __init__(self):
        self.wtd = WaterTableDepthLoader()
        self.mask = CountryMaskLoader()

    def _process(self, spec: SpatialSpec) -> None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import rioxarray  # noqa: F401

        da = load(self.wtd, spec).where(load(self.mask, spec) == 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        da.plot(ax=ax, cmap='Blues_r', vmin=0, vmax=50,
                cbar_kwargs={'label': 'Water table depth (m)'})
        ax.set_title('Water Table Depth — Australia')
        fig.tight_layout()
        fig.savefig(self.get_processed_path(spec), dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Notebook 06 – Custom Drivers / _load
# ---------------------------------------------------------------------------

@dataclass
class WTDStatsLoader(Data):
    """
    Custom _load: reads a cached .npz stats file and returns a dict.
    Demonstrates overriding _load to handle a non-standard format.
    """

    wtd: WaterTableDepthLoader
    mask: CountryMaskLoader

    ext = 'npz'

    def _process(self, spec: SpatialSpec) -> None:
        import rioxarray  # noqa: F401

        da = load(self.wtd, spec).where(load(self.mask, spec) == 1)
        values = da.values
        path = self.get_processed_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=values.mean(), std=values.std(),
                 min=values.min(), max=values.max())

    def _load(self, path):
        data = np.load(path)
        return {k: float(data[k]) for k in data.files}


# ---------------------------------------------------------------------------
# Notebook 02 – dependency injection demo
# ---------------------------------------------------------------------------

@dataclass
class Red(Data):
    """Stub: returns a fixed reflectance value (no real processing)."""

    year: int

    def _process(self, spec: SpatialSpec) -> None:
        return

    def _load(self, path):
        return 0.05

    driver = RioXArrayDriver()


@dataclass
class NIR(Data):
    """Stub: returns a fixed near-infrared value (no real processing)."""

    year: int

    def _process(self, spec: SpatialSpec) -> None:
        return

    def _load(self, path):
        return 0.5

    driver = RioXArrayDriver()


@dataclass
class NDVI(Data):
    """NDVI computed internally from Red and NIR stubs."""

    year: int

    def _process(self, spec: SpatialSpec) -> None:
        return

    def _load(self, path):
        red = load(Red(year=self.year), spec)
        nir = load(NIR(year=self.year), spec)
        return (nir - red) / (nir + red)

    driver = RioXArrayDriver()


@dataclass
class NDVIInjection(Data):
    """NDVI computed from Red and NIR passed as constructor parameters."""

    red: Red
    nir: NIR

    def _process(self, spec: SpatialSpec) -> None:
        return

    def _load(self, path):
        red = load(self.red, spec)
        nir = load(self.nir, spec)
        return (nir - red) / (nir + red)

    driver = RioXArrayDriver()
