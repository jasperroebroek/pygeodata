from dataclasses import dataclass

import numpy as np
from pygeodata.api import load
from pygeodata.data import Data
from pygeodata.drivers.rioxarray import RioXArrayDriver
from pygeodata.spec import SpatialSpec


@dataclass
class FeatureCorrelationLoader(Data):
    """
    Pearson correlation between two spatial variables at each pixel over time.

    Both ``feature`` and ``variable`` are Data instances — they are computed
    (or loaded from cache) at runtime and their hashes are part of this
    loader's cache key.
    """

    feature: Data = None
    variable: Data = None

    ext = 'tif'
    driver = RioXArrayDriver()

    def _process(self, spec: SpatialSpec) -> None:
        x = load(self.feature, spec)
        y = load(self.variable, spec)
        corr = np.corrcoef(x.values.ravel(), y.values.ravel())[0, 1]
        import xarray as xr

        out = xr.full_like(x, fill_value=float(corr))
        out.rio.to_raster(self.get_processed_path(spec))
