from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd


@dataclass
class GeoPandasDriver:
    """Load a vector file using GeoPandas.

    Parameters
    ----------
    open_kw : dict, optional
        Additional keyword arguments to pass to gpd.read_file
    """

    open_kw: dict = field(default_factory=dict)

    def __call__(self, path: str | Path) -> gpd.GeoDataFrame:
        return gpd.read_file(path, **self.open_kw, ignore_geometry=False)

    default_ext = 'geojson'
