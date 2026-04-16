from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
from pyarrow.dataset import Partitioning


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


@dataclass
class GeoPandasParquetDriver:
    """Load a vector file using GeoPandas.

    Parameters
    ----------
    partitioning : Partitioning or str or list of str, optional
        Partitioning scheme, see pyarrow.parquet.read_table
    open_kw : dict, optional
        Additional keyword arguments to pass to gpd.read_file
    """

    partitioning: Partitioning | str | list[str] | None = None
    open_kw: dict = field(default_factory=dict)

    def __call__(self, path: str | Path) -> gpd.GeoDataFrame:
        return gpd.read_parquet(path, partitioning=self.partitioning, **self.open_kw)

    default_ext = 'parquet'
