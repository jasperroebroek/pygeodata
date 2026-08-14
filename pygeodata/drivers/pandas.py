from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pandas as pd


class PandasFormat(StrEnum):
    CSV = 'csv'
    PARQUET = 'parquet'
    EXCEL = 'excel'


_READERS = {
    PandasFormat.CSV: pd.read_csv,
    PandasFormat.PARQUET: pd.read_parquet,
    PandasFormat.EXCEL: pd.read_excel,
}

_DEFAULT_EXT = {
    PandasFormat.CSV: 'csv',
    PandasFormat.PARQUET: 'parquet',
    PandasFormat.EXCEL: 'xlsx',
}


@dataclass
class PandasDriver:
    """Load a plain (non-geospatial) table using pandas.

    Parameters
    ----------
    format : PandasFormat, default PandasFormat.CSV
        Which pandas reader to use.
    open_kw : dict, optional
        Additional keyword arguments to pass to the underlying pd.read_* call.
    """

    format: PandasFormat = PandasFormat.CSV
    open_kw: dict = field(default_factory=dict)

    def __call__(self, path: str | Path) -> pd.DataFrame:
        return _READERS[self.format](path, **self.open_kw)

    @property
    def default_ext(self) -> str:
        return _DEFAULT_EXT[self.format]
