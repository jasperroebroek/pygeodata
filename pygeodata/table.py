from pathlib import Path
from typing import ClassVar

import pandas as pd

from pygeodata.config import get_config
from pygeodata.data import Data
from pygeodata.drivers.pandas import PandasDriver
from pygeodata.protocols import Driver


class Table(Data[pd.DataFrame]):
    """
    Base class for tabular (non-spatial) outputs in pygeodata.

    Identical to :class:`~pygeodata.data.Data` in every respect except that it produces a
    :class:`pandas.DataFrame`, stores its outputs under
    :attr:`~pygeodata.config.Config.path_tables`, and forms its own ``object_type`` family.

    A subclass that only implements :meth:`_process` needs neither a ``driver`` nor an ``ext``
    declaration: the default :class:`~pygeodata.drivers.pandas.PandasDriver` reads CSV, and
    :meth:`~pygeodata.data.Data.get_ext` derives ``'csv'`` from it. Both remain overridable —
    set ``ext`` and ``driver`` together to store another format.

    Examples
    --------
    A minimal table writing a CSV:

    .. code-block:: python

        import pandas as pd
        from pygeodata.table import Table


        class SummaryTable(Table):
            def _process(self, spec):
                df = pd.DataFrame({'a': [1, 2, 3]})
                df.to_csv(self.ensure_processed_path(spec), index=False)

    A table stored as parquet:

    .. code-block:: python

        from pygeodata.drivers.pandas import PandasDriver, PandasFormat


        class WideTable(Table):
            ext = 'parquet'
            driver = PandasDriver(format=PandasFormat.PARQUET)
    """

    color: ClassVar[str] = '#c9d8c5'
    driver: ClassVar[Driver] = PandasDriver()

    @classmethod
    def get_cache_root(cls) -> Path:
        return get_config().path_tables
