from pathlib import Path
from typing import ClassVar

from pygeodata.artifact import Artifact
from pygeodata.config import get_config


class Figure(Artifact):
    """
    Base class for all figures in pygeodata.

    A :class:`Figure` encapsulates the logic for processing geospatial Figure into
    a deterministic, cached output file. Subclasses define how Figure is produced by
    implementing :meth:`_process` and/or providing a :attr:`processor`.

    The caching system works as follows:

    1. Each object computes a **state hash** from its class source code (via AST) and
       its parameter values. This hash uniquely identifies the combination of logic and
       inputs.
    2. On :meth:`process`, the hash is compared against a hash file written
       alongside the output. If they match, processing is skipped.
    3. The source code registry (``.source/``) tracks the class AST hash on disk,
       enabling detection of stale cache entries after code changes.

    Class Attributes
    ----------------
    _sort_params : tuple[str]
        Inherited from :class:`~pygeodata.tracked_object.TrackedObject`. Parameter names
        whose list/tuple values should be sorted before hashing and path generation,
        ensuring order-independence.

    Notes
    -----
    Subclasses that define ``__init__`` parameters should store them as instance
    attributes. :meth:`~pygeodata.tracked_object.TrackedObject.get_params` discovers
    parameters by inspecting ``vars(self)``.
    Recommended to use dataclasses for this.

    Subclasses must not be defined interactively (e.g. in a REPL or Jupyter notebook),
    as the caching relies on :func:`inspect.getsource` for AST parsing.

    Examples
    --------
    A minimal loader using a processor:

    .. code-block:: python

        from pygeodata.data import Figure
        from pygeodata.processors.rasterizer import Rasterizer


        class MyVectorLoader(Figure):
            def __init__(self, year: int):
                self.year = year

            @property
            def processor(self):
                return Rasterizer(src_path=f'/Figure/vectors/{self.year}.gpkg')

    A loader that co-produces outputs by yielding loaders from ``_process``:

    .. code-block:: python

        class MultiOutputLoader(Figure):
            ext = 'tif'

            def _process(self, spec):
                # ... write files for both loaders ...
                yield LoaderA()
                yield LoaderB()
    """

    color: ClassVar[str] = '#bda0bc'
    ext: ClassVar[str] = 'png'

    @classmethod
    def get_cache_root(cls) -> Path:
        return get_config().path_figures
