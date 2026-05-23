from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.config import get_config
from pygeodata.types import SpatialSpec


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
    2. On :meth:`process`, the hash is compared against a ``.hash.json`` file written
       alongside the output. If they match, processing is skipped.
    3. The source code registry (``.source/``) tracks the class AST hash on disk,
       enabling detection of stale cache entries after code changes.

    Class Attributes
    ----------------
    _sort_params : tuple[str]
        Parameter names whose list/tuple values should be sorted before hashing and
        path generation, ensuring order-independence.
    _exclude_params : tuple[str]
        Parameter names excluded from all hashing, path generation, and serialization.
        These parameters are present for internal use only, such as the number of threads
        used.
    _exclude_params_from_path : tuple[str]
        Parameter names excluded from path generation only (still included in hashes).
        This can be used for parameters that are used when overwriting get_processed_path,
        as to not cause duplication of information. It can also be used to prevent parameters
        that are only used in the meth:`load` method from being included in the path.

    Notes
    -----
    Subclasses that define ``__init__`` parameters should store them as instance
    attributes. :meth:`get_params` discovers parameters by inspecting ``vars(self)``.
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

    ext = 'png'

    def get_filename(self, ext: str | None = None) -> str:
        params = self.get_flat_params()
        stem = self.get_file_stem()
        if len(params) == 0:
            return f'{stem}.{ext}'
        return f'{stem}_{"_".join(f"{k}={v}" for k, v in params.items())}.{ext}'

    @classmethod
    def get_processed_base_dir(cls) -> Path:
        return get_config().path_figures

    @classmethod
    def get_processed_dir_pattern(cls) -> Path:
        return cls.get_processed_base_dir()

    def get_processed_dir(self, spec: SpatialSpec) -> Path:
        """
        Return the directory where this class' output is stored for the given spec.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification.

        Returns
        -------
        Path
            The output directory path.
        """
        return self.get_processed_base_dir()
