from pathlib import Path
from typing import Generic

from pygeodata.artifact import Artifact
from pygeodata.config import get_config
from pygeodata.paths import generate_path
from pygeodata.types import Driver, SpatialSpec, T


class Data(Artifact, Generic[T]):
    """
    Base class for all data loaders in pygeodata.

    A :class:`Data` encapsulates the logic for processing geospatial data into
    a deterministic, cached output file. Subclasses define how data is produced by
    implementing :meth:`_process` and/or providing a :attr:`processor`.

    The caching system works as follows:

    1. Each loader computes a **state hash** from its class source code (via AST) and
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

        from pygeodata.data import Data
        from pygeodata.processors.rasterizer import Rasterizer


        class MyVectorLoader(Data):
            def __init__(self, year: int):
                self.year = year

            @property
            def processor(self):
                return Rasterizer(src_path=f'/data/vectors/{self.year}.gpkg')

    A loader that co-produces outputs by yielding loaders from ``_process``:

    .. code-block:: python

        class MultiOutputLoader(Data):
            ext = 'tif'

            def _process(self, spec):
                # ... write files for both loaders ...
                yield LoaderA()
                yield LoaderB()
    """

    @property
    def driver(self) -> Driver:
        """
        The driver used to read the output file produced by this loader.

        Falls back to ``processor.default_driver`` if not overridden. Raises if neither
        is available.

        Returns
        -------
        Driver
            A callable that reads the processed file and returns data.

        Raises
        ------
        NotImplementedError
            If neither ``processor`` nor a custom ``driver`` is implemented.
        AttributeError
            If the processor exists but lacks a ``default_driver`` attribute.
        """
        processor = self.processor
        if processor is None:
            raise NotImplementedError(f'{self}: Either processor or driver must be implemented')

        driver = getattr(processor, 'default_driver')
        if driver is None:
            raise AttributeError(f'Processor {processor} lacks default_driver and no driver is set')

        return driver

    @property
    def ext(self) -> str:
        ext = getattr(self.processor, 'ext', None)
        if ext is None:
            ext = self.driver.default_ext
        return ext

    @classmethod
    def get_processed_base_dir(cls) -> Path:
        return get_config().path_data_processed

    @classmethod
    def get_processed_dir_pattern(cls) -> Path:
        return generate_path(name=cls.get_class_name(), base_dir=cls.get_processed_base_dir())

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
        spec = self.resolve_spec(spec)

        flat_params = self.get_flat_params()

        return generate_path(
            spec=spec,
            name=self.get_class_name(),
            base_dir=self.get_processed_base_dir(),
            **flat_params,
        )

    def load(self, spec: SpatialSpec) -> T:
        """
        Load and return the processed data for the given spec.

        Parameters
        ----------
        spec : SpatialSpec

        Returns
        -------
        T
            The data returned by the loader's :attr:`driver`.
        """
        spec = self.resolve_spec(spec)
        return self.driver(self.get_processed_path(spec))
