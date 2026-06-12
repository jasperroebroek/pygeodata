from pathlib import Path
from typing import ClassVar, Generic

from pygeodata.artifact import Artifact
from pygeodata.config import get_config
from pygeodata.paths import CACHE_META_SUFFIXES, CachePathResolver
from pygeodata.protocols import Driver, T
from pygeodata.registry import EntryRegistry
from pygeodata.spec import SpatialSpec


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
    2. On :meth:`process`, the hash is compared against a hash file written
       alongside the output. If they match, processing is skipped.
    3. The source code registry (``.source/``) tracks the class AST hash on disk,
       enabling detection of stale cache entries after code changes.

    Class Attributes
    ----------------
    _sort_params : tuple[str]
        Parameter names whose list/tuple values should be sorted before hashing and
        path generation, ensuring order-independence.

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

    color: ClassVar[str] = '#d0dceb'

    def get_ext(self) -> str:
        if self.ext is not None:
            return self.ext
        processor_ext = getattr(self.processor, 'ext', None)
        if processor_ext is not None:
            return processor_ext
        driver = self.driver
        driver_ext = getattr(driver, 'default_ext', None)
        if driver_ext is not None:
            return driver_ext
        raise ValueError(f'{self.get_class_name()} must define ext or processor.ext or driver.default_ext')

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

    @classmethod
    def get_cache_root(cls) -> Path:
        return get_config().path_cache

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
        self.process(spec)
        return self._load(self.get_processed_path(spec))

    @classmethod
    def load_by_hash(cls, state_hash: str) -> T:
        """Load a cached output by (truncated) state hash.

        Looks up the full hash in :class:`~pygeodata.registry.EntryRegistry`,
        resolves the processed file path, and returns the data without
        re-running ``process``.

        Raises
        ------
        KeyError
            If *state_hash* matches zero or more than one entry.
        FileNotFoundError
            If the resolved output file does not exist on disk.
        """
        registry = EntryRegistry.instance()
        matches = [h for h in registry.records if h.startswith(state_hash)]
        if len(matches) == 0:
            raise KeyError(f'No entry found for hash prefix {state_hash!r}')
        if len(matches) > 1:
            raise KeyError(
                f'Ambiguous hash prefix {state_hash!r} matches {len(matches)} entries: ' + ', '.join(matches),
            )
        record = registry.records[matches[0]]
        if record.hash_path is None:
            raise FileNotFoundError(f'Entry {matches[0]!r} has no hash_path')
        resolver = CachePathResolver.from_path(Path(record.hash_path))
        candidates = [
            p
            for p in resolver.directory.iterdir()
            if p.stem == resolver.stem and ''.join(p.suffixes) not in CACHE_META_SUFFIXES
        ]
        if not candidates:
            raise FileNotFoundError(f'No output file found for hash {matches[0]!r} in {resolver.directory}')
        path = candidates[0]
        instance = cls.__new__(cls)
        try:
            return instance._load(path)
        except AttributeError as e:
            raise TypeError(
                f'{cls.__name__}._load requires instance attributes (params). '
                f'Override _load as a classmethod or ensure driver is param-independent.',
            ) from e

    def _load(self, path: Path) -> T:
        return self.driver(path)
