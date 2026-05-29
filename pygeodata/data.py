from pathlib import Path
from typing import ClassVar, Generic

from pygeodata.artifact import Artifact
from pygeodata.cache import handle_invalid, is_zarr_root, path_matches_hash
from pygeodata.config import get_config
from pygeodata.paths import CachePathResolver, generate_path
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
    2. On :meth:`process`, the hash is compared against a hash file written
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

    @classmethod
    def get_general_cache_pattern(cls) -> str:
        parts = ('*',) * len(Path(cls.get_cls_cache_pattern()).parts)
        return str(Path(*parts))

    @classmethod
    def purge_cls_cache(cls, dry_run: bool = True) -> None:
        dependency_tree_hash = cls.get_dependency_tree_hash()
        root = cls.get_cache_root()
        pattern = cls.get_cls_cache_pattern()

        for path in root.glob(pattern):
            for dirpath, dirs, files in path.walk(top_down=True, follow_symlinks=True):
                if dirpath != path and is_zarr_root(dirpath):
                    dirs.clear()
                    hash_path = CachePathResolver.from_path(dirpath).state_hash_path
                    if not path_matches_hash(hash_path, dependency_tree_hash):
                        handle_invalid(dirpath, dry_run=dry_run, hash_path=hash_path)
                    continue

                for file in files:
                    file_path = dirpath / file
                    hash_path = CachePathResolver.from_path(file_path).state_hash_path
                    if not path_matches_hash(hash_path, dependency_tree_hash):
                        handle_invalid(file_path, dry_run=dry_run, hash_path=hash_path)

                if dry_run:
                    continue

                for dir in dirs:
                    path_dir = dirpath / dir
                    if next(path_dir.iterdir(), None) is None:
                        print(f'Removing {path_dir}')
                        path_dir.rmdir()

                if next(dirpath.iterdir(), None) is None:
                    print(f'Removing {dirpath}')
                    dirpath.rmdir()

    @classmethod
    def get_cls_cache_pattern(cls) -> str:
        root = cls.get_cache_root()
        full_pattern = generate_path(
            name=cls.get_class_name(),
            base_dir=root,
        )
        return str(full_pattern.relative_to(root))

    @classmethod
    def matches_cache_path(cls, path: Path) -> bool:
        return path.name == cls.get_class_name()

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

        return generate_path(
            spec=spec,
            name=self.get_class_name(),
            base_dir=self.get_cache_root(),
            **self.get_params(exclude=True),
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
        self.process(spec)
        return self._load(spec)

    def _load(self, spec: SpatialSpec) -> T:
        return self.driver(self.get_processed_path(spec))
