import json
from pathlib import Path
from typing import ClassVar

from pygeodata.artifact import Artifact
from pygeodata.cache import handle_invalid, path_matches_hash
from pygeodata.config import JSONKeys, get_config
from pygeodata.extraction import flatten_parameter_dict_for_path
from pygeodata.hash import calculate_dict_hash
from pygeodata.paths import CachePathResolver, generate_path
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

    color: ClassVar[str] = '#bda0bc'
    ext: ClassVar[str] = 'png'

    def get_filename(self, ext: str | None = None, spec: SpatialSpec | None = None) -> str:
        resolved_ext = ext or self.get_ext()
        config = get_config()

        if not config.human_readable_paths and config.flatten_figures:
            if spec is None:
                raise ValueError(
                    f'{self.get_class_name()}.get_filename() requires spec '
                    'when human_readable_paths=False and flatten_figures=True',
                )
            return f'{self.get_state_hash(spec)}.{resolved_ext}'

        if not config.flatten_figures:
            # directory carries all identification — plain stem suffices
            return f'{self.get_file_stem()}.{resolved_ext}'

        # flatten_figures=True, human_readable_paths=True: encode params in filename
        params = self.get_params()
        stem_part = self.get_file_stem()
        sep = '_' if config.filesystem_allows_punctuation else ' '
        es = '=' if config.filesystem_allows_punctuation else '-'

        if not params:
            return f'{stem_part}.{resolved_ext}'

        flat_params = flatten_parameter_dict_for_path(params)
        if len(flat_params) <= config.max_file_param_depth:
            param_str = sep.join(f'{k}{es}{v}' for k, v in sorted(flat_params.items()))
        else:
            param_str = calculate_dict_hash(flat_params)
        return f'{stem_part}{sep}{param_str}.{resolved_ext}'

    @classmethod
    def get_cache_root(cls) -> Path:
        return get_config().path_figures

    @classmethod
    def get_general_cache_pattern(cls) -> str:
        parts = ('*',) * len(Path(cls.get_cls_cache_pattern()).parts)
        return str(Path(*parts))

    @classmethod
    def get_cls_cache_pattern(cls) -> str:
        config = get_config()
        if config.flatten_figures and not config.human_readable_paths:
            return f'*.{cls.ext}'
        if config.flatten_figures:
            return f'{cls.get_file_stem()}*.*'
        if not config.human_readable_paths:
            return str(Path(cls.get_class_name()) / '*')
        return str(Path('*') / '*' / cls.get_class_name() / '*')

    @classmethod
    def purge_cls_cache(cls, dry_run: bool = True) -> None:
        dependency_tree_hash = cls.get_dependency_tree_hash()
        root = cls.get_cache_root()
        pattern = cls.get_cls_cache_pattern()

        for path in root.glob(pattern):
            hash_path = CachePathResolver.from_path(path).state_hash_path
            if not path_matches_hash(hash_path, dependency_tree_hash):
                handle_invalid(path, dry_run=dry_run, hash_path=hash_path)

    @classmethod
    def matches_cache_path(cls, path: Path) -> bool:
        hash_path = CachePathResolver.from_path(path).state_hash_path
        if not hash_path.exists():
            return False
        with hash_path.open(encoding='utf-8') as f:
            return json.load(f).get(JSONKeys.CLASS_NAME, None) == cls.get_class_name()

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
        config = get_config()
        root = self.get_cache_root()

        if config.flatten_figures:
            return root

        return generate_path(
            spec=spec,
            name=self.get_class_name(),
            base_dir=root,
            **self.get_params(),
        )
