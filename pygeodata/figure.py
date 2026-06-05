import hashlib
import json
from pathlib import Path
from typing import ClassVar

from pygeodata.artifact import Artifact
from pygeodata.cache import handle_invalid, path_matches_hash
from pygeodata.config import JSONKeys, get_config
from pygeodata.extraction import flatten_parameter_dict_for_path
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

    color: ClassVar[str] = '#bda0bc'
    ext: ClassVar[str] = 'png'

    def get_filename(self, ext: str | None = None) -> str:
        resolved_ext = ext or self.get_ext()
        params = self.get_params(exclude=False)
        stem_part = self.get_file_stem()
        config = get_config()

        sep = '_' if config.filesystem_allows_punctuation else ' '
        es = '=' if config.filesystem_allows_punctuation else '-'

        if len(params) == 0:
            return f'{stem_part}.{resolved_ext}'

        flat_params = flatten_parameter_dict_for_path(params)

        if len(flat_params) <= config.max_file_param_depth:
            param_str = sep.join(f'{k}{es}{v}' for k, v in sorted(flat_params.items()))
        else:
            json_params = json.dumps(flat_params, sort_keys=True)
            param_str = hashlib.sha256(json_params.encode('utf-8')).hexdigest()
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
        root = cls.get_cache_root()
        full_pattern = (
            generate_path(
                base_dir=root,
            )
            / f'*{cls.get_file_stem()}.*'
        )
        return str(full_pattern.relative_to(root))

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

        return generate_path(
            spec=spec,
            base_dir=self.get_cache_root(),
        )
