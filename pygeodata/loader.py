import ast
import functools
import hashlib
import inspect
import json
import re
import textwrap
from collections import deque
from collections.abc import Generator, Iterable, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic

from filelock import FileLock

from pygeodata.config import get_config
from pygeodata.formatting import format_value_as_json, format_value_as_string
from pygeodata.hash import define_hash_from_class
from pygeodata.paths import generate_path
from pygeodata.types import Driver, Processor, SpatialSpec, T


def extract_dataloaders(value: Any) -> Generator['DataLoader']:
    """
    Recursively yield all :class:`DataLoader` instances found in a nested structure.

    Traverses lists, tuples, sets, and dicts recursively.

    Parameters
    ----------
    value : Any
        A value that may contain :class:`DataLoader` instances at any level of nesting.

    Yields
    ------
    DataLoader
        Each :class:`DataLoader` instance found in the structure.
    """
    if isinstance(value, DataLoader):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from extract_dataloaders(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from extract_dataloaders(item)


def normalize_value_for_json(value: Any) -> Any:
    """
    Recursively normalize a value to a JSON-serializable form.

    :class:`DataLoader` instances are represented as ``{"class_name": ..., "params": ...}``.
    Sequences are converted to tuples. Dicts are sorted by key.

    Parameters
    ----------
    value : Any
        The value to normalize.

    Returns
    -------
    Any
        A JSON-serializable representation of the value.
    """
    if isinstance(value, DataLoader):
        return {
            'class_name': value.get_class_name(),
            'params': normalize_value_for_json(value.get_params(exclude=False)),
        }
    if isinstance(value, dict):
        return {k: normalize_value_for_json(v) for k, v in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(normalize_value_for_json(i) for i in value)
    return format_value_as_json(value)


def normalize_value_for_hash(value: Any) -> Any:
    """
    Recursively normalize a value for stable hashing.

    :class:`DataLoader` instances are replaced by their state hash. Sets are sorted
    before conversion to ensure determinism.

    Parameters
    ----------
    value : Any
        The value to normalize.

    Returns
    -------
    Any
        A normalized, JSON-serializable form suitable for hashing.
    """
    if isinstance(value, set):
        value = tuple(sorted(value, key=repr))

    if isinstance(value, DataLoader):
        return value.get_state_hash()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(normalize_value_for_hash(i) for i in value)
    if isinstance(value, dict):
        return {str(k): normalize_value_for_hash(v) for k, v in sorted(value.items())}
    return format_value_as_json(value)


def render_value_for_path(value: Any) -> str:
    """
    Render a parameter value as a compact string for use in filesystem paths.

    :class:`DataLoader` instances render as their class name. Sequences and dicts
    are rendered with their contents inline. Sets are sorted before rendering.

    Parameters
    ----------
    value : Any
        The value to render.

    Returns
    -------
    str
        A compact string representation safe for use in directory names.
    """
    if isinstance(value, set):
        value = tuple(sorted(value, key=repr))

    if isinstance(value, DataLoader):
        return value.get_class_name()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        inner = ', '.join(render_value_for_path(v) for v in value)
        return f'({inner})'
    if isinstance(value, dict):
        inner = ', '.join(f'{k}:{render_value_for_path(v)}' for k, v in sorted(value.items()))
        return f'{{{inner}}}'
    return format_value_as_string(value)


def flatten_param_for_path(prefix: str, value: Any) -> dict[str, Any]:
    """
    Flatten a potentially nested parameter value into a dict of path-safe key-value pairs.

    Nested :class:`DataLoader` instances are expanded with ``__``-separated keys.
    Lists and dicts containing loaders get additional entries for each loader element.

    Parameters
    ----------
    prefix : str
        The key prefix for the flattened entry.
    value : Any
        The value to flatten.

    Returns
    -------
    dict
        A flat dict of path-safe string values, keyed by their dotted parameter names.
    """
    flat: dict[str, Any] = {}

    if isinstance(value, set):
        value = tuple(sorted(value, key=repr))

    if isinstance(value, DataLoader):
        flat[prefix] = value.get_class_name()
        for nk, nv in value.get_flat_params(exclude=False).items():
            flat[f'{prefix}__{nk}'] = nv
    elif isinstance(value, (list, tuple)):
        loader_indices = {idx for idx, item in enumerate(value) if any(extract_dataloaders(item))}
        flat[prefix] = render_value_for_path(value)
        for idx in loader_indices:
            flat.update(flatten_param_for_path(f'{prefix}_{idx}', value[idx]))
    elif isinstance(value, dict):
        keys = sorted(value.keys())
        loader_keys = {k for k in keys if any(extract_dataloaders(value[k]))}
        flat[prefix] = render_value_for_path(value)
        for k in loader_keys:
            flat.update(flatten_param_for_path(f'{prefix}_{k}', value[k]))
    else:
        flat[prefix] = render_value_for_path(value)

    return flat


class DataLoader(Generic[T]):
    """
    Abstract base class for all data loaders in pygeodata.

    A :class:`DataLoader` encapsulates the logic for processing geospatial data into
    a deterministic, cached output file. Subclasses define how data is produced by
    implementing :meth:`_process` and/or providing a :attr:`processor`.

    The caching system works as follows:

    1. Each loader computes a **state hash** from its class source code (via AST) and
       its parameter values. This hash uniquely identifies the combination of logic and
       inputs.
    2. On :meth:`process`, the hash is compared against a ``.hash.json`` file written
       alongside the output. If they match, processing is skipped.
    3. The source code registry (``_source/``) tracks the class AST hash on disk,
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
        as to not cause duplication of information.

    Notes
    -----
    Subclasses that define ``__init__`` parameters should store them as instance
    attributes. :meth:`get_params` discovers parameters by inspecting ``vars(self)``.
    Recommended to use dataclasses for this.

    Subclasses must not be defined interactively (e.g. in a REPL or Jupyter notebook),
    as :meth:`get_source_metadata` relies on :func:`inspect.getsource` for AST parsing.

    Examples
    --------
    A minimal loader using a processor:

    .. code-block:: python

        from pygeodata.loader import DataLoader
        from pygeodata.processors.rasterizer import Rasterizer


        class MyVectorLoader(DataLoader):
            def __init__(self, year: int):
                self.year = year

            @property
            def processor(self):
                return Rasterizer(src_path=f'/data/vectors/{self.year}.gpkg')

    A loader that co-produces outputs by yielding loaders from ``_process``:

    .. code-block:: python

        class MultiOutputLoader(DataLoader):
            ext = 'tif'

            def _process(self, spec):
                # ... write files for both loaders ...
                yield LoaderA()
                yield LoaderB()
    """

    _sort_params: tuple[str] = ()
    _exclude_params: tuple[str] = ()
    _exclude_params_from_path: tuple[str] = ()

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.get_class_name()}({", ".join(parts)})'

    def resolve_spec(self, spec: SpatialSpec) -> SpatialSpec:
        """
        Resolve an underspecified :class:`~pygeodata.types.SpatialSpec`.

        If the spec is already fully defined, it is returned unchanged. Otherwise,
        delegates to ``processor.resolve_spec`` if available.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification to resolve.

        Returns
        -------
        SpatialSpec
            The resolved specification.
        """
        if spec.is_fully_defined:
            return spec
        resolver = getattr(self.processor, 'resolve_spec', None)
        return resolver(spec) if resolver is not None else spec

    @property
    def processor(self) -> Processor | None:
        """
        The processor callable used to produce the output file.

        Override in subclasses to return a processor instance (e.g. :class:`~pygeodata.processors.rasterizer.Rasterizer`).
        The processor is called as ``processor(output_path, spec)`` during :meth:`_process`.

        Returns
        -------
        Processor or None
            The processor, or ``None`` if not implemented.
        """
        return None

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
    def get_class_name(cls) -> str:
        return cls.__name__

    @classmethod
    def get_name(cls) -> str:
        """
        Return the snake_case name of this loader class.

        Converts CamelCase and acronyms to snake_case. Used as the output filename stem.

        Returns
        -------
        str
            Snake_case name, e.g. ``"my_vector_loader"``.

        Examples
        --------
        >>> USGSElevationLoader.get_name()
        'usgs_elevation_loader'
        """
        # Handle acronym → word transitions (e.g. XMLHTTPRequest → XML_Http_Request)
        s1 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', cls.get_class_name())
        # Handle normal camelCase → camel_Case
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    @property
    def ext(self) -> str:
        ext = getattr(self.processor, 'ext', None)
        if ext is None:
            ext = self.driver.default_ext
        return ext

    def get_params(self, exclude: bool = True) -> dict[str, Any]:
        """
        Return the instance parameters of this loader.

        Inspects ``vars(self)`` and filters out private attributes, reserved names,
        and any names listed in :attr:`_exclude_params` or :attr:`_exclude_params_from_path`.

        Parameters
        ----------
        exclude : bool, default True
            If ``True``, also excludes parameters listed in :attr:`_exclude_params_from_path`.

        Returns
        -------
        dict[str, Any]
            A dict of parameter name-value pairs.
        """
        params = {}
        for key, value in vars(self).items():
            if key in ('name', 'class_name', 'processor', 'driver', 'process', 'load'):
                continue
            if key.startswith('_'):
                continue
            if key in self._exclude_params:
                continue
            if key in self._exclude_params_from_path and exclude:
                continue

            if key in self._sort_params and isinstance(value, (list, tuple)):
                parsed_value = type(value)(sorted(value, key=repr))
            else:
                parsed_value = value

            params.update({key: parsed_value})
        return params

    def get_params_as_json(self, exclude: bool = True) -> dict[str, Any]:
        return normalize_value_for_json(self.get_params(exclude=exclude))

    def get_flat_params(self, exclude: bool = True) -> dict[str, Any]:
        """
        Return a flattened dict of parameters for deterministic path generation.

        Nested :class:`DataLoader` values are expanded with ``__``-separated keys.

        Parameters
        ----------
        exclude : bool, default True
            Passed through to :meth:`get_params`.

        Returns
        -------
        dict[str, Any]
            A flat dict suitable for use in :func:`~pygeodata.paths.generate_path`.
        """
        flat_params = {}
        for k, v in self.get_params(exclude=exclude).items():
            flat_params.update(flatten_param_for_path(k, v))
        return flat_params

    def get_src_path(self) -> Path:
        """
        Return the source file path for this loader.

        Raises if not implemented.

        Returns
        -------
        Path
            The source file path.
        """
        processor = self.processor
        if processor is None:
            raise NotImplementedError('Processor must be implemented to get src_path')

        if not hasattr(processor, 'src_path'):
            raise NotImplementedError(f'Processor {processor} lacks src_path')

        return Path(getattr(processor, 'src_path'))

    def get_processed_dir(self, spec: SpatialSpec) -> Path:
        """
        Return the directory where this loader's output is stored for the given spec.

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
            base_dir=get_config().path_data_processed,
            **flat_params,
        )

    def get_processed_path(self, spec: SpatialSpec, ext: str | None = None) -> Path:
        """
        Return the full path to the processed output file.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification.
        ext : str, optional
            File extension override. Falls back to :attr:`ext`.

        Returns
        -------
        Path
            Full path to the output file, e.g. ``/data_processed/.../my_loader.tif``.

        Raises
        ------
        ValueError
            If no extension is available from either the argument or :attr:`ext`.
        """
        ext = ext or self.ext
        if ext is None:
            raise ValueError('ext must be specified as parameter or as instance variable')
        return self.get_processed_dir(spec) / f'{self.get_name()}.{ext}'

    def write_parameters(self, spec: SpatialSpec) -> None:
        spec = self.resolve_spec(spec)
        path = self.get_processed_path(spec).with_suffix('.params.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(self.get_params_as_json(exclude=False), f, indent=4)

    @classmethod
    def find_loader_class(cls, name: str) -> type | None:
        queue = deque([DataLoader])
        while queue:
            curr = queue.popleft()
            if curr.get_class_name() == name:
                return curr
            queue.extend(curr.__subclasses__())
        return None

    @classmethod
    def get_source_registry_dir(cls) -> Path:
        """Centralized storage: data_processed/_source/module/path/ClassName/."""
        module_path = cls.__module__.replace('.', '/')
        return get_config().path_data_processed / '_source' / module_path / cls.get_class_name()

    @classmethod
    def get_source_registry_path(cls) -> Path:
        return cls.get_source_registry_dir() / 'source.json'

    @classmethod
    def get_source_registry_code_path(cls) -> Path:
        return cls.get_source_registry_dir() / 'source.py'

    @classmethod
    @functools.cache
    def get_source_metadata(cls) -> MappingProxyType[str, Any]:
        """
        Parse the class source via AST and extract hashing and dependency metadata.

        Results are cached in memory (once per class per process).

        Returns
        -------
        MappingProxyType
            An immutable mapping with the following keys:

            - ``source_code`` *(str)*: The dedented source code of the class.
            - ``source_hash`` *(str)*: SHA-256 of the AST dump, formatting-agnostic.
            - ``call_dependencies`` *(list[type])*: :class:`DataLoader` subclasses called
              within the class body.
            - ``inheritance_dependencies`` *(list[type])*: :class:`DataLoader` subclasses
              in the MRO.
            - ``all_dependencies`` *(list[type])*: Union of call and inheritance dependencies.

        Raises
        ------
        OSError
            If source code cannot be retrieved (e.g. in a REPL or notebook environment).
        """
        try:
            source = inspect.getsource(cls)
            clean_source = textwrap.dedent(source)
            tree = ast.parse(clean_source)
        except (TypeError, OSError) as err:
            raise OSError(
                'AST Parsing failed. Caching is disabled. You are likely in a REPL/Notebook environment. Use standard .py files.',
            ) from err

        source_hash = hashlib.sha256(ast.dump(tree).encode()).hexdigest()

        called_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)

        call_dependencies = []
        for name in called_names:
            if name != cls.get_class_name():
                dep_cls = cls.find_loader_class(name)
                if dep_cls and issubclass(dep_cls, DataLoader):
                    call_dependencies.append(dep_cls)

        inheritance_dependencies = [
            base
            for base in cls.__mro__
            if issubclass(base, DataLoader) and base is not cls and base.get_class_name() != 'DataLoader'
        ]

        all_dependencies = set(call_dependencies) | set(inheritance_dependencies)

        return MappingProxyType(
            {
                'source_code': clean_source,
                'source_hash': source_hash,
                'call_dependencies': sorted(call_dependencies, key=lambda c: c.get_class_name()),
                'inheritance_dependencies': sorted(inheritance_dependencies, key=lambda c: c.get_class_name()),
                'all_dependencies': sorted(all_dependencies, key=lambda c: c.get_class_name()),
            },
        )

    @classmethod
    def get_source_hash(cls) -> str:
        """
        Compute a hash uniquely identifying this loader's code and parameter state.

        Combines the source hierarchy hash (from AST) with the normalized parameter
        values. If a processor is set, its class hash is also included.

        Returns
        -------
        str
            A SHA-256 hex digest representing the full loader state.
        """
        return cls.get_source_metadata()['source_hash']

    @classmethod
    def get_source_code(cls) -> str:
        return cls.get_source_metadata()['source_code']

    @classmethod
    def _get_dependency_tree_recursive(cls, visited: frozenset[type] | None = None) -> dict[str, Any] | str:
        if visited is None:
            visited = frozenset()

        if cls in visited:
            return 'circular'

        next_visited = visited | frozenset([cls])
        metadata = cls.get_source_metadata()

        tree = {
            'class_name': cls.get_class_name(),
            'source_hash': cls.get_source_hash(),
            'call_dependencies': {},
            'inheritance_dependencies': {},
        }

        for dep_cls in metadata['call_dependencies']:
            tree['call_dependencies'][dep_cls.get_class_name()] = dep_cls._get_dependency_tree_recursive(next_visited)

        for dep_cls in metadata['inheritance_dependencies']:
            tree['inheritance_dependencies'][dep_cls.get_class_name()] = dep_cls._get_dependency_tree_recursive(
                next_visited,
            )

        return tree

    @classmethod
    @functools.cache
    def get_dependency_tree(cls) -> dict[str, Any]:
        """
        Build a nested dict representing the full dependency tree of this class.

        Cached via :func:`functools.cache`. Handles circular dependencies by substituting
        ``"circular"`` for repeated nodes.

        Returns
        -------
        dict
            Nested structure with keys ``class_name``, ``source_hash``,
            ``call_dependencies``, and ``inheritance_dependencies``.
        """
        d: dict[str, Any] = cls._get_dependency_tree_recursive(visited=None)
        return d

    @classmethod
    def get_dependency_graph(cls) -> dict[str, Any]:
        """
        Build a flat graph of all :class:`DataLoader` dependencies from this class.

        Returns
        -------
        dict with keys:

        - ``nodes`` *(dict[type, type])*: All reachable loader classes.
        - ``call_edges`` *(set[tuple[type, type]])*: Edges from call dependencies.
        - ``inheritance_edges`` *(set[tuple[type, type]])*: Edges from inheritance.
        """
        nodes: dict[type, type] = {}
        call_edges: set[tuple[type, type]] = set()
        inheritance_edges: set[tuple[type, type]] = set()

        def visit(current_cls: type['DataLoader']) -> None:
            if current_cls in nodes:
                return

            nodes[current_cls] = current_cls
            metadata = current_cls.get_source_metadata()

            for dep_cls in metadata.get('call_dependencies', []):
                call_edges.add((current_cls, dep_cls))
                visit(dep_cls)

            for dep_cls in metadata.get('inheritance_dependencies', []):
                inheritance_edges.add((current_cls, dep_cls))
                visit(dep_cls)

        visit(cls)

        return {
            'nodes': nodes,
            'call_edges': call_edges,
            'inheritance_edges': inheritance_edges,
        }

    @classmethod
    @functools.cache
    def get_source_hierarchy_hash(cls) -> str:
        """
        Compute a hash over the full dependency tree of this class.

        Hashes the JSON-serialized :meth:`get_dependency_tree` result, so any change
        in this class or any of its transitive dependencies will produce a different hash.

        Returns
        -------
        str
            A SHA-256 hex digest of the full dependency tree.
        """
        tree = cls.get_dependency_tree()
        return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()

    @classmethod
    def save_source_code_and_hash(cls) -> None:
        """Saves the current AST hash and writes the source code for inspection."""
        from pygeodata.visualisations import plot_class_dependency_graph

        tree = cls.get_dependency_tree()

        json_file = cls.get_source_registry_path()
        json_file.parent.mkdir(parents=True, exist_ok=True)

        with Path.open(json_file, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'source_hash': cls.get_source_hash(),
                    'source_hierarchy_hash': cls.get_source_hierarchy_hash(),
                    'dependencies': tree,
                },
                f,
                indent=4,
            )

        py_file = cls.get_source_registry_code_path()
        with Path.open(py_file, 'w', encoding='utf-8') as f:
            f.write(cls.get_source_code())

        if len(tree['inheritance_dependencies']) + len(tree['call_dependencies']) > 0:
            plot_class_dependency_graph(loader=cls, path=cls.get_source_registry_dir() / 'dependency_graph', view=False)

    @classmethod
    def is_source_registry_valid(cls) -> bool:
        """
        Check whether the on-disk source registry matches the current class.

        Compares the ``source_hierarchy_hash`` stored in ``_source/.../source.json``
        against :meth:`get_source_hierarchy_hash`.

        Returns
        -------
        bool
        """
        registry_file = cls.get_source_registry_path()
        if not registry_file.exists():
            return False

        with Path.open(registry_file, encoding='utf-8') as f:
            saved_state = json.load(f)

        return saved_state.get('source_hierarchy_hash') == cls.get_source_hierarchy_hash()

    @classmethod
    def initialize_source_registry(cls) -> None:
        """
        Ensure this class and all its dependencies have a valid source registry on disk.

        Calls :meth:`save_source_code_and_hash` if the registry is stale, then
        recursively initializes registries for all dependencies.
        """
        if not cls.is_source_registry_valid():
            cls.save_source_code_and_hash()

        for dep_class in cls.get_source_metadata()['all_dependencies']:
            dep_class.initialize_source_registry()

    def get_state_hash(self) -> str:
        """Recursively hashes the class code AND all injected parameter DataLoaders."""
        state = {
            'blueprint_hash': self.get_source_hierarchy_hash(),
            'params': {k: normalize_value_for_hash(v) for k, v in self.get_params().items()},
        }

        if self.processor:
            state['processor_hash'] = define_hash_from_class(self.processor.__class__)

        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def get_state_hash_path(self, spec: SpatialSpec) -> Path:
        """
        Return the path to the ``.hash.json`` file for the given spec.

        Parameters
        ----------
        spec : SpatialSpec

        Returns
        -------
        Path
        """
        return self.get_processed_path(spec).with_suffix('.hash.json')

    def write_state_hash(self, spec: SpatialSpec) -> None:
        """
        Write the state hash and source hierarchy hash to a ``.hash.json`` file.

        Creates the parent directory if it does not exist.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification for which to write the hash.
        """
        hash_path = self.get_state_hash_path(spec)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        with Path.open(hash_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'source_hierarchy_hash': self.get_source_hierarchy_hash(), 'state_hash': self.get_state_hash()},
                f,
                indent=4,
            )

    def read_state_hash(self, spec: SpatialSpec) -> str | None:
        """
        Read the previously written state hash from disk.

        Parameters
        ----------
        spec : SpatialSpec

        Returns
        -------
        str or None
            The state hash string, or ``None`` if the hash file does not exist.
        """
        hash_path = self.get_state_hash_path(spec)
        if not hash_path.exists():
            return None
        with Path.open(hash_path, encoding='utf-8') as f:
            return json.load(f).get('state_hash')

    def processed_path_exists(self, spec: SpatialSpec) -> bool:
        return self.get_processed_path(spec).exists()

    def is_processed_hash_present(self, spec: SpatialSpec) -> bool:
        return self.get_state_hash_path(spec).exists()

    def is_cache_valid(self, spec: SpatialSpec) -> bool:
        """
        Check whether the on-disk state hash matches the current loader state.

        Parameters
        ----------
        spec : SpatialSpec

        Returns
        -------
        bool
            ``True`` if the hash file exists and matches :meth:`get_state_hash`.
        """
        hash_file = self.get_state_hash_path(spec)
        if not hash_file.exists():
            return False

        saved_state_hash = self.read_state_hash(spec)
        if saved_state_hash is None:
            return False

        return saved_state_hash == self.get_state_hash()

    def is_processed(self, spec: SpatialSpec) -> bool:
        """
        Check whether the output file exists and the cache is valid.

        Parameters
        ----------
        spec : SpatialSpec

        Returns
        -------
        bool
            ``True`` if the output file exists and the state hash is current.
        """
        if not self.is_cache_valid(spec):
            return False
        return self.processed_path_exists(spec)

    def _process(self, spec: SpatialSpec) -> Iterable['DataLoader'] | None:
        """
        Internal processing method. Override in subclasses for custom logic.

        The default implementation calls ``self.processor(output_path, spec)``.

        If this method yields :class:`DataLoader` instances, those loaders (rather than
        ``self``) will have their state hashes and parameters written after processing.
        This supports the **co-output pattern**, where a single processing step produces
        outputs for multiple loaders simultaneously.

        Parameters
        ----------
        spec : SpatialSpec
            The resolved spatial specification.

        Returns
        -------
        Iterable[DataLoader] or None
            Yield loaders to write hashes for, or return ``None`` to write ``self``'s hash.

        Raises
        ------
        NotImplementedError
            If no processor is set and the method is not overridden.
        """
        spec = self.resolve_spec(spec)
        processor = self.processor
        if processor is None:
            raise NotImplementedError('Either load, processor or process must be implemented')
        processor(self.get_processed_path(spec), spec)

    def process(self, spec: SpatialSpec) -> None:
        """
        Process the loader for the given spec, with locking and cache validation.

        This is the main entry point for producing output. It:

        1. Acquires a file lock on the source registry to safely update the code registry.
        2. Acquires a file lock on the output directory to prevent concurrent writes.
        3. Skips processing if :meth:`is_processed` returns ``True``.
        4. Calls :meth:`_process` and writes state hashes for all produced loaders.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification to process for.
        """
        from pygeodata.visualisations import plot_compact_execution_graph

        spec = self.resolve_spec(spec)

        code_registry_dir = self.get_source_registry_dir()
        code_lock_path = code_registry_dir / 'source.lock'
        with FileLock(code_lock_path, timeout=60):
            self.initialize_source_registry()

        processed_path = self.get_processed_path(spec)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        data_lock_path = processed_path.parent / 'process.lock'

        with FileLock(data_lock_path, timeout=3600):
            if self.is_processed(spec):
                return

            if self.is_processed_hash_present(spec) and not self.is_cache_valid(spec):
                print(f'# Cache invalid for {self.get_class_name()}. Recomputing ...')

            produced = self._process(spec)
            loaders = (self,) if produced is None else tuple(produced)
            for loader in loaders:
                loader.write_parameters(spec)
                loader.write_state_hash(spec)

                if next(extract_dataloaders(loader.get_params(exclude=False)), None) is not None:
                    plot_compact_execution_graph(loader=self, out_path=processed_path.with_suffix('.graph'))

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
