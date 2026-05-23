import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from filelock import FileLock

from pygeodata.config import get_config
from pygeodata.extraction import extract_instances
from pygeodata.formatting import format_value_as_json, format_value_as_string
from pygeodata.hash import define_hash_from_class
from pygeodata.paths import generate_path
from pygeodata.tracked_object import TrackedObject
from pygeodata.types import Processor, SpatialSpec


def normalize_value_for_json(value: Any) -> Any:
    """
    Recursively normalize a value to a JSON-serializable form.

    :class:`Artifact` instances are represented as ``{"class_name": ..., "params": ...}``.
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
    if isinstance(value, Artifact):
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

    :class:`Artifact` instances are replaced by their state hash. Sets are sorted
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

    if isinstance(value, Artifact):
        return value.get_state_hash()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(normalize_value_for_hash(i) for i in value)
    if isinstance(value, dict):
        return {str(k): normalize_value_for_hash(v) for k, v in sorted(value.items())}
    return format_value_as_json(value)


def render_value_for_path(value: Any) -> str:
    """
    Render a parameter value as a compact string for use in filesystem paths.

    :class:`Artifact` instances render as their class name. Sequences and dicts
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

    if isinstance(value, Artifact):
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

    Nested :class:`Artifact` instances are expanded with ``__``-separated keys.
    Lists and dicts containing artifacts get additional entries for each.

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

    if isinstance(value, Artifact):
        flat[prefix] = value.get_class_name()
        for nk, nv in value.get_flat_params(exclude=False).items():
            flat[f'{prefix}__{nk}'] = nv
    elif isinstance(value, (list, tuple)):
        artifact_indices = {idx for idx, item in enumerate(value) if any(extract_instances(item, Artifact))}
        flat[prefix] = render_value_for_path(value)
        for idx in artifact_indices:
            flat.update(flatten_param_for_path(f'{prefix}_{idx}', value[idx]))
    elif isinstance(value, dict):
        keys = sorted(value.keys())
        artifact_keys = {k for k in keys if any(extract_instances(value[k], Artifact))}
        flat[prefix] = render_value_for_path(value)
        for k in artifact_keys:
            flat.update(flatten_param_for_path(f'{prefix}_{k}', value[k]))
    else:
        flat[prefix] = render_value_for_path(value)

    return flat


class Artifact(TrackedObject, ABC):
    _sort_params: tuple[str] = ()
    _exclude_params: tuple[str] = ()
    _exclude_params_from_path: tuple[str] = ()

    @property
    def ext(self) -> str:
        ext = getattr(self.processor, 'ext', None)
        if ext is None:
            raise ValueError('ext must be specified as an instance variable')
        return ext

    @classmethod
    def get_file_stem(cls) -> str:
        # Handle acronym → word transitions
        s1 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', cls.get_class_name())
        # Handle normal camelCase → camel_Case
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    def get_filename(self, ext: str | None = None) -> str:
        ext = ext or self.ext
        if ext is None:
            raise ValueError('ext must be specified as parameter or as instance variable')
        return f'{self.get_file_stem()}.{ext}'

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.get_class_name()}({", ".join(parts)})'

    def resolve_spec(self, spec: SpatialSpec) -> SpatialSpec:
        """
        Resolve a potentially underspecified :class:`~pygeodata.types.SpatialSpec`.

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

    def get_params(self, exclude: bool = True) -> dict[str, Any]:
        """
        Return the instance parameters of the artifact.

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

        Nested :class:`Artifact` values are expanded with ``__``-separated keys.

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
        Return the source file path for this artifact.

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

    @classmethod
    @abstractmethod
    def get_processed_base_dir(cls) -> Path:
        pass

    @classmethod
    @abstractmethod
    def get_processed_dir_pattern(cls) -> Path:
        return generate_path(name=cls.get_class_name(), base_dir=get_config().path_data_processed)

    @abstractmethod
    def get_processed_dir(self, spec: SpatialSpec) -> Path:
        """
        Return the directory where this artifact's output is stored for the given spec.

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
            Full path to the output file.

        Raises
        ------
        ValueError
            If no extension is available from either the argument or :attr:`ext`.
        """
        if ext is None:
            ext = self.ext
        return self.get_processed_dir(spec) / self.get_filename(ext=ext)

    def get_params_path(self, spec: SpatialSpec) -> Path:
        path = self.get_processed_path(spec)
        return path.parent / f'.{path.stem}.params.json'

    def write_parameters(self, spec: SpatialSpec) -> None:
        spec = self.resolve_spec(spec)
        path = self.get_params_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(self.get_params_as_json(exclude=False), f, indent=4)

    def get_state_hash(self) -> str:
        """Recursively hashes the class code and all injected parameter artifacts."""
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
        path = self.get_processed_path(spec)
        return path.parent / f'.{path.stem}.hash.json'

    def get_execution_graph_path(self, spec: SpatialSpec) -> Path:
        """
        Return the path to the ``.graph.json`` file for the given spec.

        Parameters
        ----------
        spec : SpatialSpec

        Returns
        -------
        Path
        """
        path = self.get_processed_path(spec)
        return path.parent / f'.{path.stem}.graph.json'

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
        Check whether the on-disk state hash matches the current state.

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

    def _process(self, spec: SpatialSpec) -> Iterable['Artifact'] | None:
        """
        Internal processing method. Override in subclasses for custom logic.

        The default implementation calls ``self.processor(output_path, spec)``.

        If this method yields :class:`Artifact` instances, those artifacts (rather than
        ``self``) will have their state hashes and parameters written after processing.
        This supports the **co-output pattern**, where a single processing step produces
        outputs for multiple artifacts simultaneously.

        Parameters
        ----------
        spec : SpatialSpec
            The resolved spatial specification.

        Returns
        -------
        Iterable[Artifact] or None
            Yield artifacts to write hashes for, or return ``None`` to write ``self``'s hash.

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
        Process the artifact for the given spec, with locking and cache validation.

        This is the main entry point for producing output. It:

        1. Acquires a file lock on the source registry to safely update the code registry.
        2. Acquires a file lock on the output directory to prevent concurrent writes.
        3. Skips processing if :meth:`is_processed` returns ``True``.
        4. Calls :meth:`_process` and writes state hashes for all produced artifacts.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification to process for.
        """
        from pygeodata.visualisations import plot_compact_execution_graph

        spec = self.resolve_spec(spec)

        processed_path = self.get_processed_path(spec)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        data_lock_path = processed_path.parent / 'process.lock'

        with FileLock(data_lock_path, timeout=3600):
            if self.is_processed(spec):
                return

            if self.is_processed_hash_present(spec) and not self.is_cache_valid(spec):
                print(f'# Cache invalid for {self.get_class_name()}. Recomputing ...')

            produced = self._process(spec)
            artifacts = () if produced is None else tuple(produced)
            for artifact in (*artifacts, self):
                artifact.init_source_registry()
                artifact.write_parameters(spec)
                artifact.write_state_hash(spec)

                if next(extract_instances(artifact.get_params(exclude=False), TrackedObject), None) is not None:
                    plot_compact_execution_graph(artifact=self, out_path=self.get_execution_graph_path(spec))
