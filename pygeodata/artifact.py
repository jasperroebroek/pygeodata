import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from filelock import FileLock

from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config
from pygeodata.extraction import extract_instances
from pygeodata.graph_types import RuntimeDependencyGraph, RuntimeNode, RuntimeParamEdge
from pygeodata.graphs import plot_compact_execution_graph
from pygeodata.hash import calculate_cls_source_hash, calculate_dict_hash
from pygeodata.paths import CachePathConstructor
from pygeodata.protocols import Processor
from pygeodata.registry_types import EntryRecord
from pygeodata.spec import SpatialSpec, SpecKeys
from pygeodata.tracked_object import TrackedObject


class Artifact(TrackedObject, ABC):
    _sort_params: ClassVar[tuple[str]] = ()
    ext: ClassVar[str | None] = None
    processor: ClassVar[Processor | None] = None
    color: ClassVar[str] = '#f8f9fa'

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.get_class_name()}({", ".join(parts)})'

    def get_ext(self) -> str:
        if self.ext is not None:
            return self.ext
        processor_ext = getattr(self.processor, 'ext', None)
        if processor_ext is not None:
            return processor_ext
        raise ValueError(f'{self.get_class_name()} must define ext or processor.ext')

    @classmethod
    def get_file_stem(cls) -> str:
        # Handle acronym → word transitions
        s1 = re.sub('([A-Z]+)([A-Z][a-z])', r'\1_\2', cls.get_class_name())
        # Handle normal camelCase → camel_Case
        s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
        return s2.lower()

    def get_filename(self, ext: str | None = None, spec: SpatialSpec | None = None) -> str:
        resolved_ext = ext or self.get_ext()
        return f'{self.get_file_stem()}.{resolved_ext}'

    def resolve_cache_paths(self, spec: SpatialSpec) -> CachePathConstructor:
        spec = self.resolve_spec(spec)
        return CachePathConstructor.from_state_hash(self.get_state_hash(spec), self.get_cache_root())

    def format_for_display(self) -> str:
        return self.get_class_name()

    def format_as_json(self, spec: SpatialSpec | None = None) -> Any:
        from pygeodata.formatting.json import format_json

        d = {
            JSONKeys.CLASS_NAME: self.get_class_name(),
            JSONKeys.PARAMS: format_json(self.get_params(), spec=spec),
            JSONKeys.INSTANCE_HASH: self.get_instance_hash(),
        }
        if spec is not None:
            d[JSONKeys.STATE_HASH] = self.get_state_hash(spec)
        return d

    def resolve_spec(self, spec: SpatialSpec | None) -> SpatialSpec:
        """
        Resolve a potentially underspecified :class:`~pygeodata.types.SpatialSpec`.

        If the spec is already fully defined, it is returned unchanged. Otherwise,
        delegates to ``processor.resolve_spec`` if available.

        Parameters
        ----------
        spec : SpatialSpec, optional
            The spatial specification to resolve.

        Returns
        -------
        SpatialSpec
            The resolved specification.
        """
        spec = spec or get_config().spec
        if spec is None:
            raise ValueError('No spatial specification (spec) provided')
        if spec.is_fully_defined:
            return spec
        resolver = getattr(self.processor, 'resolve_spec', None)
        return resolver(spec) if resolver is not None else spec

    def get_params(self) -> dict[str, Any]:
        """
        Return the instance parameters of the artifact.

        Inspects ``vars(self)`` and filters out private attributes, reserved names,
        and any names listed in :attr:`_exclude_params`.

        Returns
        -------
        dict[str, Any]
            A dict of parameter name-value pairs.
        """
        params = {}
        for key, value in vars(self).items():
            if key in ('name', 'class_name', 'processor', 'driver', 'process', 'load', 'ext', '_load', '_process'):
                continue
            if key.startswith('_'):
                continue

            if key in self._sort_params and isinstance(value, (list, tuple)):
                parsed_value = type(value)(sorted(value, key=repr))
            else:
                parsed_value = value

            params.update({key: parsed_value})
        return params

    def get_params_as_json(self, spec: SpatialSpec | None = None) -> dict[str, Any]:
        from pygeodata.formatting.json import format_json

        return format_json(self.get_params(), spec=spec)  # type: ignore[return-value]

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
    def get_cache_root(cls) -> Path:
        pass

    def get_processed_path(self, spec: SpatialSpec) -> Path:
        """
        Return the full path to the processed output file.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification.

        Returns
        -------
        Path
            Full path to the output file.

        Raises
        ------
        ValueError
            If no extension is available from either the argument or :attr:`ext`.
        """
        spec = self.resolve_spec(spec)
        return self.resolve_cache_paths(spec).directory / self.get_filename(spec=spec)

    def ensure_processed_path(self, spec: SpatialSpec) -> Path:
        path = self.get_processed_path(spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_parameters(self, spec: SpatialSpec) -> None:
        path = self.resolve_cache_paths(spec).params_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(self.get_params_as_json(spec=spec), f, indent=4)

    def write_spec(self, spec: SpatialSpec) -> None:
        path = self.resolve_cache_paths(spec).spec_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(spec.to_dict(), f, indent=4)

    def get_instance_hash(self) -> str:
        """Hash of class code and params — spec-independent. Stable identifier for this artifact instance."""
        from pygeodata.formatting.json import format_json

        state = {
            JSONKeys.DEPENDENCY_TREE_HASH: self.get_dependency_tree_hash(),
            JSONKeys.PARAMS: {k: format_json(v) for k, v in self.get_params().items()},
        }
        if self.processor:
            state[JSONKeys.PROCESSOR_HASH] = calculate_cls_source_hash(self.processor.__class__)
        return calculate_dict_hash(state)

    def get_state_hash(self, spec: SpatialSpec) -> str:
        """Hash of instance identity combined with spec — unique per (artifact, spec) pair."""
        return calculate_dict_hash(
            {
                JSONKeys.INSTANCE_HASH: self.get_instance_hash(),
                SpecKeys.SPEC: spec.to_dict(),
            },
        )

    def write_cache_metadata(self, spec: SpatialSpec, co_outputs: tuple[str, ...] = ()) -> None:
        """
        Write the state hash and source hierarchy hash to the hash file.

        Creates the parent directory if it does not exist.

        Parameters
        ----------
        spec : SpatialSpec
            The spatial specification for which to write the hash.
        co_outputs : tuple[str, ...]
            State hashes of artifacts produced in the same _process call.
        """
        hash_path = self.resolve_cache_paths(spec).state_hash_path
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        EntryRecord(
            class_name=self.get_class_name(),
            source_hash=calculate_cls_source_hash(self.__class__),
            dependency_tree_hash=self.get_dependency_tree_hash(),
            instance_hash=self.get_instance_hash(),
            state_hash=self.get_state_hash(spec),
            object_type=self.object_type.get_class_name(),
            hash_path=str(hash_path),
            co_output_hashes=list(co_outputs),
            format_version=FORMAT_VERSION,
        ).dump(hash_path)

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
        hash_path = self.resolve_cache_paths(spec).state_hash_path
        if not hash_path.exists():
            return None
        return EntryRecord.from_file(hash_path).state_hash

    def processed_path_exists(self, spec: SpatialSpec) -> bool:
        return self.get_processed_path(spec).exists()

    def is_processed_hash_present(self, spec: SpatialSpec) -> bool:
        return self.resolve_cache_paths(spec).state_hash_path.exists()

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
        hash_file = self.resolve_cache_paths(spec).state_hash_path
        if not hash_file.exists():
            return False

        record = EntryRecord.from_file(hash_file)
        if record.format_version != FORMAT_VERSION:
            return False
        if record.state_hash is None:
            return False

        return record.state_hash == self.get_state_hash(spec)

    def get_runtime_dependency_graph(self, spec: SpatialSpec) -> RuntimeDependencyGraph:
        nodes: dict[str, RuntimeNode] = {}
        param_edges: set[RuntimeParamEdge] = set()

        def collect(artifact: Artifact) -> None:
            node_id = artifact.get_instance_hash()
            if node_id in nodes:
                return

            params = artifact.get_params()

            call_deps = artifact.get_call_dependencies()
            inh_deps = artifact.get_inheritance_dependencies()

            nodes[node_id] = RuntimeNode(
                node_id=node_id,
                cls=artifact.__class__,
                name=artifact.get_class_name(),
                params=params,
                call_dependencies=tuple(call_deps),
                inheritance_dependencies=tuple(inh_deps),
            )

            def walk(value: Any, name: str) -> None:
                if isinstance(value, Artifact):
                    dep_id = value.get_instance_hash()
                    param_edges.add(RuntimeParamEdge(src_id=dep_id, dst_id=node_id, param_name=name))
                    collect(value)
                    return

                if isinstance(value, dict):
                    for k, item in value.items():
                        walk(item, f'{name}[{k}]')
                    return

                if isinstance(value, (list, tuple, set)):
                    for i, item in enumerate(value):
                        walk(item, f'{name}[{i}]')
                    return

            for k, v in params.items():
                walk(v, k)

        collect(self)
        return RuntimeDependencyGraph(nodes=nodes, param_edges=param_edges)

    def plot_runtime_execution_graph(self, spec: SpatialSpec) -> None:
        graph_data = self.get_runtime_dependency_graph(spec)
        graph_path = self.resolve_cache_paths(spec).execution_graph_path
        plot_compact_execution_graph(
            graph_data=graph_data,
            root_id=self.get_instance_hash(),
            path=graph_path,
            show_params=True,
            show_inheritance=True,
            show_calls=True,
        )

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
        spec = self.resolve_spec(spec)

        if self.is_processed(spec):
            return

        processed_path = self.get_processed_path(spec)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        data_lock_path = processed_path.parent / 'process.lock'

        with FileLock(data_lock_path, timeout=3600):
            if self.is_processed_hash_present(spec) and not self.is_cache_valid(spec):
                print(f'# Cache invalid for {self.get_class_name()}. Recomputing ...')

            produced = self._process(spec)
            artifacts = () if produced is None else tuple(produced)

            # Deduplicate by state hash; self appended last so yielded artifacts take precedence
            seen: dict[str, Artifact] = {}
            for a in (*artifacts, self):
                h = a.get_state_hash(spec)
                if h not in seen:
                    seen[h] = a

            for state_hash, artifact in seen.items():
                others = tuple(h for h in seen if h != state_hash)
                artifact.update_registry()
                artifact.write_parameters(spec)
                artifact.write_spec(spec)
                artifact.write_cache_metadata(spec, co_outputs=others)

                if next(extract_instances(artifact.get_params(), TrackedObject), None) is not None:
                    self.plot_runtime_execution_graph(spec)
