import ast
import functools
import hashlib
import inspect
import json
import re
import textwrap
from collections import deque
from collections.abc import Generator, Iterable
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic

import numpy as np
from filelock import FileLock

from pygeodata.config import get_config
from pygeodata.hash import define_hash_from_class
from pygeodata.paths import generate_path
from pygeodata.types import Driver, Processor, SpatialSpec, T


def extract_dataloaders(value: Any) -> Generator['DataLoader']:
    """Recursively yields all DataLoader instances in nested structures."""
    if isinstance(value, DataLoader):
        yield value
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from extract_dataloaders(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from extract_dataloaders(item)


def parse_dict_to_json(value: Any) -> Any:
    JSON_SAFE = (str, int, float, bool, type(None))
    if isinstance(value, JSON_SAFE):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, DataLoader):
        return {'_class_name': value.get_class_name(), **parse_dict_to_json(value.get_params(exclude=False))}
    if isinstance(value, dict):
        return {k: parse_dict_to_json(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [parse_dict_to_json(i) for i in value]
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return repr(value)


def _get_recursive_state_hash(value: Any) -> Any:
    """Recursively extracts execution hashes from deeply nested parameters."""
    if isinstance(value, DataLoader):
        return value.get_state_hash()
    if isinstance(value, (list, tuple)):
        return [_get_recursive_state_hash(i) for i in value]
    if isinstance(value, dict):
        return {str(k): _get_recursive_state_hash(v) for k, v in sorted(value.items())}
    return repr(value)


def _serialize_path_value(value: Any) -> str:
    if isinstance(value, Enum):
        return value.name.lower()

    if isinstance(value, DataLoader):
        return value.get_class_name()

    if isinstance(value, (tuple, list)):
        inner = ', '.join(_serialize_path_value(v) for v in value)
        return f'({inner})'

    if isinstance(value, dict):
        inner = ', '.join(f'{k}:{_serialize_path_value(v)}' for k, v in value.items())
        return f'{{{inner}}}'

    return str(value)


def _flatten_param(prefix: str, value: Any) -> dict[str, Any]:
    flat: dict[str, Any] = {}

    if isinstance(value, DataLoader):
        flat[prefix] = value.get_class_name()
        for nk, nv in value.get_flat_params(exclude=False).items():
            flat[f'{prefix}__{nk}'] = nv

    elif isinstance(value, (list, tuple)):
        loader_indices = {idx for idx, item in enumerate(value) if any(extract_dataloaders(item))}
        flat[prefix] = _serialize_path_value(value)
        for idx in loader_indices:
            flat.update(_flatten_param(f'{prefix}_{idx}', value[idx]))

    elif isinstance(value, dict):
        keys = sorted(value.keys())
        loader_keys = {k for k in keys if any(extract_dataloaders(value[k]))}
        flat[prefix] = _serialize_path_value(value)
        for k in loader_keys:
            flat.update(_flatten_param(f'{prefix}_{k}', value[k]))

    else:
        flat[prefix] = _serialize_path_value(value)

    return flat


class DataLoader(Generic[T]):
    _sort_params: tuple[str] = ()
    _exclude_params: tuple[str] = ()
    _exclude_params_from_path: tuple[str] = ()

    def __repr__(self) -> str:
        params = self.get_params()
        parts = [f'{k}={v!r}' for k, v in sorted(params.items())]
        return f'{self.get_class_name()}({", ".join(parts)})'

    def resolve_spec(self, spec: SpatialSpec) -> SpatialSpec:
        if spec.is_fully_defined:
            return spec
        resolver = getattr(self.processor, 'resolve_spec', None)
        return resolver(spec) if resolver is not None else spec

    @property
    def processor(self) -> Processor | None:
        return None

    @property
    def driver(self) -> Driver:
        processor = self.processor
        if processor is None:
            raise NotImplementedError(f'{self}: Either processor or driver must be implemented')

        driver = getattr(processor, 'default_driver')
        if driver is None:
            raise AttributeError(f'Processor {processor} lacks default_driver and no driver is set')

        return driver

    @classmethod
    def get_class_name(cls) -> str:
        return cls.__name__.split('.')[-1]

    @classmethod
    def get_name(cls) -> str:
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
        """Extracts parameters attached to the loader instance."""
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
        return parse_dict_to_json(self.get_params(exclude=exclude))

    def get_flat_params(self, exclude: bool = True) -> dict[str, Any]:
        """Recursively flatten parameters for deterministic path generation."""
        flat_params = {}
        for k, v in self.get_params(exclude=exclude).items():
            flat_params.update(_flatten_param(k, v))
        return flat_params

    def get_src_path(self) -> Path:
        processor = self.processor
        if processor is None:
            raise NotImplementedError('Processor must be implemented to get src_path')

        if not hasattr(processor, 'src_path'):
            raise NotImplementedError(f'Processor {processor} lacks src_path')

        return Path(getattr(processor, 'src_path'))

    def get_processed_dir(self, spec: SpatialSpec) -> Path:
        spec = self.resolve_spec(spec)

        flat_params = self.get_flat_params()

        return generate_path(
            spec=spec,
            name=self.get_class_name(),
            base_dir=get_config().path_data_processed,
            **flat_params,
        )

    def get_processed_path(self, spec: SpatialSpec, ext: str | None = None) -> Path:
        ext = ext or self.ext
        if ext is None:
            raise ValueError('ext must be specified as parameter or as instance variable')
        return self.get_processed_dir(spec) / f'{self.get_name()}.{ext}'

    def write_parameters(self, spec: SpatialSpec) -> None:
        spec = self.resolve_spec(spec)
        path = self.get_processed_dir(spec) / 'parameters.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(self.get_params_as_json(), f, indent=4)

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
        base = get_config().path_data_processed / '_source' / module_path / cls.get_class_name()
        base.mkdir(parents=True, exist_ok=True)
        return base

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
        Parses AST to get a formatting-agnostic hash and hidden dependencies.
        Cached in memory to ensure AST parsing only happens once per class per run.
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
        """Cached entry point — only call without visited argument."""
        d: dict[str, Any] = cls._get_dependency_tree_recursive(visited=None)
        return d

    @classmethod
    def get_dependency_graph(cls) -> dict[str, Any]:
        """
        Builds a graph of DataLoader class dependencies starting from this class.
        Separates call dependencies from inheritance dependencies.
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
        """Computes the execution hash purely from the canonical dependency tree."""
        tree = cls.get_dependency_tree()
        return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()

    @classmethod
    def save_source_code_and_hash(cls) -> None:
        """Saves the current AST hash and writes the source code for inspection."""
        from pygeodata.visualisations import plot_class_dependency_graph

        tree = cls.get_dependency_tree()

        json_file = cls.get_source_registry_path()
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
        """Checks if the source.py file on disk matches the live Python code."""
        registry_file = cls.get_source_registry_path()
        if not registry_file.exists():
            return False

        with Path.open(registry_file, encoding='utf-8') as f:
            saved_state = json.load(f)

        return saved_state.get('source_hierarchy_hash') == cls.get_source_hierarchy_hash()

    @classmethod
    def initialize_source_registry(cls) -> None:
        """Ensures this class and its hidden dependencies have an up-to-date source.py file."""
        if not cls.is_source_registry_valid():
            cls.save_source_code_and_hash()

        for dep_class in cls.get_source_metadata()['all_dependencies']:
            dep_class.initialize_source_registry()

    def get_state_hash(self) -> str:
        """Recursively hashes the class code AND all injected parameter DataLoaders."""
        state = {
            'blueprint_hash': self.get_source_hierarchy_hash(),
            'params': {k: _get_recursive_state_hash(v) for k, v in self.get_params().items()},
        }

        if self.processor:
            state['processor_hash'] = define_hash_from_class(self.processor.__class__)

        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def get_state_hash_path(self, spec: SpatialSpec) -> Path:
        return self.get_processed_path(spec).with_suffix('.hash.json')

    def write_state_hash(self, spec: SpatialSpec) -> None:
        hash_path = self.get_state_hash_path(spec)
        hash_path.parent.mkdir(parents=True, exist_ok=True)
        with Path.open(hash_path, 'w', encoding='utf-8') as f:
            json.dump(
                {'source_hierarchy_hash': self.get_source_hierarchy_hash(), 'state_hash': self.get_state_hash()},
                f,
                indent=4,
            )

    def read_state_hash(self, spec: SpatialSpec) -> str | None:
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
        hash_file = self.get_state_hash_path(spec)
        if not hash_file.exists():
            return False

        saved_state_hash = self.read_state_hash(spec)
        if saved_state_hash is None:
            return False

        return saved_state_hash == self.get_state_hash()

    def is_processed(self, spec: SpatialSpec) -> bool:
        if not self.is_cache_valid(spec):
            return False
        return self.processed_path_exists(spec)

    def _process(self, spec: SpatialSpec) -> Iterable['DataLoader'] | None:
        spec = self.resolve_spec(spec)
        processor = self.processor
        if processor is None:
            raise NotImplementedError('Either load, processor or process must be implemented')
        processor(self.get_processed_path(spec), spec)

    def process(self, spec: SpatialSpec) -> None:
        from pygeodata.visualisations import plot_compact_execution_graph

        spec = self.resolve_spec(spec)

        code_registry_dir = self.get_source_registry_dir()
        code_lock_path = code_registry_dir / 'source.lock'
        with FileLock(code_lock_path, timeout=60):
            self.initialize_source_registry()

        processed_path = self.get_processed_path(spec)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.write_parameters(spec)
        data_lock_path = processed_path.parent / 'process.lock'

        with FileLock(data_lock_path, timeout=3600):
            if self.is_processed(spec):
                return

            if self.is_processed_hash_present(spec) and not self.is_cache_valid(spec):
                print(f'# Cache invalid for {self.get_class_name()}. Recomputing ...')

            produced = self._process(spec)
            loaders = (self,) if produced is None else tuple(produced)
            for loader in loaders:
                loader.write_state_hash(spec)

            if len(list(extract_dataloaders(self.get_params(exclude=False)))) > 0:
                plot_compact_execution_graph(loader=self, out_path=processed_path.with_name('execution_graph'))

    def load(self, spec: SpatialSpec) -> T:
        spec = self.resolve_spec(spec)
        return self.driver(self.get_processed_path(spec))
