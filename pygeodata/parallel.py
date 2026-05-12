from dask.delayed import Delayed, delayed

from pygeodata.base import process

# Import your top level process function
from pygeodata.loader import DataLoader, extract_dataloaders
from pygeodata.types import SpatialSpec


def build_dask_graph(
    loader: DataLoader,
    spec: SpatialSpec | None = None,
    _cache: dict[str, Delayed] | None = None,
) -> Delayed:
    if _cache is None:
        _cache = {}

    node_id = loader.get_state_hash()
    if node_id in _cache:
        return _cache[node_id]

    deps = list(extract_dataloaders(loader.get_params(exclude=False)))

    delayed_deps = [build_dask_graph(dep, spec, _cache) for dep in deps]

    @delayed
    def run_node(l: DataLoader, s: SpatialSpec | None, *args) -> DataLoader:
        process(l, s)
        return l

    task_name = f'{loader.get_class_name()}-{node_id[:8]}'
    task = run_node(loader, spec, *delayed_deps, dask_key_name=task_name)
    _cache[node_id] = task
    return task
