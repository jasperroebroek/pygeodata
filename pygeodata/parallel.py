from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dask.delayed import Delayed
from pygeodata.base import process

# Import your top level process function
from pygeodata.loader import DataLoader, extract_dataloaders
from pygeodata.types import SpatialSpec


def build_dask_graph(
    loader: DataLoader,
    spec: SpatialSpec | None = None,
    _cache: dict[str, Delayed] | None = None,
) -> Delayed:
    """
    Recursively build a Dask delayed computation graph for a loader and its dependencies.

    Traverses all :class:`~pygeodata.loader.DataLoader` instances embedded in the loader's
    parameters (via :func:`~pygeodata.loader.extract_dataloaders`) and constructs a graph
    where each node is a delayed call to :func:`~pygeodata.base.process`. Dependencies are
    wired as upstream tasks so Dask can schedule them in the correct order.

    Parameters
    ----------
    loader : DataLoader
        The root loader for which to build the graph.
    spec : SpatialSpec, optional
        Spatial specification passed to each :func:`~pygeodata.base.process` call.
    _cache : dict[str, Delayed], optional
        Internal memoization cache keyed by loader state hash. Shared across recursive
        calls to avoid duplicate nodes. Should not be provided by the caller.

    Returns
    -------
    dask.delayed.Delayed
        A delayed task representing the root loader, with all upstream dependencies
        already wired in the graph.

    Notes
    -----
    The Dask task name for each node is ``"{ClassName}-{hash[:8]}"`` for easier
    identification in the Dask dashboard.

    Examples
    --------
    .. code-block:: python

        from dask.distributed import Client
        from pygeodata.parallel import build_dask_graph

        client = Client()
        graph = build_dask_graph(my_loader, spec=my_spec)
        graph.compute()
    """
    try:
        from dask.delayed import delayed
    except ImportError as err:
        raise ImportError('Dask is required for parallel processing.') from err

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
