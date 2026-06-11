from __future__ import annotations

from typing import TYPE_CHECKING

from pygeodata.api import process
from pygeodata.artifact import Artifact
from pygeodata.extraction import extract_instances

try:
    from dask.delayed import delayed as dask_delayed

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

if TYPE_CHECKING:
    from dask.delayed import Delayed

    from pygeodata.data import Data
    from pygeodata.spec import SpatialSpec


def build_dask_graph(
    artifact: Data,
    spec: SpatialSpec | None = None,
    _cache: dict[str, Delayed] | None = None,
) -> Delayed:
    """
    Recursively build a Dask delayed computation graph for a artifact and its dependencies.

    Traverses all :class:`~pygeodata.data.Data` instances embedded in the artifact's
    parameters`) and constructs a graph where each node is a delayed call to
    :func:`~pygeodata.api.process`. Dependencies are wired as upstream tasks so Dask can
    schedule them in the correct order.

    Parameters
    ----------
    artifact : Data
        The root artifact for which to build the graph.
    spec : SpatialSpec, optional
        Spatial specification passed to each :func:`~pygeodata.base.process` call.
    _cache : dict[str, Delayed], optional
        Internal memoization cache keyed by artifact state hash. Shared across recursive
        calls to avoid duplicate nodes. Should not be provided by the caller.

    Returns
    -------
    dask.delayed.Delayed
        A delayed task representing the root artifact, with all upstream dependencies
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
        graph = build_dask_graph(my_artifact, spec=my_spec)
        graph.compute()
    """
    if not HAS_DASK:
        raise ImportError('Dask is required for parallel processing. Install it with: pip install pygeodata[parallel]')

    delayed = dask_delayed

    if _cache is None:
        _cache = {}

    node_id = artifact.get_state_hash(spec)
    if node_id in _cache:
        return _cache[node_id]

    deps = list(extract_instances(artifact.get_params(), Artifact))

    delayed_deps = [build_dask_graph(dep, spec, _cache) for dep in deps]

    @delayed
    def run_node(l: Data, s: SpatialSpec | None, *args) -> Data:
        process(l, s)
        return l

    task_name = f'{artifact.get_class_name()}-{node_id[:8]}'
    task = run_node(artifact, spec, *delayed_deps, dask_key_name=task_name)
    _cache[node_id] = task
    return task
