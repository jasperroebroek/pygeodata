from pygeodata.config import get_config
from pygeodata.loader import DataLoader
from pygeodata.types import SpatialSpec, T


def process(loader: DataLoader, spec: SpatialSpec | None = None) -> SpatialSpec:
    """
    Process a loader for a given spatial specification.

    Resolves the spec via the loader, then calls :meth:`~pygeodata.loader.DataLoader.process`
    if the output is not already cached. Falls back to the global config spec if none is provided.

    Parameters
    ----------
    loader : DataLoader
        The loader to process.
    spec : SpatialSpec, optional
        Spatial specification defining CRS, transform, and shape. If not provided,
        the global config spec is used.

    Returns
    -------
    SpatialSpec
        The resolved spatial specification used for processing.

    Raises
    ------
    ValueError
        If no spec is provided and none is set in the global config.
    """
    spec = spec or get_config().spec
    if spec is None:
        raise ValueError('No spatial specification (spec) provided')
    spec = loader.resolve_spec(spec)

    if not loader.is_processed(spec):
        loader.process(spec)

    return spec


def load(loader: DataLoader[T], spec: SpatialSpec | None = None) -> T:
    """
    Process and load data for a given loader and spatial specification.

    Calls :func:`process` first to ensure the output exists, then reads and
    returns the data via the loader's driver.

    Parameters
    ----------
    loader : DataLoader[T]
        The loader to process and load from.
    spec : SpatialSpec, optional
        Spatial specification. Falls back to the global config spec if not provided.

    Returns
    -------
    T
        The loaded data, as returned by the loader's driver.

    Raises
    ------
    ValueError
        If no spec is provided and none is set in the global config.
    """
    spec = process(loader, spec)
    return loader.load(spec)
