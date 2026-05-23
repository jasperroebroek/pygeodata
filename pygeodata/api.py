from pygeodata.artifact import Artifact
from pygeodata.config import get_config
from pygeodata.data import Data
from pygeodata.types import SpatialSpec, T


def process(artifact: Artifact, spec: SpatialSpec | None = None) -> SpatialSpec:
    """
    Process an Artifact for a given spatial specification.

    Resolves the spec via the Artifact, then calls :meth:`~pygeodata.data.Data.process`
    if the output is not already cached. Falls back to the global config spec if none is provided.

    Parameters
    ----------
    loader : Data
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
    spec = artifact.resolve_spec(spec)

    if not artifact.is_processed(spec):
        artifact.process(spec)

    return spec


def load(artifact: Data[T], spec: SpatialSpec | None = None) -> T:
    """
    Process and load data for a given Artifact and spatial specification.

    Calls :func:`process` first to ensure the output exists, then reads and
    returns the data via the Artifact's driver.

    Parameters
    ----------
    loader : Data[T]
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
    spec = process(artifact, spec)
    return artifact.load(spec)
