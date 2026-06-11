from pygeodata.artifact import Artifact
from pygeodata.data import Data
from pygeodata.protocols import T
from pygeodata.spec import SpatialSpec


def process(artifact: Artifact, spec: SpatialSpec | None = None) -> None:
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

    Raises
    ------
    ValueError
        If no spec is provided and none is set in the global config.
    """
    return artifact.process(spec)


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
    return artifact.load(spec)
