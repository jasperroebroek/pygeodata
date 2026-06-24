import contextlib
import inspect
import json
from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.data import Data
from pygeodata.paths import CACHE_META_FILES, CachePathConstructor
from pygeodata.protocols import T
from pygeodata.registries.registry import EntryRegistry
from pygeodata.spec import SpatialSpec
from pygeodata.tracked_object import TrackedObject


def _params_hint(params_path: Path | None, cls: type) -> str:
    stored: dict = {}
    if params_path is not None and params_path.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            stored = json.loads(params_path.read_text(encoding='utf-8'))
    if stored:
        return f'Pass them via params={{...}}. Stored parameters for this entry: {stored}'
    try:
        sig = inspect.signature(cls.__init__)
        names = [n for n in sig.parameters if n != 'self']
    except (ValueError, TypeError):
        names = []
    if names:
        return f'Pass them via params={{...}}. Expected parameters: {names}'
    return 'Pass them via params={...}.'


def _resolve_output_path(instance: Artifact, directory: Path, full_hash: str) -> Path:
    """Return the output file path for *instance* inside *directory*.

    Tries the class-derived filename first; falls back to scanning the directory
    for non-meta files. Raises FileNotFoundError when nothing is found or when
    multiple candidates are present (listing them so the caller knows what's there).
    """
    try:
        candidate = directory / instance.get_filename()
        if candidate.exists():
            return candidate
    except ValueError:
        pass
    files = [p for p in directory.iterdir() if p.name not in CACHE_META_FILES]
    if not files:
        raise FileNotFoundError(f'No output file found for hash {full_hash!r} in {directory}')
    if len(files) > 1:
        names = ', '.join(p.name for p in sorted(files))
        raise FileNotFoundError(
            f'Multiple output files for hash {full_hash!r} in {directory}: {names}. '
            f'Cannot determine which to load automatically.',
        )
    return files[0]


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


def load_from_hash(
    state_hash: str,
    filename: str | None = None,
    params: dict | None = None,
) -> object:
    """Load a cached output by (truncated) state hash.

    Looks up the full hash in :class:`~pygeodata.registry.EntryRegistry`,
    resolves the concrete :class:`~pygeodata.data.Data` subclass from the
    registry record, and loads the output without re-running ``process``.

    Parameters
    ----------
    state_hash : str
        Full or truncated state hash identifying the cache entry.
    filename : str, optional
        Filename of the output file within the cache directory. Required when
        the entry directory contains multiple output files. When omitted, the
        filename is derived from the class or discovered by scanning the directory.
    params : dict, optional
        Instance attributes to set on the loader before calling ``_load``.
        Required for loaders whose ``_load`` / ``driver`` accesses instance
        attributes (e.g. ``self.path``, ``self.scale``). The stored
        ``parameters.json`` is shown in the error message when this is needed
        but not provided.

    Raises
    ------
    KeyError
        If *state_hash* matches zero or more than one entry.
    FileNotFoundError
        If the resolved output file does not exist on disk.
    TypeError
        If the matched class's ``_load`` requires instance attributes that were
        not supplied via *params*.
    RuntimeError
        If the matched class is not imported/registered in the current process.
    """
    registry = EntryRegistry()
    full_hash = registry.resolve_hash_prefix(state_hash)
    if full_hash is None:
        raise KeyError(f'No entry found for hash prefix {state_hash!r}')
    record = registry.records[full_hash]
    if record.hash_path is None:
        raise FileNotFoundError(f'Entry {full_hash!r} has no hash_path')
    cls = TrackedObject.find_object_class(record.class_name)
    if cls is None:
        raise RuntimeError(
            f'Class {record.class_name!r} is not registered in the current process. '
            f'Import the module that defines it before calling load_from_hash.',
        )
    instance = cls.__new__(cls)
    if params is not None:
        for k, v in params.items():
            setattr(instance, k, v)
    resolver = CachePathConstructor.from_path(Path(record.hash_path))
    if filename is not None:
        path = resolver.directory / filename
        if not path.exists():
            raise FileNotFoundError(f'Output file {filename!r} not found in {resolver.directory}')
    else:
        path = _resolve_output_path(instance, resolver.directory, full_hash)
    try:
        return instance._load(path)
    except AttributeError as e:
        raise TypeError(
            f'{cls.__name__}._load requires instance attributes that are not set. '
            f'{_params_hint(record.params_path, cls)}',
        ) from e
