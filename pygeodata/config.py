import importlib.util
import sys
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from pygeodata.enums import ObjectType
from pygeodata.rasters import RasterCreationOptions
from pygeodata.spec import SpatialSpec

FORMAT_VERSION = 2


@dataclass
class Config:
    path_cache: Path = Path('data_processed')
    path_tables: Path = Path('tables')
    path_figures: Path = Path('figures')
    path_registry: Path = Path('.source')
    path_catalog: Path = Path('.catalog.json')
    catalog_enabled: bool = True
    num_threads: int = 1
    warp_mem_limit: int = 0  # GDAL default, indicates 64 MB
    spec: SpatialSpec | None = None
    raster_creation_options: RasterCreationOptions = field(default_factory=RasterCreationOptions)
    removable_system_files: tuple[str] = (
        '.DS_Store',
        'Thumbs.db',
        'desktop.ini',
        '.Trash-1000',
    )
    removable_system_suffixes: tuple[str] = (
        '.tmp',
        '~',
    )

    def get_cache_root(self, object_type: ObjectType) -> Path:
        return {
            ObjectType.DATA: self.path_cache,
            ObjectType.TABLE: self.path_tables,
            ObjectType.FIGURE: self.path_figures,
        }[object_type]

    @property
    def cache_roots(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(self.get_cache_root(t) for t in ObjectType))

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f'Invalid config key: {key}')
            current = getattr(self, key)
            if isinstance(value, str) and isinstance(current, Path):
                setattr(self, key, Path(value).expanduser())
            else:
                setattr(self, key, value)


CONFIG = Config()


def get_config() -> Config:
    return CONFIG


def load_project_settings(
    directory: Path | None = None,
    settings_path: Path | None = None,
) -> list[Path]:
    """
    Import every ``pygeodata_settings.py`` — plain Python configuration shared by code and CLI.

    The settings modules apply the same :meth:`~pygeodata.config.Config.update` calls
    the Python code uses, so CLI commands see the project's configuration without
    importing the whole project. The CLI imports them automatically before running
    any command; Python code can simply import them (or the project's own settings
    module) instead.

    Parameters
    ----------
    directory : Path, optional
        The directory to look in. Defaults to the working directory.
    settings_path : Path, optional
        An additional settings module to import, alongside the discovered ones.

    Returns
    -------
    list[Path]
        The imported settings module paths, in load order: ``settings_path`` (when
        given), then ``pygeodata_settings.py``, then ``src/pygeodata_settings.py``
        in the directory — no parent directories are searched, and missing files
        are skipped. A module that exists but fails to import raises, and modules
        that set the same config key raise, so a broken or ambiguous configuration
        never silently wins.
    """
    base = Path.cwd() if directory is None else directory
    candidates: list[Path] = []
    if settings_path is not None:
        candidates.append(settings_path)
    candidates += [base / 'pygeodata_settings.py', base / 'src' / 'pygeodata_settings.py']

    loaded: list[Path] = []
    keys_by_source: dict[Path, set[str]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        module_name = _settings_module_name(path, base)
        cached = sys.modules.get(module_name)
        if (
            cached is not None
            and getattr(cached, '__file__', None) is not None
            and Path(cached.__file__).resolve() == path.resolve()
        ):
            keys_by_source[path] = set(getattr(cached, '_config_keys', set()))
            loaded.append(path)
            continue
        keys_by_source[path] = _import_settings_module(path, module_name)
        loaded.append(path)

    key_sources: dict[str, list[Path]] = {}
    for path, keys in keys_by_source.items():
        for key in keys:
            key_sources.setdefault(key, []).append(path)
    overlaps = {key: sources for key, sources in key_sources.items() if len(sources) > 1}
    if overlaps:
        detail = '; '.join(f'{key} set by {", ".join(str(s) for s in sources)}' for key, sources in overlaps.items())
        raise ValueError(f'Settings modules set overlapping config keys: {detail}')
    return loaded


def _settings_module_name(path: Path, base: Path) -> str:
    """Dotted path relative to the project base, so same-named files in different folders coexist."""
    try:
        relative = path.resolve().relative_to(base.resolve()).with_suffix('')
    except ValueError:
        relative = Path(path.stem)
    return '.'.join(relative.parts)


def _import_settings_module(path: Path, module_name: str) -> set[str]:
    """Import one settings module, recording which config keys it sets through ``update()``."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import settings module {path}')
    keys: set[str] = set()
    original_update = CONFIG.update

    def _recording_update(**kwargs) -> None:
        keys.update(kwargs)
        original_update(**kwargs)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    CONFIG.update = _recording_update  # type: ignore[method-assign]
    try:
        spec.loader.exec_module(module)
    finally:
        del CONFIG.update  # type: ignore[attr-defined]
    module._config_keys = keys  # type: ignore[attr-defined]
    return keys


@contextmanager
def set_config(**overrides: Any) -> Generator[Config, None, None]:
    old_values = {k.name: getattr(CONFIG, k.name) for k in fields(CONFIG)}
    CONFIG.update(**overrides)
    try:
        yield CONFIG
    finally:
        CONFIG.update(**old_values)
