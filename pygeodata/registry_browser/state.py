import contextlib
import importlib.util
import logging
import sys
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path

from pygeodata.registries.registry import EntryRegistry
from pygeodata.catalog.class_catalog import (
    discover_loaded_classes,
    merge_unloaded_classes,
)
from pygeodata.catalog.entry_catalog import _cache_file, discover_entries
from pygeodata.catalog.types import ClassInfo, EntryInfo
from pygeodata.spec import SpecKeys
from pygeodata.registries.versioning import VersionRegistry

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class AppState:
    """Wires the entry and version registries together for the browser.

    ``entries`` is the browser-enriched view produced by ``discover_entries``;
    ``groups`` is a thin accessor over the held entry registry instance.
    """

    classes: dict[str, ClassInfo]
    entries: dict[str, EntryInfo]
    diagnostics: dict
    spec_options: dict[str, list[str]]
    entry_registry: EntryRegistry
    version_registry: VersionRegistry

    def get_state_hashes(self, class_name: str) -> list[str]:
        return self.entry_registry.get_state_hashes(class_name)

    def get_object_type(self, class_name: str) -> str | None:
        return self.entry_registry.get_object_type(class_name)


class AppContext:
    def __init__(self) -> None:
        self.state: AppState | None = None
        self.ready = threading.Event()
        self.progress: dict = {}
        self.load_error: str | None = None

    def start_load(self) -> None:
        def _load() -> None:
            try:
                self.state = build_state(progress=self.progress)
                self.load_error = None
            except Exception:
                self.load_error = traceback.format_exc()
            finally:
                self.ready.set()

        threading.Thread(target=_load, daemon=True).start()

    def start_reload(self, reimport: bool = False) -> None:
        self.ready.clear()
        self.progress.clear()

        def _reload() -> None:
            try:
                _purge_caches()
                if reimport:
                    _reimport_modules(Path.cwd())
                self.state = build_state(progress=self.progress)
                self.load_error = None
            except Exception:
                self.load_error = traceback.format_exc()
            finally:
                self.ready.set()

        threading.Thread(target=_reload, daemon=True).start()

    def is_loading(self) -> bool:
        return not self.ready.is_set()


def _purge_caches() -> None:
    """Delete the two tmp JSONs so the next scan starts clean."""
    with contextlib.suppress(OSError):
        _cache_file().unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        EntryRegistry()._cache_path().unlink(missing_ok=True)


_SKIP_DIRS = {'venv', 'env', '.venv', 'build', 'dist', 'site-packages', 'tests', 'test', 'cache'}
_SKIP_PREFIXES = ('test_', 'conftest', 'setup', 'manage')


def _reimport_modules(root: Path) -> int:
    """Import (or re-execute) all project .py files under root.

    Unlike the CLI helper this re-executes already-loaded modules so edits to
    existing classes are picked up.  Per-module failures are logged and skipped.
    Returns the number of modules processed.
    """
    if root not in sys.path:
        sys.path.insert(0, str(root))
    count = 0
    for py_file in sorted(root.rglob('*.py')):
        parts = py_file.relative_to(root).parts
        if any(p.startswith(('.', '__pycache__')) or p in _SKIP_DIRS for p in parts):
            continue
        if parts[-1].startswith(_SKIP_PREFIXES):
            continue
        module_name = '.'.join(py_file.relative_to(root).with_suffix('').parts)
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                count += 1
        except Exception:  # noqa: BLE001
            _log.debug('reimport skip %s', module_name, exc_info=True)
    return count


def build_state(progress: dict | None = None) -> AppState:
    version_registry = VersionRegistry()
    entries, entry_registry, diagnostics = discover_entries(progress=progress, version_registry=version_registry)
    classes = merge_unloaded_classes(
        discover_loaded_classes(),
        entry_registry,
        entries=entries,
        version_registry=version_registry,
        src=version_registry.source_registry,
        trees=version_registry.tree_registry,
    )
    spec_options = {
        SpecKeys.CRS: sorted({entry.spec.crs for entry in entries.values() if entry.spec.crs}),
        SpecKeys.RESOLUTION: sorted({entry.spec.resolution for entry in entries.values() if entry.spec.resolution}),
        SpecKeys.SHAPE: sorted({entry.spec.shape for entry in entries.values() if entry.spec.shape}),
        SpecKeys.BOUNDS: sorted(
            {str(list(entry.spec.bounds_latlon)) for entry in entries.values() if entry.spec.bounds_latlon},
        ),
    }
    return AppState(
        classes=classes,
        entries=entries,
        diagnostics=diagnostics,
        spec_options=spec_options,
        entry_registry=entry_registry,
        version_registry=version_registry,
    )
