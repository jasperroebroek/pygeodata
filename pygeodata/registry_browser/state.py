import contextlib
import importlib.util
import logging
import sys
import threading
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pygeodata.catalog.class_catalog import (
    discover_loaded_classes,
    merge_unloaded_classes,
)
from pygeodata.catalog.entry_catalog import _cache_file, discover_entries
from pygeodata.catalog.types import ClassInfo, EntryInfo
from pygeodata.registries.registry import EntryRegistry
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.spec import SpecKeys
from pygeodata.tracked_object import TrackedObject

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
                    root = Path.cwd()
                    py_files = _collect_reimport_files(root)
                    total = len(py_files)
                    self.progress['reimport_done'] = 0
                    self.progress['reimport_total'] = total
                    TrackedObject._registry.clear()
                    for done in _reimport_modules(py_files, root):
                        self.progress['reimport_done'] = done
                    self.progress['reimport_done'] = total
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


def _collect_reimport_files(root: Path) -> list[Path]:
    """Return project .py files under root sorted deepest-first.

    Deepest-first means leaf modules are loaded before the top-level packages
    that import them, so cascading re-imports are mostly no-ops.
    """
    files = [
        py_file
        for py_file in root.rglob('*.py')
        if not any(p.startswith(('.', '__pycache__')) or p in _SKIP_DIRS for p in py_file.relative_to(root).parts)
        and not py_file.relative_to(root).parts[-1].startswith(_SKIP_PREFIXES)
    ]
    files.sort(key=lambda p: (-len(p.parts), p.name == '__init__.py', p))
    return files


def _reimport_modules(py_files: list[Path], root: Path):
    """Import (or re-execute) a list of project .py files.

    Unlike the CLI helper this re-executes already-loaded modules so edits to
    existing classes are picked up.  Per-module failures are logged and skipped.

    Yields the index of each file just before it is processed so callers can
    track progress against len(py_files).
    """
    if root not in sys.path:
        sys.path.insert(0, str(root))

    module_names = {
        py_file: '.'.join(py_file.relative_to(root).with_suffix('').parts)
        for py_file in py_files
    }
    for name in module_names.values():
        sys.modules.pop(name, None)

    for i, py_file in enumerate(py_files):
        module_name = module_names[py_file]
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001
            _log.debug('reimport skip %s', module_name, exc_info=True)
        yield i


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
    def _none_first(keys: Iterable[str]) -> list[str]:
        return sorted(keys, key=lambda k: (k != 'None', k))

    bounds_by_key = {str(entry.spec.bounds): (entry.spec.bounds_display or 'None') for entry in entries.values()}
    resolution_keys = _none_first({entry.spec.resolution_display or 'None' for entry in entries.values()})
    spec_options = {
        SpecKeys.CRS: sorted({entry.spec.crs for entry in entries.values() if entry.spec.crs}),
        SpecKeys.RESOLUTION: [{'value': k, 'label': k} for k in resolution_keys],
        SpecKeys.SHAPE: sorted({entry.spec.shape for entry in entries.values() if entry.spec.shape}),
        SpecKeys.BOUNDS: [{'value': k, 'label': bounds_by_key[k]} for k in _none_first(bounds_by_key)],
    }
    return AppState(
        classes=classes,
        entries=entries,
        diagnostics=diagnostics,
        spec_options=spec_options,
        entry_registry=entry_registry,
        version_registry=version_registry,
    )
