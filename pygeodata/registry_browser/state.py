import threading
from dataclasses import dataclass

from pygeodata.registry_browser.class_catalog import (
    discover_loaded_classes,
    merge_unloaded_classes,
)
from pygeodata.registry_browser.entry_catalog import discover_entries
from pygeodata.registry_browser.models import ClassInfo, EntryInfo, GroupInfo
from pygeodata.spec import SpecKeys
from pygeodata.versioning import VersionRegistry


@dataclass(slots=True)
class AppState:
    classes: dict[str, ClassInfo]
    entries: dict[str, EntryInfo]
    groups: dict[str, GroupInfo]
    diagnostics: dict
    spec_options: dict[str, list[str]]
    code_groups: dict[str, list[dict]]


class AppContext:
    def __init__(self) -> None:
        self.state: AppState | None = None
        self.ready = threading.Event()
        self.progress: dict = {}

    def start_load(self) -> None:
        def _load() -> None:
            self.state = build_state(progress=self.progress)
            self.ready.set()

        threading.Thread(target=_load, daemon=True).start()

    def start_reload(self) -> None:
        self.ready.clear()
        self.progress.clear()
        self.start_load()

    def is_loading(self) -> bool:
        return not self.ready.is_set()


def build_state(progress: dict | None = None) -> AppState:
    from pygeodata.tracked_object import TrackedObject
    entries, groups, diagnostics = discover_entries(progress=progress)
    registry = VersionRegistry.instance()
    for entry in entries.values():
        if entry.dep_hash and not entry.dep_hash_stale and TrackedObject.find_object_class(entry.class_name) is None:
            entry.dep_hash_stale = registry.is_dep_hash_stale(entry.dep_hash)
    classes = merge_unloaded_classes(discover_loaded_classes(), groups, entries=entries, version_registry=registry)
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
        groups=groups,
        diagnostics=diagnostics,
        spec_options=spec_options,
        code_groups=registry.code_groups,
    )
