import threading
from dataclasses import dataclass

from pygeodata.registry import EntryRegistry
from pygeodata.registry_browser.class_catalog import (
    discover_loaded_classes,
    merge_unloaded_classes,
)
from pygeodata.registry_browser.entry_catalog import discover_entries
from pygeodata.registry_browser.models import ClassInfo, EntryInfo
from pygeodata.registry_types import GroupRecord
from pygeodata.spec import SpecKeys
from pygeodata.versioning import VersionRegistry


@dataclass(slots=True)
class AppState:
    """Wires the entry and version registries together for the browser.

    ``entries`` is the browser-enriched view produced by ``discover_entries``;
    ``groups`` and ``code_groups`` are thin accessors over the held registry
    instances so there is a single owner for that data.
    """

    classes: dict[str, ClassInfo]
    entries: dict[str, EntryInfo]
    diagnostics: dict
    spec_options: dict[str, list[str]]
    entry_registry: EntryRegistry
    version_registry: VersionRegistry

    @property
    def groups(self) -> dict[str, GroupRecord]:
        return self.entry_registry.groups

    @property
    def code_groups(self) -> dict[str, list[dict]]:
        return self.version_registry.code_groups


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
    entries, groups, diagnostics = discover_entries(progress=progress)
    entry_registry = EntryRegistry.instance()
    version_registry = VersionRegistry.instance()
    classes = merge_unloaded_classes(
        discover_loaded_classes(), groups, entries=entries, version_registry=version_registry,
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
