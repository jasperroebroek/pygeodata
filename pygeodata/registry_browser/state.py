from dataclasses import dataclass

from pygeodata.types import SpecKeys

from pygeodata.registry_browser.class_catalog import discover_loaded_classes, merge_unloaded_classes
from pygeodata.registry_browser.entry_catalog import discover_entries
from pygeodata.registry_browser.models import ClassInfo, EntryInfo, GroupInfo


@dataclass(slots=True)
class AppState:
    classes: dict[str, ClassInfo]
    entries: dict[str, EntryInfo]
    groups: dict[str, GroupInfo]
    diagnostics: dict
    spec_options: dict[str, list[str]]


def build_state(progress: dict | None = None) -> AppState:
    entries, groups, diagnostics = discover_entries(progress=progress)
    classes = merge_unloaded_classes(discover_loaded_classes(), groups)
    spec_options = {
        SpecKeys.CRS: sorted({entry.spec.crs for entry in entries.values() if entry.spec.crs}),
        SpecKeys.RESOLUTION: sorted({entry.spec.resolution for entry in entries.values() if entry.spec.resolution}),
        SpecKeys.SHAPE: sorted({entry.spec.shape for entry in entries.values() if entry.spec.shape}),
        SpecKeys.BOUNDS: sorted(
            {str(list(entry.spec.bounds_latlon)) for entry in entries.values() if entry.spec.bounds_latlon},
        ),
    }
    return AppState(classes=classes, entries=entries, groups=groups, diagnostics=diagnostics, spec_options=spec_options)
