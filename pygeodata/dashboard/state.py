from dataclasses import dataclass

from .class_catalog import discover_loaded_classes, merge_unloaded_classes
from .entry_catalog import discover_entries
from .models import ClassInfo, EntryInfo, GroupInfo


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
        'crs':        sorted({entry.spec.crs        for entry in entries.values() if entry.spec.crs}),
        'resolution': sorted({entry.spec.resolution for entry in entries.values() if entry.spec.resolution}),
        'shape':      sorted({entry.spec.shape      for entry in entries.values() if entry.spec.shape}),
        'bounds':     sorted({
            str(list(entry.spec.bounds_latlon))
            for entry in entries.values()
            if entry.spec.bounds_latlon
        }),
    }
    return AppState(classes=classes, entries=entries, groups=groups, diagnostics=diagnostics, spec_options=spec_options)
