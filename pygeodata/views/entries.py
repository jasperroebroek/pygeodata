"""Display assembly over the entry catalog: specs, staleness and co-outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pygeodata.enums import SpecKeys, Staleness
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.staleness import dependency_staleness
from pygeodata.views.types import spec_info

if TYPE_CHECKING:
    from collections.abc import Callable

    from pygeodata.catalog.types import CatalogEntry
    from pygeodata.registries.registry import EntryRegistry


def stale_state_hashes(
    registry: EntryRegistry,
    current: Callable[[str], str | None],
    version_registry: VersionRegistry | None = None,
) -> set[str]:
    """Entries whose dependency tree has changed since they were computed; see :mod:`pygeodata.staleness`."""
    vreg = version_registry if version_registry is not None else VersionRegistry()
    staleness: dict[tuple[str, str], Staleness] = {}
    stale: set[str] = set()
    for state_hash, entry in registry.entries.items():
        key = (entry.class_name, entry.record.dependency_tree_hash)
        if key not in staleness:
            staleness[key] = dependency_staleness(*key, current, vreg)
        if staleness[key] is Staleness.STALE:
            stale.add(state_hash)
    return stale


def co_outputs(registry: EntryRegistry, entry: CatalogEntry) -> list[CatalogEntry]:
    return [co for h in entry.record.co_output_hashes if (co := registry.get_entry(h)) is not None]


def _none_first(keys: set[str]) -> list[str]:
    return sorted(keys, key=lambda k: (k != 'None', k))


def spec_options(registry: EntryRegistry) -> dict[str, list]:
    """Distinct spec values across the entries, for filter choices; entries lacking a resolution or
    bounds contribute a 'None' option.
    """
    specs = [spec_info(registry.specs.get(h)) for h in {e.record.spec_hash for e in registry.entries.values()}]
    resolutions = {spec.resolution_display or 'None' for spec in specs}
    bounds = {spec.bounds or 'None': spec.bounds_display or 'None' for spec in specs}
    return {
        SpecKeys.CRS: sorted({spec.crs for spec in specs if spec.crs}),
        SpecKeys.RESOLUTION: [{'value': k, 'label': k} for k in _none_first(resolutions)],
        SpecKeys.SHAPE: sorted({spec.shape for spec in specs if spec.shape}),
        SpecKeys.BOUNDS: [{'value': k, 'label': bounds[k]} for k in _none_first(set(bounds))],
    }
