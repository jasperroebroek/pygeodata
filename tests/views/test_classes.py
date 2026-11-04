from unittest.mock import MagicMock

import pytest

from pygeodata.catalog.types import CatalogData, CatalogEntry
from pygeodata.dependencies import ClassFacts
from pygeodata.registries.registry import EntryRegistry
from pygeodata.views.classes import current_classes
from tests.fixtures.records import make_record


def _registry(*records: tuple[str, str]) -> EntryRegistry:
    entries = {
        f's{i}': CatalogEntry(
            record=make_record(f's{i}', 'MyLoader', 'Data', source_hash=source_hash, dependency_tree_hash=tree_hash),
            dir_relpath=f's{i}',
            rows=[],
            linked_entries=[],
            files=[],
            created_at=0.0,
            updated_at=0.0,
        )
        for i, (source_hash, tree_hash) in enumerate(records)
    }
    return EntryRegistry(CatalogData(header=MagicMock(), entries=entries, specs={}))


@pytest.mark.parametrize(
    ('records', 'source_stale', 'deps_stale'),
    [
        ((), False, False),
        ((('src', 'tree'),), False, False),
        ((('old', 'old_tree'), ('src', 'tree')), False, False),
        ((('src', 'old_tree'),), False, True),
        ((('old', 'old_tree'),), True, True),
    ],
)
def test_staleness_follows_entry_hashes(records: tuple, source_stale: bool, deps_stale: bool) -> None:
    facts = {'MyLoader': ClassFacts('MyLoader', 'Data', 'src', frozenset(), frozenset())}

    info = current_classes(facts, {'MyLoader': 'tree'}, _registry(*records))['MyLoader']

    assert (info.source_stale, info.deps_stale) == (source_stale, deps_stale)
