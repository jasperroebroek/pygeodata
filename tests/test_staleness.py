from pygeodata.enums import Staleness
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.staleness import class_staleness, dependency_staleness
from pygeodata.tracked_object import live_tree_hash
from tests.fixtures.data import LoaderA


class _Versions:
    """Stand-in for VersionRegistry: versions are ints, verdicts are fixed per dependency hash."""

    def __init__(self, versions: dict[str, int], verdicts: dict[str, Staleness]) -> None:
        self._versions = versions
        self._verdicts = verdicts

    def version_for_dep_hash(self, dep_hash: str) -> int | None:
        return self._versions.get(dep_hash)

    def dependency_staleness(self, dep_hash: str) -> Staleness:
        return self._verdicts[dep_hash]


def test_live_class_current() -> None:
    assert dependency_staleness('LoaderA', LoaderA.get_dependency_tree_hash(), live_tree_hash) is Staleness.CURRENT


def test_live_class_stale() -> None:
    assert dependency_staleness('LoaderA', 'old-hash', live_tree_hash) is Staleness.STALE


def test_missing_hash_is_unknown() -> None:
    assert dependency_staleness('LoaderA', '', live_tree_hash) is Staleness.UNKNOWN


def test_unloaded_class_without_versions_is_unknown() -> None:
    assert dependency_staleness('NotImported', 'h', live_tree_hash) is Staleness.UNKNOWN


def test_unloaded_class_with_empty_registry_is_unknown() -> None:
    assert dependency_staleness('NotImported', 'h', live_tree_hash, VersionRegistry()) is Staleness.UNKNOWN


def test_unloaded_class_uses_versions() -> None:
    versions = _Versions({}, {'h': Staleness.STALE})
    assert dependency_staleness('NotImported', 'h', live_tree_hash, versions) is Staleness.STALE


def test_class_staleness_uses_newest_version() -> None:
    versions = _Versions({'old': 1, 'new': 2}, {'old': Staleness.STALE, 'new': Staleness.CURRENT})
    assert class_staleness('NotImported', ['old', 'new', ''], live_tree_hash, versions) is Staleness.CURRENT


def test_class_staleness_without_versioned_hashes_is_unknown() -> None:
    assert class_staleness('NotImported', ['unversioned', ''], live_tree_hash, _Versions({}, {})) is Staleness.UNKNOWN


def test_source_hashes_serve_as_current() -> None:
    current = {'Edited': 'new'}.get
    assert dependency_staleness('Edited', 'new', current) is Staleness.CURRENT
    assert dependency_staleness('Edited', 'old', current) is Staleness.STALE
    assert dependency_staleness('Unknown', 'old', current) is Staleness.UNKNOWN
