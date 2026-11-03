from pygeodata.enums import Staleness
from pygeodata.registries.versioning import VersionRegistry
from pygeodata.staleness import class_staleness, dependency_staleness
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
    assert dependency_staleness('LoaderA', LoaderA.get_dependency_tree_hash()) is Staleness.CURRENT


def test_live_class_stale() -> None:
    assert dependency_staleness('LoaderA', 'old-hash') is Staleness.STALE


def test_missing_hash_is_unknown() -> None:
    assert dependency_staleness('LoaderA', None) is Staleness.UNKNOWN


def test_unloaded_class_without_versions_is_unknown() -> None:
    assert dependency_staleness('NotImported', 'h') is Staleness.UNKNOWN


def test_unloaded_class_with_empty_registry_is_unknown() -> None:
    assert dependency_staleness('NotImported', 'h', VersionRegistry()) is Staleness.UNKNOWN


def test_unloaded_class_uses_versions() -> None:
    versions = _Versions({}, {'h': Staleness.STALE})
    assert dependency_staleness('NotImported', 'h', versions) is Staleness.STALE


def test_class_staleness_uses_newest_version() -> None:
    versions = _Versions({'old': 1, 'new': 2}, {'old': Staleness.STALE, 'new': Staleness.CURRENT})
    assert class_staleness('NotImported', ['old', 'new', None], versions) is Staleness.CURRENT


def test_class_staleness_without_versioned_hashes_is_unknown() -> None:
    assert class_staleness('NotImported', ['unversioned', None], _Versions({}, {})) is Staleness.UNKNOWN
