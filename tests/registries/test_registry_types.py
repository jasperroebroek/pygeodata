import json
from pathlib import Path

import pytest

from pygeodata.config import FORMAT_VERSION
from pygeodata.enums import JSONKeys
from pygeodata.registries.registry_types import CodeState, EntryRecord, TreeSnapshot, is_current_format
from pygeodata.spec import SpatialSpec
from tests.fixtures.data import SimpleLoader


def test_is_current_format_true(sample_spatial_spec: SpatialSpec) -> None:
    loader = SimpleLoader()
    loader.write_entry(sample_spatial_spec)
    hash_path = loader.resolve_cache_paths(sample_spatial_spec).state_hash_path
    assert is_current_format(hash_path) is True


@pytest.mark.parametrize(
    'content',
    [
        json.dumps({JSONKeys.CLASS_NAME: 'SimpleLoader'}),
        json.dumps({JSONKeys.FORMAT_VERSION: FORMAT_VERSION + 1}),
        json.dumps({JSONKeys.FORMAT_VERSION: str(FORMAT_VERSION)}),
        json.dumps([FORMAT_VERSION]),
        '{not json',
    ],
)
def test_is_current_format_false(tmp_path: Path, content: str) -> None:
    path = tmp_path / 'meta.json'
    path.write_text(content)
    assert is_current_format(path) is False


def test_is_current_format_false_when_file_missing(tmp_path: Path) -> None:
    assert is_current_format(tmp_path / 'meta.json') is False


def test_entry_record_is_current_format() -> None:
    assert EntryRecord(format_version=FORMAT_VERSION).is_current_format
    assert not EntryRecord(format_version=FORMAT_VERSION - 1).is_current_format
    assert not EntryRecord().is_current_format


def test_missing_format_version_reads_as_unversioned(tmp_path: Path) -> None:
    code_path = tmp_path / 'source.json'
    code_path.write_text(
        json.dumps(
            {
                JSONKeys.SOURCE_HASH: 'abc',
                JSONKeys.CLASS_NAME: 'Foo',
                JSONKeys.OBJECT_TYPE: 'Data',
                JSONKeys.REGISTERED_AT: '2026-01-01T00:00:00+00:00',
            },
        ),
    )
    tree_path = tmp_path / 'tree.json'
    tree_path.write_text(json.dumps({JSONKeys.DEPENDENCY_TREE_HASH: 'def'}))

    assert CodeState.from_file(code_path).format_version == 0
    assert TreeSnapshot.from_file(tree_path).format_version == 0
