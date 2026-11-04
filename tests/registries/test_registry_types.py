import json
from pathlib import Path

from pygeodata.enums import JSONKeys
from pygeodata.registries.registry_types import CodeState, TreeSnapshot


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
    tree_path.write_text(
        json.dumps(
            {
                JSONKeys.DEPENDENCY_TREE_HASH: 'def',
                JSONKeys.NODES: {'Foo': {'hash': 'abc', JSONKeys.OBJECT_TYPE: 'Data'}},
                JSONKeys.CALL_EDGES: [],
                JSONKeys.INHERITANCE_EDGES: [],
                JSONKeys.ROOT_CLASS: 'Foo',
                JSONKeys.REGISTERED_AT: '2026-01-01T00:00:00+00:00',
            },
        ),
    )

    assert CodeState.from_file(code_path).format_version == 0
    assert TreeSnapshot.from_file(tree_path).format_version == 0
