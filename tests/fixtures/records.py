from pathlib import Path

from pygeodata._version import FORMAT_VERSION
from pygeodata.enums import CacheFile, RegistryDir, SnapshotFile
from pygeodata.records import EntryRecord
from pygeodata.registries.registry_types import TreeNode, TreeSnapshot


def make_record(
    state_hash: str = 's1',
    class_name: str = 'SimpleTable',
    object_type: str = 'Table',
    **fields: object,
) -> EntryRecord:
    values = {
        'class_name': class_name,
        'source_hash': 'src1',
        'dependency_tree_hash': 'tree1',
        'instance_hash': 'inst1',
        'params_hash': 'params1',
        'spec_hash': 'spec1',
        'state_hash': state_hash,
        'object_type': object_type,
        'co_output_hashes': [],
        'format_version': FORMAT_VERSION,
    }
    values.update(fields)
    return EntryRecord(**values)


def write_meta(entry_dir: Path, record: EntryRecord) -> Path:
    entry_dir.mkdir(parents=True, exist_ok=True)
    path = entry_dir / CacheFile.META
    record.dump(path)
    return path


def write_tree(registry: Path, dep_hash: str, nodes: dict[str, dict], root_class: str = '') -> None:
    """``nodes`` maps class names to ``{'hash': ..., 'object_type': ...}``; object_type defaults to Data."""
    snapshot = TreeSnapshot(
        dependency_tree_hash=dep_hash,
        nodes={
            name: TreeNode(hash=node['hash'], object_type=node.get('object_type', 'Data'))
            for name, node in nodes.items()
        },
        call_edges=[],
        inheritance_edges=[],
        root_class=root_class,
        registered_at='2026-01-01T00:00:00+00:00',
        format_version=FORMAT_VERSION,
    )
    snapshot_dir = registry / RegistryDir.SNAPSHOTS / dep_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot.dump(snapshot_dir / SnapshotFile.TREE)
