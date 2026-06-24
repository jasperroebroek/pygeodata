import functools
import json
import shutil
from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.config import FORMAT_VERSION, JSONKeys, get_config
from pygeodata.paths import CachePathConstructor
from pygeodata.registry_types import EntryRecord
from pygeodata.tracked_object import TrackedObject

ZARR_MARKERS = (
    'zarr.json',  # v3
    '.zgroup',  # v2 group
    '.zarray',  # v2 array
    '.zattrs',  # v2 attributes
    '.zmetadata',  # v2 consolidated metadata
)


def is_zarr_root(path: Path) -> bool:
    """Detect zarr v2 and v3 archive roots by marker files, with fast suffix check."""
    if path.suffix == '.zarr':
        return True
    return any((path / marker).exists() for marker in ZARR_MARKERS)


def read_cache_class_name(hash_path: Path) -> str | None:
    if not hash_path.exists():
        return None
    return EntryRecord.from_file(hash_path).class_name


def format_version_matches(hash_path: Path) -> bool:
    """Return False if the cache entry was written by a different format version."""
    if not hash_path.exists():
        return False
    return EntryRecord.from_file(hash_path).format_version == FORMAT_VERSION


def hash_matches_live(hash_path: Path) -> bool | None:
    if not hash_path.exists():
        return False

    record = EntryRecord.from_file(hash_path)
    class_object = TrackedObject.find_object_class(record.class_name)

    if class_object is None:
        return None

    return record.dependency_tree_hash == class_object.get_dependency_tree_hash()


def handle_invalid(path: Path, dry_run: bool, label: str, class_name: str | None = None) -> None:
    suffix = f' ({class_name})' if class_name else ''
    if dry_run:
        print(f'[dry_run] {label}{suffix}: {path}')
        return
    print(f'[Deleting] {label}{suffix}: {path}')
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def prune_empty_dirs(root: Path) -> None:
    for dirpath, dirs, files in root.walk(top_down=False):
        if dirpath == root:
            continue
        try:
            dirpath.rmdir()
            print(f'[Deleting] Empty dir: {dirpath}')
        except OSError:
            pass


@functools.cache
def _confirm_class_deletion(class_name: str) -> bool:
    """Confirm with the user if a class can be deleted if not found in registry."""
    answer = input(f'{class_name} not found in registry. Delete? [y/N] ')
    return answer.lower() == 'y'


def _delete_manually(hash_path: Path, delete_unregistered: bool) -> bool:
    """Confirm with the user if an entry can be deleted if not found in registry."""
    if not delete_unregistered:
        return False
    class_name = read_cache_class_name(hash_path)
    if class_name is None:
        return True
    if TrackedObject.find_object_class(class_name) is not None:
        return False
    return _confirm_class_deletion(class_name)


def _purge_dir(dirpath: Path, dry_run: bool, delete_unregistered: bool) -> bool:
    """
    Validate a single cache directory. Returns True if the directory was deleted.

    No meta.json  → delete as invalid.
    meta.json     → validate; delete directory if stale or unregistered.
    """
    resolver = CachePathConstructor(dirpath)
    hash_path = resolver.state_hash_path

    if not hash_path.exists():
        label = 'Invalid (empty dir)' if not any(dirpath.iterdir()) else 'Hash missing'
        handle_invalid(dirpath, dry_run=dry_run, label=label)
        return True

    class_name = read_cache_class_name(hash_path)

    if not format_version_matches(hash_path):
        handle_invalid(dirpath, dry_run=dry_run, label='Format version mismatch', class_name=class_name)
        return True

    valid = hash_matches_live(hash_path)

    if valid is None and _delete_manually(hash_path, delete_unregistered):
        valid = False

    if not valid:
        handle_invalid(dirpath, dry_run=dry_run, label='Hash wrong', class_name=class_name)
        return True

    return False


def _purge_cache(dry_run: bool = True, delete_unregistered: bool = True) -> None:
    for family in Artifact.__subclasses__():
        root = family.get_cache_root()
        if not root.exists():
            continue

        for dirpath, dirs, files in root.walk(top_down=True, follow_symlinks=True):
            if dirpath == root:
                continue

            dirs[:] = [d for d in dirs if not is_zarr_root(dirpath / d)]

            if dirs:
                continue
            deleted = _purge_dir(dirpath, dry_run=dry_run, delete_unregistered=delete_unregistered)
            if deleted:
                dirs.clear()

        if not dry_run:
            prune_empty_dirs(root)


def clean_cache(
    dry_run: bool = True,
    delete_unregistered: bool = True,
) -> None:
    return _purge_cache(dry_run=dry_run, delete_unregistered=delete_unregistered)


def rebuild_registry() -> None:
    root = get_config().path_registry
    if root.exists():
        shutil.rmtree(root)

    for tracked_cls in TrackedObject.get_registered_objects():
        tracked_cls.write_registry()


def clean_registry(dry_run: bool = True) -> None:
    """Remove .source/ entries written by a different format version.

    Walks ``code/`` and ``snapshots/`` under the registry root and deletes any
    directory whose metadata JSON does not carry the current FORMAT_VERSION.
    Missing or unreadable metadata is treated as a version mismatch.
    """
    root = get_config().path_registry
    if not root.exists():
        return

    for meta_path in [*root.rglob('source.json'), *root.rglob('tree.json')]:
        try:
            data = json.loads(meta_path.read_text(encoding='utf-8'))
            stale = data.get(JSONKeys.FORMAT_VERSION) != FORMAT_VERSION
        except (OSError, json.JSONDecodeError):
            stale = True
        if stale:
            handle_invalid(meta_path.parent, dry_run=dry_run, label='Format version mismatch')


def clean_source_registry(dry_run: bool = True) -> None:
    """Remove orphaned code snapshots and dependency trees from .source/.

    Policy: KEEP LATEST + REFERENCED.  A snapshot or tree is kept iff:
    - it is referenced by a live cache entry (entry.dep_hash → tree nodes →
      source hashes), OR
    - it is the latest snapshot for its class (so in-progress classes with no
      entry yet are not pruned).

    Everything else is prunable.  Dry-run prints what would be deleted without
    removing anything.
    """
    from pygeodata.registry import EntryRegistry, SourceRegistry, TreeRegistry

    root = get_config().path_registry
    if not root.exists():
        return

    src = SourceRegistry(root)
    trees = TreeRegistry(root)
    entries = EntryRegistry()

    # Build keep-set of source_hashes: latest per class
    keep_source_hashes: set[str] = set()
    for class_name in src.class_names:
        latest = src.get_latest_state_for_class(class_name)
        if latest is not None:
            keep_source_hashes.add(latest.source_hash)

    # Build keep-set of dep_hashes: referenced by live entries
    keep_dep_hashes: set[str] = set()
    for record in entries.records.values():
        if not record.dependency_tree_hash:
            continue
        keep_dep_hashes.add(record.dependency_tree_hash)
        snapshot = trees.get_snapshot_from_hash(record.dependency_tree_hash)
        if snapshot is None:
            continue
        for node in snapshot.nodes.values():
            if isinstance(node, dict) and (h := node.get('hash')):
                keep_source_hashes.add(h)

    # Prune orphan code/ dirs
    code_root = root / 'code'
    if code_root.exists():
        for entry in sorted(code_root.iterdir()):
            if not entry.is_dir():
                continue
            source_hash = entry.name
            if source_hash not in keep_source_hashes:
                handle_invalid(entry, dry_run=dry_run, label='Orphan code snapshot')

    # Prune orphan snapshots/ dirs
    snapshots_root = root / 'snapshots'
    if snapshots_root.exists():
        for entry in sorted(snapshots_root.iterdir()):
            if not entry.is_dir():
                continue
            dep_hash = entry.name
            if dep_hash not in keep_dep_hashes:
                handle_invalid(entry, dry_run=dry_run, label='Orphan tree snapshot')

    if not dry_run:
        prune_empty_dirs(code_root)
        prune_empty_dirs(snapshots_root)
