import functools
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

from filelock import FileLock, Timeout

from pygeodata.catalog.store import reconcile
from pygeodata.config import get_config
from pygeodata.enums import JSONKeys, Staleness
from pygeodata.paths import CachePathConstructor, CachePathResolver, RegistryResolver
from pygeodata.registries.registry import EntryRegistry, SourceRegistry, TreeRegistry
from pygeodata.registries.registry_types import EntryRecord, is_current_format
from pygeodata.staleness import dependency_staleness
from pygeodata.tracked_object import TrackedObject


def read_cache_class_name(hash_path: Path) -> str | None:
    if not hash_path.exists():
        return None
    return EntryRecord.from_file(hash_path).class_name


def _iter_class_params(value: object) -> Iterator[dict]:
    """Yield every serialised *class*-valued param nested anywhere in ``value``."""
    if isinstance(value, dict):
        if JSONKeys.CLASS_NAME in value and JSONKeys.DEPENDENCY_TREE_HASH in value:
            yield value
        for item in value.values():
            yield from _iter_class_params(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_class_params(item)


def class_params_match_live(record: EntryRecord) -> bool | None:
    """Compare the recorded dependency tree hash of each class-valued param against the live one."""
    params_path = record.params_path
    if params_path is None or not params_path.exists():
        return False

    try:
        params = json.loads(params_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False

    for stored in _iter_class_params(params):
        live_cls = TrackedObject.find_object_class(stored[JSONKeys.CLASS_NAME])
        if live_cls is None:
            return None
        if live_cls.get_dependency_tree_hash() != stored[JSONKeys.DEPENDENCY_TREE_HASH]:
            return False

    return True


def hash_matches_live(hash_path: Path, check_class_params: bool = False) -> bool | None:
    """Validate a cache entry's recorded hashes against the live classes.

    With ``check_class_params``, additionally validate every ``TrackedObject`` subclass held
    as a *parameter* — see :func:`class_params_match_live`. Off by default: it reads a second
    file per entry and reports as stale any entry whose ``parameters.json`` is absent.
    """
    if not hash_path.exists():
        return False

    record = EntryRecord.from_file(hash_path)
    if not record.dependency_tree_hash:
        return False
    staleness = dependency_staleness(record.class_name, record.dependency_tree_hash)
    if staleness is Staleness.UNKNOWN:
        return None
    if staleness is Staleness.STALE:
        return False

    if not check_class_params:
        return True

    return class_params_match_live(record)


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
    for dirpath_str, dirs, files in os.walk(root, topdown=False):
        dirpath = Path(dirpath_str)
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


def _purge_dir(dirpath: Path, dry_run: bool, delete_unregistered: bool, check_class_params: bool = False) -> bool:
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

    if not is_current_format(hash_path):
        handle_invalid(dirpath, dry_run=dry_run, label='Format version mismatch', class_name=class_name)
        return True

    valid = hash_matches_live(hash_path, check_class_params=check_class_params)

    if valid is None and _delete_manually(hash_path, delete_unregistered):
        valid = False

    if not valid:
        handle_invalid(dirpath, dry_run=dry_run, label='Hash wrong', class_name=class_name)
        return True

    return False


def clean_cache(
    dry_run: bool = True,
    delete_unregistered: bool = True,
    check_class_params: bool = False,
) -> None:
    """Remove stale or invalid cache entries.

    Parameters
    ----------
    dry_run : bool
        Report what would be deleted without removing anything.
    delete_unregistered : bool
        Prompt to delete entries whose class is no longer in the registry.
    check_class_params : bool
        Also validate ``TrackedObject`` subclasses held as parameters. These never reach the
        entry's dependency tree, so without this a class-valued param that changed is
        reported as current even though the runtime correctly recomputes it. Entries with no
        ``parameters.json`` are reported as stale under this check.
    """
    roots = get_config().cache_roots
    for entry_dir in CachePathResolver(roots).iter_entry_dirs():
        _purge_dir(
            entry_dir,
            dry_run=dry_run,
            delete_unregistered=delete_unregistered,
            check_class_params=check_class_params,
        )

    if not dry_run:
        for root in roots:
            if root.exists():
                prune_empty_dirs(root)
        prune_catalog()


def clean_lock_files(dry_run: bool = True) -> None:
    """Remove ``process.lock`` files stranded by terminated runs.

    A lock currently held by a running process is skipped; a dry run cannot
    detect held locks and lists every lock file found.
    """
    for entry_dir in CachePathResolver().iter_entry_dirs():
        lock_path = CachePathConstructor(entry_dir).lock_path
        if not lock_path.exists():
            continue
        if dry_run:
            print(f'[dry_run] process.lock: {lock_path}')
            continue

        lock = FileLock(lock_path, timeout=0)
        try:
            lock.acquire()
        except Timeout:
            print(f'[Skipping] process.lock in use: {lock_path}')
            continue
        lock_path.unlink(missing_ok=True)
        lock.release()
        print(f'[Deleting] process.lock: {lock_path}')


def prune_catalog() -> None:
    deleted = reconcile()
    if deleted:
        print(f'[Catalog] Pruned {deleted} row(s)')


def rebuild_registry() -> None:
    root = get_config().path_registry
    if root.exists():
        shutil.rmtree(root)

    registry = SourceRegistry()
    for tracked_cls in TrackedObject.get_registered_objects():
        tracked_cls.write_registry(registry)


def clean_registry(dry_run: bool = True) -> None:
    """Remove .source/ entries written by a different format version.

    Walks ``code/`` and ``snapshots/`` under the registry root and deletes any
    directory whose metadata JSON does not carry the current FORMAT_VERSION.
    Missing or unreadable metadata is treated as a version mismatch.
    """
    resolver = RegistryResolver()
    for meta_path in [*resolver.glob_source_paths(), *resolver.glob_tree_paths()]:
        if not is_current_format(meta_path):
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
    code_root = RegistryResolver(root).code_dir
    if code_root.exists():
        for entry in sorted(code_root.iterdir()):
            if not entry.is_dir():
                continue
            source_hash = entry.name
            if source_hash not in keep_source_hashes:
                handle_invalid(entry, dry_run=dry_run, label='Orphan code snapshot')

    # Prune orphan snapshots/ dirs
    snapshots_root = RegistryResolver(root).snapshots_dir
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
