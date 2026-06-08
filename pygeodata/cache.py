import functools
import json
import shutil
from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.config import JSONKeys, get_config
from pygeodata.paths import CachePathResolver
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
    with hash_path.open(encoding='utf-8') as f:
        return json.load(f).get(JSONKeys.CLASS_NAME, None)


def hash_matches_live(hash_path: Path) -> bool | None:
    if not hash_path.exists():
        return False

    class_name = read_cache_class_name(hash_path)
    class_object = TrackedObject.find_object_class(class_name)

    if class_object is None:
        return None

    with hash_path.open(encoding='utf-8') as f:
        saved_state = json.load(f)

    return saved_state.get(JSONKeys.DEPENDENCY_TREE_HASH, None) == class_object.get_dependency_tree_hash()


def handle_invalid(path: Path, dry_run: bool, hash_path: Path | None = None) -> None:
    if hash_path is None:
        label = 'Invalid'
    elif not hash_path.exists():
        label = 'Hash missing'
    else:
        label = 'Hash wrong'

    if dry_run:
        print(f'[dry_run] {label}: {path}')
        return

    print(f'[Deleting] {label}: {path}')
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


def _find_hash_files(dirpath: Path, files: list[str]) -> list[Path]:
    return [dirpath / f for f in files if f.startswith('.') and f.endswith('.hash.json')]


def _stem_from_hash_file(hash_path: Path) -> str:
    return hash_path.name.removeprefix('.').removesuffix('.hash.json')


def _purge_dir(dirpath: Path, hash_files: list[Path], dry_run: bool, delete_unregistered: bool) -> bool:
    """
    Validate a single cache directory. Returns True if the directory was deleted.

    No hash files   → delete as invalid.
    One hash file   → validate; delete directory if stale or unregistered.
    Two+ hash files → ask the user which entry to keep; delete the rest (or all).
    """
    if not hash_files:
        data_files = [dirpath / f for f in dirpath.iterdir() if not f.name.startswith('.')]
        expected_hash = CachePathResolver.from_path(data_files[0]).state_hash_path if len(data_files) == 1 else None
        handle_invalid(dirpath, dry_run=dry_run, hash_path=expected_hash)
        return True

    if len(hash_files) > 1:
        print(f'\nMultiple cache entries found in {dirpath}:')
        for i, hp in enumerate(hash_files):
            print(f'  [{i}] {hp.name}  (class: {read_cache_class_name(hp)})')
        raw = input('Enter index to keep (or blank to delete all): ').strip()
        if raw.isdigit() and int(raw) < len(hash_files):
            keep = hash_files[int(raw)]
            for hp in hash_files:
                if hp == keep:
                    continue
                stem = _stem_from_hash_file(hp)
                zarr_candidate = dirpath / f'{stem}.zarr'
                data_path = zarr_candidate if zarr_candidate.exists() else dirpath / stem
                handle_invalid(data_path, dry_run=dry_run, hash_path=hp)
            return False
        handle_invalid(dirpath, dry_run=dry_run)
        return True

    hash_path = hash_files[0]
    valid = hash_matches_live(hash_path)

    if valid is None and _delete_manually(hash_path, delete_unregistered):
        valid = False

    if not valid:
        handle_invalid(dirpath, dry_run=dry_run, hash_path=hash_path)
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

            hash_files = _find_hash_files(dirpath, files)
            if dirs and not hash_files:
                continue
            deleted = _purge_dir(dirpath, hash_files, dry_run=dry_run, delete_unregistered=delete_unregistered)
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
