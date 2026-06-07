import json
import shutil
from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.config import JSONKeys, get_config
from pygeodata.paths import ARTIFACT_SUFFIXES, CachePathResolver
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


def path_matches_hash(path: Path, dependency_tree_hash: str) -> bool:
    if not path.exists():
        return False

    with path.open(encoding='utf-8') as f:
        saved_state = json.load(f)

    return saved_state.get(JSONKeys.DEPENDENCY_TREE_HASH, None) == dependency_tree_hash


def read_cache_class_name(path: Path) -> str | None:
    hash_path = CachePathResolver.from_path(path).state_hash_path
    if not hash_path.exists():
        return None
    with hash_path.open(encoding='utf-8') as f:
        return json.load(f).get(JSONKeys.CLASS_NAME, None)


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


def _is_system_path(path: Path) -> bool:
    cfg = get_config()
    return (
        path.name in cfg.removable_system_files
        or path.name.endswith(cfg.removable_system_suffixes)
        or (path.name.startswith('.') and not path.name.endswith(ARTIFACT_SUFFIXES))
    )


def _contains_registered_hash(path: Path, registered_names: set[str]) -> bool:
    """Return True if the directory contains a .hash.json belonging to a registered class."""
    for hash_file in path.rglob('*.hash.json'):
        with hash_file.open(encoding='utf-8') as f:
            if json.load(f).get(JSONKeys.CLASS_NAME) in registered_names:
                return True
    return False


def purge_unregistered_cache(dry_run: bool = True) -> None:
    registered: list[type[Artifact]] = Artifact.get_registered_objects()
    registered_names = {a.get_class_name() for a in registered}

    for family in Artifact.__subclasses__():
        root = family.get_cache_root()

        for path in root.glob(family.get_general_cache_pattern()):
            if _is_system_path(path):
                if not dry_run:
                    handle_invalid(path, dry_run)
                continue

            if any(artifact.matches_cache_path(path) for artifact in registered):
                continue

            if path.is_dir() and _contains_registered_hash(path, registered_names):
                continue

            if dry_run:
                print(f'[dry_run] {path}')
                continue

            answer = input(f'Delete {path}? [y/N] ')
            if answer.lower() == 'y':
                handle_invalid(path, dry_run)
            else:
                print(f'Skipping {path}')

        if not dry_run:
            print('Pruning empty directories')
            prune_empty_dirs(root)


def clean_cache(
    loader: type[Artifact] | Artifact | None = None,
    dry_run: bool = True,
) -> None:
    artifacts: set[type[Artifact]]
    if loader is None:
        artifacts = Artifact.get_registered_objects()
    else:
        artifacts = loader.get_all_dependencies()
        artifacts.add(loader)

    for artifact in artifacts:
        artifact.purge_cls_cache(dry_run=dry_run)

    if loader is None and not dry_run:
        purge_unregistered_cache(dry_run=dry_run)


def rebuild_registry() -> None:
    root = get_config().path_registry
    if root.exists():
        shutil.rmtree(root)

    for tracked_cls in TrackedObject.get_registered_objects():
        tracked_cls.write_registry()
