import json
import shutil
from pathlib import Path

from pygeodata.config import get_config
from pygeodata.loader import DataLoader


def find_leaf_folders_with_name_at_level(
    root: str | Path,
    name_at_level: str,
    level: int,
) -> list[Path]:
    """
    Find all leaf subfolders (no further subfolders) under `root`, where the folder at position `level`
    in the hierarchy equals `name_at_level`.

    Level 1 = direct children of root
    Level 2 = grandchildren, etc.
    """
    root = Path(root)
    results = []

    def _recurse(path: Path) -> None:
        subdirs = [p for p in path.iterdir() if p.is_dir()]

        if not subdirs:
            parts = path.relative_to(root).parts
            if len(parts) >= level and parts[level - 1] == name_at_level:
                results.append(path)
            return

        for subdir in subdirs:
            _recurse(subdir)

    _recurse(root)
    return results


def path_matches_hash(path: Path, source_hierarchy_hash: str) -> bool:
    if not path.exists():
        return False

    with Path.open(path, encoding='utf-8') as f:
        saved_state = json.load(f)

    return saved_state.get('source_hierarchy_hash', None) == source_hierarchy_hash


def purge_cache_invalid(loader: type[DataLoader] | DataLoader, root: Path, dry_run: bool = True) -> None:
    source_hierarchy_hash = loader.get_source_hierarchy_hash()

    leaves = find_leaf_folders_with_name_at_level(root, loader.get_class_name(), 3)

    candidates = set()
    for leaf in leaves:
        path = leaf
        while path != root:
            candidates.add(path)
            if path.parts[-1] == loader.get_class_name():
                break
            path = path.parent

    for folder in sorted(candidates, key=lambda p: len(p.parts), reverse=True):
        if not folder.exists():
            continue

        hash_path = folder / f'{loader.get_name()}.hash.json'
        match = path_matches_hash(hash_path, source_hierarchy_hash)

        if match:
            continue

        if not hash_path.exists() and any(folder.iterdir()):
            continue

        if dry_run:
            if not hash_path.exists():
                print(f'[dry_run] Hash missing: {folder}')
            else:
                print(f'[dry run] Hash wrong: {folder}')
        else:
            print(f'Deleting: {folder}')
            shutil.rmtree(folder)


def clean_cache(
    loader: type[DataLoader] | DataLoader | None = None,
    root: str | Path | None = None,
    dry_run: bool = True,
) -> None:
    if root is None:
        root = get_config().path_data_processed

    nodes = DataLoader.__subclasses__() if loader is None else loader.get_dependency_graph()['nodes']

    for loader_cls in nodes:
        purge_cache_invalid(loader_cls, root, dry_run)
