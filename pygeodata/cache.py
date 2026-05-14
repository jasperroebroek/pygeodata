import json
from pathlib import Path

from pygeodata.config import get_config
from pygeodata.loader import DataLoader


def path_matches_hash(path: Path, source_hierarchy_hash: str) -> bool:
    if not path.exists():
        return False

    with Path.open(path, encoding='utf-8') as f:
        saved_state = json.load(f)

    return saved_state.get('source_hierarchy_hash', None) == source_hierarchy_hash


def purge_cache_invalid(loader: type[DataLoader] | DataLoader, root: Path, dry_run: bool = True) -> None:
    source_hierarchy_hash = loader.get_source_hierarchy_hash()

    paths = root.glob(f'**/{loader.get_class_name()}')
    for path in paths:
        if '_source' in path.parts:
            continue

        for dirpath, dirs, files in path.walk(top_down=False, follow_symlinks=True):
            for file in files:
                path_file = dirpath / file
                hash_filename = f'{file.split(".")[0]}.hash.json'
                path_hash = dirpath / hash_filename
                match = path_matches_hash(path_hash, source_hierarchy_hash)

                if match:
                    continue

                if dry_run:
                    if not path_hash.exists():
                        print(f'[dry_run] Hash missing: {path_file}')
                    else:
                        print(f'[dry run] Hash wrong: {path_file}')
                else:
                    print(f'Deleting: {path_file}')
                    path_file.unlink()

            if dry_run:
                continue

            for dir in dirs:
                path_dir = dirpath / dir
                if next((path_dir).iterdir(), None) is None:
                    print(f'Removing {path_dir}')
                    path_dir.rmdir()

            if next(dirpath.iterdir(), None) is None:
                print(f'Removing {dirpath}')
                dirpath.rmdir()


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
