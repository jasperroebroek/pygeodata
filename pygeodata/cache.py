import json
from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.tracked_object import TrackedObject


def path_matches_hash(path: Path, source_hierarchy_hash: str) -> bool:
    if not path.exists():
        return False

    with Path.open(path, encoding='utf-8') as f:
        saved_state = json.load(f)

    return saved_state.get('source_hierarchy_hash', None) == source_hierarchy_hash


def purge_cache_invalid(artifact: type[Artifact] | Artifact, dry_run: bool = True) -> None:
    source_hierarchy_hash = artifact.get_source_hierarchy_hash()

    root = artifact.get_processed_base_dir()
    dir_pattern = artifact.get_processed_dir_pattern()

    pattern = str(Path(*dir_pattern.parts[len(root.parts) :]))
    paths = root.rglob(pattern)

    for path in paths:
        for dirpath, dirs, files in path.walk(top_down=False, follow_symlinks=True):
            for file in files:
                filename = file.removeprefix('.')
                stem = filename.split('.')[0]
                hash_filename = f'.{stem}.hash.json'
                path_hash = dirpath / hash_filename

                match = path_matches_hash(path_hash, source_hierarchy_hash)

                if match:
                    continue

                path_file = dirpath / file
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
    loader: type[Artifact] | Artifact | None = None,
    dry_run: bool = True,
) -> None:
    nodes = Artifact._registry.values() if loader is None else loader.get_dependency_graph()['nodes']

    for loader_cls in nodes:
        if loader_cls.object_type == TrackedObject:
            continue
        purge_cache_invalid(loader_cls, dry_run)
