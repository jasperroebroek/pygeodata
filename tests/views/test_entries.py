import json
from pathlib import Path

import pytest

from pygeodata.config import get_config, set_config
from pygeodata.enums import ObjectType, SpecKeys, Staleness
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.registry import EntryRegistry
from pygeodata.spec import SpatialSpec
from pygeodata.views import entries as views_entries
from pygeodata.views.entries import co_outputs, spec_options, stale_state_hashes
from tests.fixtures.records import make_record, write_meta


def write_entry(
    state_hash: str,
    class_name: str = 'MyLoader',
    object_type: str = 'Data',
    outputs: tuple[str, ...] = ('out.tif',),
    spec: SpatialSpec | None = None,
    **fields: object,
) -> Path:
    entry_dir = get_config().get_cache_root(ObjectType(object_type)) / state_hash
    spec_hash = spec.get_hash() if spec is not None else 'nospec'
    write_meta(entry_dir, make_record(state_hash, class_name, object_type, spec_hash=spec_hash, **fields))
    paths = CachePathConstructor(entry_dir)
    paths.params_path.write_text('{"year": 2020}', encoding='utf-8')
    if spec is not None:
        paths.spec_path.write_text(json.dumps(spec.to_dict()), encoding='utf-8')
    for name in outputs:
        (entry_dir / name).write_bytes(b'data')
    return entry_dir


def test_table_output_is_primary_and_csv_a_sidecar() -> None:
    write_entry('t1', 'MyTable', 'Table', outputs=('my_table.html', 'my_table.csv'))

    entry = EntryRegistry().get_entry('t1')

    assert entry.primary_file.relpath == 'my_table.html'
    assert [f.relpath for f in entry.sidecar_files] == ['my_table.csv']


def test_entry_dir_is_absolute_with_relative_config_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with set_config(path_cache=Path('data_processed'), path_tables=Path('tables'), path_figures=Path('figures')):
        write_entry('a1')
        entry_dir = EntryRegistry().entry_dir('a1')
    assert entry_dir.is_absolute()
    assert entry_dir.exists()


def test_co_outputs_resolve_to_known_entries() -> None:
    write_entry('child')
    write_entry('parent', co_output_hashes=['child', 'unknown'])
    reg = EntryRegistry()

    assert [e.state_hash for e in co_outputs(reg, reg.get_entry('parent'))] == ['child']


def test_staleness_is_evaluated_once_per_class_and_dependency_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    write_entry('a1', dependency_tree_hash='old')
    write_entry('a2', dependency_tree_hash='old')
    write_entry('a3', dependency_tree_hash='new')
    calls: list[tuple[str, str]] = []

    def fake_staleness(class_name: str, dep_hash: str, _vreg: object) -> Staleness:
        calls.append((class_name, dep_hash))
        return Staleness.STALE if dep_hash == 'old' else Staleness.CURRENT

    monkeypatch.setattr(views_entries, 'dependency_staleness', fake_staleness)

    assert stale_state_hashes(EntryRegistry(), version_registry=object()) == {'a1', 'a2'}
    assert sorted(calls) == [('MyLoader', 'new'), ('MyLoader', 'old')]


def test_spec_options(sample_spatial_spec: SpatialSpec) -> None:
    write_entry('a1', spec=sample_spatial_spec)
    write_entry('a2')

    options = spec_options(EntryRegistry())

    assert options[SpecKeys.CRS] == [sample_spatial_spec.crs.to_string()]
    assert options[SpecKeys.RESOLUTION][0] == {'value': 'None', 'label': 'None'}
    assert len(options[SpecKeys.RESOLUTION]) == 2
    assert options[SpecKeys.BOUNDS][0] == {'value': 'None', 'label': 'None'}
