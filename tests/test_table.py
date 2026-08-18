import json
from dataclasses import dataclass
from pathlib import Path

from pygeodata.artifact import Artifact
from pygeodata.config import JSONKeys, get_config
from pygeodata.data import Data
from pygeodata.registries.registry import EntryRegistry
from pygeodata.spec import SpatialSpec
from pygeodata.table import Table
from tests.fixtures.table import DummyTable, SimpleTable


def test_table_is_its_own_object_type() -> None:
    assert Table.object_type is Table


def test_table_subclass_inherits_object_type() -> None:
    @dataclass
    class MyTable(Table):
        pass

    assert MyTable.object_type is Table
    assert MyTable.object_type.get_class_name() == 'Table'


def test_table_subclasses_artifact_not_data() -> None:
    assert issubclass(Table, Artifact)
    assert not issubclass(Table, Data)


def test_table_has_no_load_or_driver() -> None:
    assert not hasattr(Table, 'load')
    assert not hasattr(Table, 'driver')


def test_table_default_ext() -> None:
    assert Table().get_ext() == 'html'


def test_table_get_cache_root_matches_config(tmp_path: Path) -> None:
    assert Table.get_cache_root() == tmp_path / 'tables'


def test_table_cache_root_is_not_data_or_figures_root() -> None:
    cfg = get_config()
    assert Table.get_cache_root() == cfg.path_tables
    assert Table.get_cache_root() != cfg.path_cache
    assert Table.get_cache_root() != cfg.path_figures


def test_dummy_table_process_writes_html(sample_spatial_spec: SpatialSpec) -> None:
    table = DummyTable()
    table.process(sample_spatial_spec)
    out = table.get_processed_path(sample_spatial_spec)
    assert out.exists()
    assert out.suffix == '.html'
    assert '<table>' in out.read_text()


def test_dummy_table_entry_lands_under_tables_root(
    tmp_path: Path,
    sample_spatial_spec: SpatialSpec,
) -> None:
    table = DummyTable()
    table.process(sample_spatial_spec)
    out = table.get_processed_path(sample_spatial_spec)
    assert out.is_relative_to(tmp_path / 'tables')


def test_dummy_table_discovered_by_entry_registry(sample_spatial_spec: SpatialSpec) -> None:
    table = DummyTable()
    table.process(sample_spatial_spec)
    registry = EntryRegistry()
    class_names = {rec.class_name for rec in registry.records.values()}
    assert 'DummyTable' in class_names


def test_meta_json_records_table_object_type(sample_spatial_spec: SpatialSpec) -> None:
    table = DummyTable()
    table.process(sample_spatial_spec)
    hash_path = table.resolve_cache_paths(sample_spatial_spec).state_hash_path
    meta = json.loads(hash_path.read_text())
    assert meta[JSONKeys.OBJECT_TYPE] == 'Table'


def test_simple_table_processed_path_contains_hash(sample_spatial_spec: SpatialSpec) -> None:
    path = SimpleTable().get_processed_path(sample_spatial_spec)
    hash_part = path.parent.name
    assert len(hash_part) == 64
    assert all(c in '0123456789abcdef' for c in hash_part)
