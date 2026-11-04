import json
from pathlib import Path

import msgspec
import pytest

from pygeodata._version import FORMAT_VERSION
from pygeodata.records import EntryRecord
from tests.fixtures.records import make_record, write_meta


def test_entry_record_is_current_format() -> None:
    assert make_record(format_version=FORMAT_VERSION).is_current_format
    assert not make_record(format_version=FORMAT_VERSION - 1).is_current_format


def test_entry_record_round_trip(tmp_path: Path) -> None:
    record = make_record(co_output_hashes=['co1'])
    assert EntryRecord.from_file(write_meta(tmp_path, record)) == record


def test_entry_record_without_format_version_is_unversioned(tmp_path: Path) -> None:
    data = msgspec.to_builtins(make_record())
    del data['format_version']
    path = tmp_path / 'meta.json'
    path.write_text(json.dumps(data))
    assert EntryRecord.from_file(path).format_version == 0


def test_entry_record_missing_field_is_rejected(tmp_path: Path) -> None:
    data = msgspec.to_builtins(make_record())
    del data['state_hash']
    path = tmp_path / 'meta.json'
    path.write_text(json.dumps(data))
    with pytest.raises(msgspec.ValidationError, match='state_hash'):
        EntryRecord.from_file(path)


def test_entry_record_without_ext_falls_back_to_family_default(tmp_path: Path) -> None:
    data = msgspec.to_builtins(make_record(object_type='Figure'))
    del data['ext']
    path = tmp_path / 'meta.json'
    path.write_text(json.dumps(data))

    record = EntryRecord.from_file(path)
    assert record.ext is None
    assert record.output_ext == 'png'


def test_entry_record_output_ext_prefers_recorded_ext() -> None:
    assert make_record(object_type='Figure', ext='svg').output_ext == 'svg'
    assert make_record(object_type='Data').output_ext is None
