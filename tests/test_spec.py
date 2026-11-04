import json
from pathlib import Path

import msgspec
import pytest
from affine import Affine
from pyproj import CRS

from pygeodata.records import SpecRecord
from pygeodata.spec import SpatialSpec

# Every state hash includes the spec hash: a change here invalidates every cache entry.
PINNED_SPEC_HASHES = {
    'full_4326': (
        SpatialSpec(
            crs=CRS.from_string('EPSG:4326'),
            transform=Affine(0.1, 0.0, -180.0, 0.0, -0.1, 90.0),
            shape=(1800, 3600),
        ),
        '2bf71e8aca2ba90d7851985ba2769ffe3dbc56c4c2731e5609d22565af5ce22e',
    ),
    'full_esri': (
        SpatialSpec(
            crs=CRS.from_string('ESRI:54012'),
            transform=Affine(1000.0, 0.0, -17000000.0, 0.0, -1000.0, 8000000.0),
            shape=(16000, 34000),
        ),
        '65d5821d1ba4df98909930d02798829958053ad0603ed8c4b1f07232cbc056af',
    ),
    'crs_only': (
        SpatialSpec(crs=CRS.from_string('EPSG:3035')),
        'b9d71aed34d2bea8c31ff1ed949099513445e4220c7032565c7128d1dab8733b',
    ),
    'no_crs': (
        SpatialSpec(crs=None),
        '8f63a097dfccbd75276947d8161a403f557efba5e44fc1c2a3d3da0c4a0369ed',
    ),
}


@pytest.mark.parametrize('name', PINNED_SPEC_HASHES)
def test_get_hash_is_pinned(name: str) -> None:
    spec, expected = PINNED_SPEC_HASHES[name]
    assert spec.get_hash() == expected


@pytest.mark.parametrize('name', PINNED_SPEC_HASHES)
def test_record_round_trip_keeps_hash(name: str) -> None:
    spec, expected = PINNED_SPEC_HASHES[name]
    record = spec.to_record()
    decoded = msgspec.json.decode(msgspec.json.encode(record), type=SpecRecord)
    assert decoded == record
    assert SpatialSpec.from_record(decoded).get_hash() == expected


def test_spec_json_decodes_into_record(tmp_path: Path, sample_spatial_spec: SpatialSpec) -> None:
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(sample_spatial_spec.to_dict()))
    assert SpecRecord.from_file(path) == sample_spatial_spec.to_record()
    assert SpatialSpec.from_file(path).get_hash() == sample_spatial_spec.get_hash()
