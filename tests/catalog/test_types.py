from pygeodata._version import CATALOG_VERSION
from pygeodata.catalog.types import schema_fingerprint


def test_schema_fingerprint_is_pinned() -> None:
    """A change to the catalog types fails here: bump CATALOG_VERSION, then update the pin."""
    assert (CATALOG_VERSION, schema_fingerprint()) == (
        4,
        '71c7ef91b5dcb7af0ec0fd5bd9c1c99bc165e3d412fd71348a627ddd239be79f',
    )
