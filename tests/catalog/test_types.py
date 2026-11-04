from pygeodata._version import CATALOG_VERSION
from pygeodata.catalog.types import schema_fingerprint


def test_schema_fingerprint_is_pinned() -> None:
    """A change to the catalog types fails here: bump CATALOG_VERSION, then update the pin."""
    assert (CATALOG_VERSION, schema_fingerprint()) == (
        3,
        '83f58e127b1041a5e5f795916b751cf0c47a9c522d735e592766ac30232cbb86',
    )
