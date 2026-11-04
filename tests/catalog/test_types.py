from pygeodata._version import CATALOG_VERSION
from pygeodata.catalog.types import schema_fingerprint


def test_schema_fingerprint_is_pinned() -> None:
    """A change to the catalog types fails here: bump CATALOG_VERSION, then update the pin."""
    assert (CATALOG_VERSION, schema_fingerprint()) == (
        1,
        '8ecedc0968b656520c9db9c3d8d316c090fbc6266cd4aa7fb7d2f190265961f6',
    )
