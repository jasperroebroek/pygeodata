from pygeodata._version import CATALOG_VERSION
from pygeodata.catalog.types import schema_fingerprint


def test_schema_fingerprint_is_pinned() -> None:
    """A change to the catalog types fails here: bump CATALOG_VERSION, then update the pin."""
    assert (CATALOG_VERSION, schema_fingerprint()) == (
        2,
        'c78539dbbff7cf54013ac0695613c283cc25fd1d250dd315cabdddfa341767f0',
    )
