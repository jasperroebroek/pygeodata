from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('pygeodata')
except PackageNotFoundError:
    __version__ = '0+unknown'

# Bump only when hash semantics change: it invalidates every cache entry.
FORMAT_VERSION = 2
CATALOG_VERSION = 3
