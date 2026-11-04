from enum import StrEnum


class JSONKeys(StrEnum):
    CLASS_NAME = 'class_name'
    FORMAT_VERSION = 'format_version'
    OBJECT_TYPE = 'object_type'
    PARAMS = 'params'
    SOURCE_HASH = 'source_hash'
    PROCESSOR_HASH = 'processor_hash'
    INSTANCE_HASH = 'instance_hash'
    STATE_HASH = 'state_hash'
    DEPENDENCY_TREE_HASH = 'dependency_tree_hash'
    CO_OUTPUTS = 'co_output_hashes'
    NODES = 'nodes'
    CALL_EDGES = 'call_edges'
    INHERITANCE_EDGES = 'inheritance_edges'
    REGISTERED_AT = 'registered_at'
    ROOT_CLASS = 'root_class'


class SpecKeys(StrEnum):
    SPEC = 'spec'
    CRS = 'crs'
    TRANSFORM = 'transform'
    RESOLUTION = 'resolution'
    RESOLUTION_DISPLAY = 'resolution_display'
    SHAPE = 'shape'
    BOUNDS = 'bounds'
    BOUNDS_LATLON = 'bounds_latlon'
    BOUNDS_DISPLAY = 'bounds_display'


class CacheFile(StrEnum):
    PARAMS = 'parameters.json'
    META = 'meta.json'
    SPEC = 'spec.json'
    GRAPH = 'graph.pdf'
    LOCK = 'process.lock'


class RegistryDir(StrEnum):
    CODE = 'code'
    SNAPSHOTS = 'snapshots'


class CodeFile(StrEnum):
    SOURCE = 'source.py'
    META = 'source.json'


class SnapshotFile(StrEnum):
    TREE = 'tree.json'
    GRAPH = 'graph.pdf'


class ObjectType(StrEnum):
    DATA = 'Data'
    TABLE = 'Table'
    FIGURE = 'Figure'

    @property
    def default_ext(self) -> str | None:
        return {ObjectType.TABLE: 'html', ObjectType.FIGURE: 'png'}.get(self)


class Staleness(StrEnum):
    CURRENT = 'current'
    STALE = 'stale'
    UNKNOWN = 'unknown'
