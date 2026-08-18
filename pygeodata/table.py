from pathlib import Path
from typing import ClassVar

from pygeodata.artifact import Artifact
from pygeodata.config import get_config


class Table(Artifact):
    """
    Base class for rendered-table outputs in pygeodata.

    A :class:`Table` subclass's ``_process`` renders a :class:`pandas.DataFrame` — typically
    loaded from a separate :class:`~pygeodata.data.Data` class — to an HTML file.

    .. code-block:: python

        class SummaryTable(Table):
            def _process(self, spec):
                df = SomeDataLoader().load(spec)
                self.ensure_processed_path(spec).write_text(df.head(20).to_html())

    Structurally identical to :class:`~pygeodata.figure.Figure` (fixed ``ext``, own colour, own
    cache root, no ``load()``) rather than to :class:`~pygeodata.data.Data`. It forms its own
    ``object_type`` family so tables are filterable and colour-coded separately in the CLI and
    the registry browser.

    Class Attributes
    ----------------
    _sort_params : tuple[str]
        Inherited from :class:`~pygeodata.tracked_object.TrackedObject`. Parameter names
        whose list/tuple values should be sorted before hashing and path generation,
        ensuring order-independence.

    Notes
    -----
    Subclasses that define ``__init__`` parameters should store them as instance
    attributes. :meth:`~pygeodata.tracked_object.TrackedObject.get_params` discovers
    parameters by inspecting ``vars(self)``. Recommended to use dataclasses for this.

    Subclasses must not be defined interactively (e.g. in a REPL or Jupyter notebook), as the
    caching relies on :func:`inspect.getsource` for AST parsing.
    """

    color: ClassVar[str] = '#c9d8c5'
    ext: ClassVar[str] = 'html'

    @classmethod
    def get_cache_root(cls) -> Path:
        return get_config().path_tables
