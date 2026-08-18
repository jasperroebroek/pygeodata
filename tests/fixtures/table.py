from dataclasses import dataclass

from pygeodata.spec import SpatialSpec
from pygeodata.table import Table


@dataclass
class NoParamsTable(Table):
    pass


@dataclass
class SimpleTable(Table):
    a: int = 1


@dataclass
class DummyTable(Table):
    a: int = 1

    def _process(self, spec: SpatialSpec) -> None:
        out = self.get_processed_path(spec)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('<table><tr><td>1</td></tr></table>')
