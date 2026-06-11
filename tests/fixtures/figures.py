from dataclasses import dataclass

from pygeodata.figure import Figure
from pygeodata.spec import SpatialSpec


@dataclass
class NoParamsFigure(Figure):
    pass


@dataclass
class SimpleFigure(Figure):
    a: int = 1


@dataclass
class TwoParamFigure(Figure):
    a: int
    b: str


@dataclass
class DummyFigure(Figure):
    a: int

    def _process(self, spec: SpatialSpec) -> None:
        out = self.get_processed_path(spec)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch()
