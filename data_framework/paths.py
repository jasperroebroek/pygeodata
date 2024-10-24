from pathlib import Path
from typing import Optional

from affine import Affine
from rasterio import CRS

from data_framework.types import Shape
from data_framework.utils import transform_to_str


class PathContext:
    _instance: Optional['PathContext'] = None
    _path_data_processed: Path = Path("data_processed")

    @property
    def path_data_processed(self) -> Path:
        self._path_data_processed.mkdir(exist_ok=True, parents=True)
        return self._path_data_processed

    @path_data_processed.setter
    def path_data_processed(self, path: Path) -> None:
        self._path_data_processed = path


pc = PathContext()


def generate_path(crs: CRS, transform: Affine, shape: Shape, name: Optional[str], filename: str, **kwargs) -> Path:
    """Function that converts a path of the data to the reprojected data.
    NOTE: This is not always guaranteed to work"""
    p = []
    if name is not None:
        p.append(name)

    for key in sorted(kwargs.keys()):
        p.append(f"{key}={kwargs[key]}")

    return Path(pc.path_data_processed,
                f"{crs.to_string().replace(':', '_')}",
                transform_to_str(transform),
                f"{shape[0]}-{shape[1]}",
                *p,
                f"{filename}.tif")
