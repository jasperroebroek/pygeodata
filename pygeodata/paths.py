import re
from pathlib import Path

from pygeodata.types import SpatialSpec
from pygeodata.utils import transform_to_str


def generate_path(
    spec: SpatialSpec,
    base_dir: str | Path,
    filename: str,
    ext: str,
    name: str | None = None,
    **kwargs,
) -> Path:
    """Function that converts a path of the data to the processed data."""
    base_dir = Path(base_dir)

    p = []
    if name is not None:
        p.append(name)

    for key in sorted(kwargs.keys()):
        p.append(f'{key}={kwargs[key]}')

    return Path(
        base_dir,
        re.sub(r'[^\w\-]', '_', spec.crs.to_string()),
        transform_to_str(spec.transform),
        f'{spec.shape[0]}-{spec.shape[1]}',
        *p,
        f'{filename}.{ext}',
    )
