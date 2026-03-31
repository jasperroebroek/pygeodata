import re
from pathlib import Path

from pygeodata.types import SpatialSpec


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

    shape_str = f'{spec.shape[0]}-{spec.shape[1]}' if spec.shape is not None else 'None'
    t = spec.transform
    transform_str = f'affine_{t.a:.4f}_{t.b:.4f}_{t.c:.4f}_{t.d:.4f}_{t.e:.4f}_{t.f:.4f}' if t is not None else 'None'

    return Path(
        base_dir,
        re.sub(r'[^\w\-]', '_', spec.crs.to_string()),
        transform_str,
        shape_str,
        *p,
        f'{filename}.{ext}',
    )
