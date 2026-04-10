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

    if spec.shape is None or spec.transform is None:
        geo_str = ['vector']
    else:
        shape_str = f'{spec.shape[0]}-{spec.shape[1]}'
        t = spec.transform
        transform_str = f'affine_{t.a:.4f}_{t.b:.4f}_{t.c:.4f}_{t.d:.4f}_{t.e:.4f}_{t.f:.4f}'
        geo_str = [transform_str, shape_str]

    return Path(
        base_dir,
        re.sub(r'[^\w\-]', '_', spec.crs.to_string()),
        *geo_str,
        *p,
        f'{filename}.{ext}',
    )
