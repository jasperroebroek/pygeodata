import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pygeodata.config import get_config
from pygeodata.types import SpatialSpec


def sanitize(val: Any) -> str:
    return re.sub(r'[^\w\-.,\[\]\(\)\{\} ]', '_', str(val))


def generate_path(
    spec: SpatialSpec,
    base_dir: str | Path,
    name: str | None = None,
    max_path_param_depth: int | None = None,
    **kwargs,
) -> Path:
    """Function that converts a path of the data to the processed data."""
    base_dir = Path(base_dir)

    max_depth = max_path_param_depth
    if max_depth is None:
        max_depth = getattr(get_config(), 'max_path_param_depth', float('inf'))
    if max_depth is None:
        max_depth = float('inf')

    if kwargs:
        safe_kwargs = {sanitize(k): sanitize(v) for k, v in kwargs.items()}
        if len(safe_kwargs) > max_depth:
            param_str = json.dumps(safe_kwargs, sort_keys=True)
            params = [hashlib.sha256(param_str.encode('utf-8')).hexdigest()]
        else:
            params = [f'{k}={v}' for k, v in sorted(safe_kwargs.items())]
    else:
        params = []

    if spec.shape is None or spec.transform is None:
        geo_str = 'vector'
    else:
        t = spec.transform
        geo_str = (
            f'affine_{t.a:.4f}_{t.b:.4f}_{t.c:.4f}_{t.d:.4f}_{t.e:.4f}_{t.f:.4f}_shape_{spec.shape[0]}_{spec.shape[1]}'
        )

    crs_str = re.sub(r'[^\w\-]', '_', spec.crs.to_string())

    parts = [crs_str, geo_str]
    if name is not None:
        parts.append(name)

    parts.extend(params)

    return Path(
        base_dir,
        *parts,
    )
