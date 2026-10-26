"""Derive catalog dimension rows and hashes from a :class:`SpatialSpec`.

The live write hook passes the spec directly; the backfill reconstructs it
with :meth:`SpatialSpec.from_dict` from the exact ``spec.json`` on disk — one
derivation path, shared with the rest of the codebase.

* ``crs_hash`` — hash of the CRS string.
* ``transform_hash`` — hash of the transform; the resolution derives from it.
* ``bounds_hash`` — hash of (transform, shape) together — the bounds derive
  from that pair, so identical bounds from different specs stay separate rows.
* ``shape_hash`` — hash of the shape tuple.

A ``None`` spec (entry without a ``spec.json``) yields ``None`` rows and
hashes, leaving the entry's foreign keys nullable.
"""

from __future__ import annotations

from typing import Any

from pygeodata.hash import calculate_dict_hash, calculate_string_hash
from pygeodata.spec import SpatialSpec, SpecKeys

_NONE_DIMS: dict[str, Any] = {
    'crs_hash': None,
    'crs_row': None,
    'transform_hash': None,
    'resolution_row': None,
    'bounds_hash': None,
    'bounds_row': None,
    'shape_hash': None,
    'shape_row': None,
}


def dimension_rows(spec: SpatialSpec | None) -> dict[str, Any]:
    if spec is None:
        return dict(_NONE_DIMS)

    crs = spec.crs.to_string() if spec.crs is not None else None
    transform = spec.to_dict()[SpecKeys.TRANSFORM] if spec.transform is not None else None
    shape = tuple(spec.shape) if spec.shape is not None else None

    dims = dict(_NONE_DIMS)
    if crs is not None:
        crs_hash = calculate_string_hash(crs)
        dims['crs_hash'] = crs_hash
        dims['crs_row'] = {'crs_hash': crs_hash, 'crs': crs}

    if transform is not None:
        transform_hash = calculate_dict_hash(transform)
        res_x, res_y = spec.resolution
        dims['transform_hash'] = transform_hash
        dims['resolution_row'] = {
            'transform_hash': transform_hash,
            'a': transform['a'],
            'b': transform['b'],
            'c': transform['c'],
            'd': transform['d'],
            'e': transform['e'],
            'f': transform['f'],
            'res_x': res_x,
            'res_y': res_y,
        }

    if transform is not None and shape is not None:
        bounds = spec.bounds
        dims['bounds_hash'] = calculate_dict_hash({'transform': transform, 'shape': list(shape)})
        dims['bounds_row'] = {
            'bounds_hash': dims['bounds_hash'],
            'left': bounds.left,
            'bottom': bounds.bottom,
            'right': bounds.right,
            'top': bounds.top,
        }

    if shape is not None:
        dims['shape_hash'] = calculate_dict_hash({'shape': list(shape)})
        dims['shape_row'] = {'shape_hash': dims['shape_hash'], 'shape': str(shape)}

    return dims
