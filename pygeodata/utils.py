from affine import Affine


def transform_to_str(t: Affine) -> str:
    return f'affine_{t.a:.4f}_{t.b:.4f}_{t.c:.4f}_{t.d:.4f}_{t.e:.4f}_{t.f:.4f}'
