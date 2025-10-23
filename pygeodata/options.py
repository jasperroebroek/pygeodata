from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class RasterCreationOptions:
    """Rasterio creation profile options.

    Parameters
    ----------
    compress : str, optional
        Compression algorithm ('lzw', 'deflate', 'zstd', 'lzma', 'jpeg', 'webp')
    compress_level : int, optional
        Compression level (1-9 for deflate, 1-12 for zstd)
    tiled : bool, default=False
        Whether to create a tiled raster (False for striped layout)
    blockxsize : int, optional
        Tile width (defaults to 256 if tiled=True, must be multiple of 16)
    blockysize : int, optional
        Tile height (defaults to 256 if tiled=True, must be multiple of 16)
    interleave : str, optional
        Band interleave ('pixel', 'band', 'line')
    photometric : str, optional
        Photometric interpretation ('minisblack', 'rgb', 'ycbcr')
    predictor : int, optional
        Predictor for compression (1=none, 2=horizontal, 3=floating point)
    bigtiff : bool | str, optional
        Create BigTIFF file ('yes', 'no', 'if_needed', 'if_safer')
    sparse_ok : bool, optional
        Allow sparse files
    """

    compress: str | None = None
    compress_level: int | None = None
    tiled: bool | None = None
    blockxsize: int | None = None
    blockysize: int | None = None
    interleave: str | None = None
    photometric: str | None = None
    predictor: int | None = None
    bigtiff: bool | str | None = None
    sparse_ok: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
