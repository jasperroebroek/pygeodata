from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class RasterioProfile:
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

    compress: Optional[str] = None
    compress_level: Optional[int] = None
    tiled: Optional[bool] = None
    blockxsize: Optional[int] = None
    blockysize: Optional[int] = None
    interleave: Optional[str] = None
    photometric: Optional[str] = None
    predictor: Optional[int] = None
    bigtiff: Optional[bool | str] = None
    sparse_ok: Optional[bool] = None
    num_threads: Optional[int | str] = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}
