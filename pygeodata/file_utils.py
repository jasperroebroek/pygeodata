from pathlib import Path


def classify_file(path: Path) -> str:
    """Return a broad kind string for the given file path based on its extension."""
    suffix = path.suffix.lower()

    if suffix in {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'}:
        return 'image'
    if suffix == '.pdf':
        return 'pdf'
    if suffix in {'.tif', '.tiff', '.nc', '.vrt', '.npy', '.zarr'}:
        return 'raster'
    if suffix == '.json':
        return 'json'
    if suffix in {'.py', '.pyx', '.ipynb'}:
        return 'code'
    return 'file'
