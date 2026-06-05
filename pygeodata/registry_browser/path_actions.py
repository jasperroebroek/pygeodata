import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def open_path(path: str) -> None:
    p = Path(path)

    if not p.exists():
        logger.error('Cannot open path because it does not exist: %s', p)
        raise FileNotFoundError(path)

    logger.info('Opening path: %s', p)

    if sys.platform.startswith('darwin'):
        try:
            result = subprocess.run(['open', str(p)], check=False)
        except OSError as exc:
            logger.error('Failed to execute "open" for %s: %s', p, exc)
            raise
        if result.returncode != 0:
            logger.warning('Command "open" returned non-zero exit status %s for %s', result.returncode, p)
        return

    if os.name == 'nt':
        try:
            os.startfile(str(p))
        except OSError as exc:
            logger.error('Failed to open path with os.startfile for %s: %s', p, exc)
            raise
        return

    try:
        result = subprocess.run(['xdg-open', str(p)], check=False)
    except OSError as exc:
        logger.error('Failed to execute "xdg-open" for %s: %s', p, exc)
        raise

    if result.returncode != 0:
        logger.warning('Command "xdg-open" returned non-zero exit status %s for %s', result.returncode, p)


def reveal_path(path: str) -> None:
    p = Path(path)
    logger.info('Revealing path: %s', p)

    if sys.platform.startswith('darwin'):
        # open -R selects the file in Finder rather than opening the directory
        subprocess.run(['open', '-R', str(p)], check=False)
        return

    # On other platforms fall back to opening the parent directory
    target = p if p.is_dir() else p.parent
    open_path(str(target))
