import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Whatever is written to stdout meanwhile, also by C code and subprocesses, goes to stderr."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)
