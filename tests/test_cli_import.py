import os
import subprocess
import sys

import click
import pytest

import pygeodata
from pygeodata.cli import COMMANDS, cli

# `import pygeodata.cli` and `pygeodata --help` must load none of these; see AGENTS.md "CLI import time".
FORBIDDEN = (
    'affine',
    'asyncio',
    'dask',
    'filelock',
    'fiona',
    'geopandas',
    'graphviz',
    'matplotlib',
    'numpy',
    'pandas',
    'pyproj',
    'rasterio',
    'shapely',
    'xarray',
)

_REPORT = f'print("LOADED:" + ",".join(m for m in {FORBIDDEN!r} if m in sys.modules))'


@pytest.mark.parametrize(
    'code',
    [
        f'import sys, pygeodata.cli\n{_REPORT}',
        f'import sys\nfrom pygeodata.cli import cli\ntry:\n    cli(["--help"])\n'
        f'except SystemExit:\n    pass\n{_REPORT}',
    ],
    ids=['import', 'help'],
)
def test_cli_leaves_heavy_modules_unloaded(code: str) -> None:
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)  # noqa: S603 — fixed interpreter and code
    report = next(line for line in result.stdout.splitlines() if line.startswith('LOADED:'))
    assert report == 'LOADED:'


@pytest.mark.parametrize('name', sorted(COMMANDS))
def test_lazy_short_help_matches_command(name: str) -> None:
    command = cli.get_command(click.Context(cli), name)
    assert command.get_short_help_str(limit=200) == COMMANDS[name][1]


def test_package_exports_resolve() -> None:
    for name in pygeodata.__all__:
        assert getattr(pygeodata, name) is not None


def test_package_stub_imports_eagerly() -> None:
    env = {**os.environ, 'EAGER_IMPORT': '1'}
    subprocess.run([sys.executable, '-c', 'import pygeodata'], env=env, check=True)  # noqa: S603 — fixed interpreter and code
