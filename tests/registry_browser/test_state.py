"""Tests for AppContext / build_state reload behaviour (Unit 8 regression guard).

Covers:
- _load wraps exceptions: ready is set and load_error is recorded.
- _load clears load_error on success.
- delete-then-reload: vanished cache dir doesn't hang (hang regression guard).
- Rebuild picks up a newly-added entry on disk (proves registries actually reloaded).
- reload reconciles the catalog before rebuilding state.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.config import (
    FORMAT_VERSION,
    JSONKeys,
    set_config,
)
from pygeodata.paths import CachePathConstructor
from pygeodata.registries.registry import EntryRegistry
from pygeodata.registry_browser.state import AppContext, _reconcile
from pygeodata.tracked_object import TrackedObject


def _write_hash_file(
    directory: Path,
    stem: str,
    state_hash: str = 'abc123',
    instance_hash: str = 'inst001',
    object_type: str = 'Data',
) -> Path:
    """Write minimal cache files for one entry. Returns the hash file path."""
    directory.mkdir(parents=True, exist_ok=True)
    resolver = CachePathConstructor(directory)
    resolver.state_hash_path.write_text(
        json.dumps(
            {
                JSONKeys.STATE_HASH: state_hash,
                JSONKeys.INSTANCE_HASH: instance_hash,
                JSONKeys.CLASS_NAME: stem,
                JSONKeys.OBJECT_TYPE: object_type,
                JSONKeys.FORMAT_VERSION: FORMAT_VERSION,
                JSONKeys.DEPENDENCY_TREE_HASH: '',
                JSONKeys.CO_OUTPUTS: [],
            },
        ),
    )
    resolver.params_path.write_text('{}')
    return resolver.state_hash_path


@pytest.fixture(autouse=True)
def isolated_config(tmp_path: Path):
    with set_config(
        path_cache=tmp_path / 'data_processed',
        path_figures=tmp_path / 'figures',
        path_registry=tmp_path / '.source',
    ):
        yield tmp_path


# ---------------------------------------------------------------------------
# _load safety: exception → ready set + load_error recorded
# ---------------------------------------------------------------------------


def test_load_error_recorded_on_exception() -> None:
    ctx = AppContext()

    def _boom(*args, **kwargs):
        raise RuntimeError('boom')

    with patch('pygeodata.registry_browser.state.build_state', side_effect=_boom):
        ctx.start_load()
        ctx.ready.wait(timeout=5)

    assert ctx.ready.is_set(), 'ready must be set even when build_state raises'
    assert ctx.load_error is not None
    assert 'RuntimeError' in ctx.load_error
    assert 'boom' in ctx.load_error
    assert ctx.state is None


def test_load_error_cleared_on_success(tmp_path: Path) -> None:
    ctx = AppContext()
    ctx.load_error = 'stale error'

    ctx.start_load()
    ctx.ready.wait(timeout=10)

    assert ctx.ready.is_set()
    assert ctx.load_error is None


# ---------------------------------------------------------------------------
# Delete-then-reload: vanished cache dir must not hang
# ---------------------------------------------------------------------------


def test_delete_then_reload_completes(tmp_path: Path) -> None:
    """Simulates the bug: rmtree an entry dir → reload must finish (not hang)."""
    cache_dir = tmp_path / 'data_processed' / 'MyClass' / 'run1'
    _write_hash_file(cache_dir, 'MyClass', state_hash='del001', instance_hash='i001')

    reg = EntryRegistry()
    assert 'del001' in reg.records

    shutil.rmtree(cache_dir)

    ctx = AppContext()
    ctx.ready.set()
    ctx.start_reload()
    finished = ctx.ready.wait(timeout=10)

    assert finished, 'reload hung — ready was never set after delete'
    assert ctx.ready.is_set()


# ---------------------------------------------------------------------------
# Rebuild picks up a newly-added entry on disk
# ---------------------------------------------------------------------------


def test_rebuild_picks_up_new_entry(tmp_path: Path) -> None:
    """After reload, a newly-written entry must appear in AppState.entries."""
    ctx = AppContext()
    ctx.start_load()
    ctx.ready.wait(timeout=10)
    assert ctx.state is not None
    initial_count = len(ctx.state.entries)

    cache_dir = tmp_path / 'data_processed' / 'NewClass' / 'run1'
    _write_hash_file(cache_dir, 'NewClass', state_hash='new001', instance_hash='i002')

    ctx.start_reload()
    ctx.ready.wait(timeout=10)

    assert ctx.ready.is_set()
    assert ctx.load_error is None, ctx.load_error
    assert ctx.state is not None
    assert len(ctx.state.entries) == initial_count + 1 or 'new001' in {e.state_hash for e in ctx.state.entries.values()}


# ---------------------------------------------------------------------------
# reload reconciles the catalog
# ---------------------------------------------------------------------------


def test_reimport_rebuild_reconciles_catalog(tmp_path: Path) -> None:
    """A reimport rebuild reconciles the catalog before rebuilding state."""
    registry_root = tmp_path / '.source'
    registry_root.mkdir(parents=True, exist_ok=True)

    with set_config(path_registry=registry_root, path_cache=tmp_path / 'data', path_figures=tmp_path / 'figures'):
        reconciled: list[bool] = []

        with (
            patch('pygeodata.registry_browser.state._reconcile', side_effect=lambda: reconciled.append(True)),
            patch('pygeodata.registry_browser.state._collect_reimport_files', return_value=[]),
            patch('pygeodata.registry_browser.state._reimport_modules', return_value=iter(())),
            patch('pygeodata.registry_browser.state.build_state'),
            patch.object(TrackedObject, '_registry', {}),
        ):
            ctx = AppContext()
            ctx.start_reload(reimport=True)
            ctx.ready.wait(timeout=60)

        assert reconciled, 'reimport must reconcile the catalog'
