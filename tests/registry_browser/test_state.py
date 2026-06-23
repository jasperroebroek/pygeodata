"""Tests for AppContext / build_state reload behaviour (Unit 8 regression guard).

Covers:
- _load wraps exceptions: ready is set and load_error is recorded.
- _load clears load_error on success.
- delete-then-reload: vanished cache dir doesn't hang (hang regression guard).
- Rebuild picks up a newly-added entry on disk (proves registries actually reloaded).
- tmp JSONs removed and regenerated across a reload.
"""

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from pygeodata.config import FORMAT_VERSION, JSONKeys, set_config
from pygeodata.paths import CachePathConstructor
from pygeodata.registry import EntryRegistry
from pygeodata.registry_browser.entry_catalog import _cache_file
from pygeodata.registry_browser.state import AppContext, _purge_caches


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
    assert len(ctx.state.entries) == initial_count + 1 or 'new001' in {
        e.state_hash for e in ctx.state.entries.values()
    }


# ---------------------------------------------------------------------------
# tmp JSONs are removed by _purge_caches and regenerated after reload
# ---------------------------------------------------------------------------


def test_purge_caches_removes_tmp_jsons(tmp_path: Path) -> None:
    """_purge_caches deletes .dashboard_cache.json and .entry_registry_cache.json."""
    registry_root = tmp_path / '.source'
    registry_root.mkdir(parents=True, exist_ok=True)

    dashboard_cache = _cache_file()
    dashboard_cache.parent.mkdir(parents=True, exist_ok=True)
    dashboard_cache.write_text('{"test": 1}')

    entry_cache = EntryRegistry()._cache_path()
    entry_cache.write_text('{"test": 2}')

    _purge_caches()

    assert not dashboard_cache.exists(), 'dashboard cache should be deleted'
    assert not entry_cache.exists(), 'entry registry cache should be deleted'


def test_purge_caches_is_idempotent() -> None:
    """_purge_caches must not raise when the files are already absent."""
    _purge_caches()


def test_tmp_jsons_regenerated_after_reload(tmp_path: Path) -> None:
    """After a reload, the entry registry cache is written back to disk."""
    registry_root = tmp_path / '.source'
    registry_root.mkdir(parents=True, exist_ok=True)

    reg = EntryRegistry()
    entry_cache = reg._cache_path()
    entry_cache.unlink(missing_ok=True)

    # Constructing a fresh EntryRegistry triggers reload → writes cache
    EntryRegistry()
    assert entry_cache.exists()
