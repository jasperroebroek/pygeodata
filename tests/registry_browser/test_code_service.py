"""Unit tests for registry_browser/code_service.py."""

import json
from pathlib import Path

from pygeodata.config import JSONKeys, set_config
from pygeodata.registry_browser import code_service
from pygeodata.catalog.types import EntryInfo, SpecInfo
from pygeodata.registries.versioning import VersionRegistry

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_code_snapshot(
    registry_path: Path,
    source_hash: str,
    class_name: str,
    source_text: str,
    mtime: str = '2026-01-01T00:00:00+00:00',
    object_type: str = 'Data',
) -> None:
    code_dir = registry_path / 'code' / source_hash
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / 'source.py').write_text(source_text, encoding='utf-8')
    (code_dir / 'source.json').write_text(
        json.dumps(
            {
                JSONKeys.CLASS_NAME: class_name,
                JSONKeys.OBJECT_TYPE: object_type,
                JSONKeys.SOURCE_HASH: source_hash,
                JSONKeys.REGISTERED_AT: mtime,
            },
        ),
        encoding='utf-8',
    )


def _make_entry(record_id: str, dep_hash: str | None, params_path: str = '') -> EntryInfo:
    return EntryInfo(
        record_id=record_id,
        class_name='MyLoader',
        object_type='Data',
        params_path=params_path,
        spec_path=None,
        state_hash_path=None,
        execution_graph_path=None,
        state_hash=None,
        instance_hash=None,
        params_hash=None,
        spec_hash=None,
        params={},
        spec=SpecInfo(),
        rows=[],
        dep_hash=dep_hash,
    )


def _noop(_path: str):
    return None


# ---------------------------------------------------------------------------
# version_diff — error-path branch coverage
# ---------------------------------------------------------------------------


def test_version_diff_error_paths(tmp_path: Path):
    registry = tmp_path / '.source'
    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg_empty = VersionRegistry()

        # dep_hash=None → no_snapshot
        entry_no_dep = _make_entry('rec1', None)
        result = code_service.version_diff(vreg_empty, record_id='rec1', entries={'rec1': entry_no_dep})
        assert result == {'error': 'no_snapshot', 'message': 'Snapshot not available for this entry'}

        # unknown record_id → __not_found__ sentinel
        result = code_service.version_diff(vreg_empty, record_id='missing', entries={})
        assert result.get('__not_found__') is True

        # dep_hash present but unknown to any version group → no_snapshot
        vreg = VersionRegistry(registry)
        entry_bad_dep = _make_entry('rec1', 'nonexistent_dep_hash')
        result = code_service.version_diff(vreg, record_id='rec1', entries={'rec1': entry_bad_dep})
        assert result['error'] == 'no_snapshot'

        # neither record_id nor base_version_id → bad_request
        result = code_service.version_diff(vreg_empty)
        assert result['error'] == 'bad_request'


def test_version_diff_explicit_base_version_changed(tmp_path: Path):
    """Explicit base_version_id: class changed between base snapshot and live."""
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'h1', 'MyLoader', 'class MyLoader:\n    x = 1\n', mtime='2026-01-01T00:00:00+00:00')
    _write_code_snapshot(registry, 'h2', 'MyLoader', 'class MyLoader:\n    x = 2\n', mtime='2026-06-01T00:00:00+00:00')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        # versions[-1] is Initial (contains h1); versions[0] is the change group (contains h2)
        initial_vid = vreg.versions[-1].version_id
        change_vid = vreg.versions[0].version_id

        # base=Initial (h1), target=change group (h2) → changed
        result = code_service.version_diff(vreg, base_version_id=initial_vid, target_version_id=change_vid)

    assert 'changes' in result
    changes = result['changes']
    assert len(changes) == 1
    c = changes[0]
    assert c['status'] == 'changed'
    assert c['class_name'] == 'MyLoader'
    assert c['hash_old'] == 'h1'
    assert c['hash_new'] == 'h2'


def test_version_diff_result_keys(tmp_path: Path):
    """Result dict must contain changes, base_version_id, has_live_stale."""
    registry = tmp_path / '.source'
    _write_code_snapshot(registry, 'h1', 'StableClass', 'class StableClass: pass\n', mtime='2026-01-01T00:00:00+00:00')

    with set_config(path_cache=tmp_path / 'data', path_figures=tmp_path / 'figs', path_registry=registry):
        vreg = VersionRegistry(registry)
        initial_vid = vreg.versions[-1].version_id
        result = code_service.version_diff(vreg, base_version_id=initial_vid, target_version_id=initial_vid)

    assert set(result.keys()) >= {'changes', 'base_version_id', 'has_live_stale'}
