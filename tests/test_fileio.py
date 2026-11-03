import json
import os
from pathlib import Path

import pytest

from pygeodata import fileio
from pygeodata.fileio import write_bytes_atomic, write_json_atomic


def test_write_json_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / 'data.json'
    write_json_atomic(path, {'a': 1}, indent=4)
    assert json.loads(path.read_text(encoding='utf-8')) == {'a': 1}
    assert path.read_text(encoding='utf-8') == json.dumps({'a': 1}, indent=4)


def test_write_follows_symlink(tmp_path: Path) -> None:
    target = tmp_path / 'target.json'
    target.write_text('{}', encoding='utf-8')
    link = tmp_path / 'link.json'
    link.symlink_to(target)

    write_json_atomic(link, {'b': 2})

    assert link.is_symlink()
    assert json.loads(target.read_text(encoding='utf-8')) == {'b': 2}


def test_failed_write_keeps_original_and_leaves_no_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'data.json'
    path.write_bytes(b'original')

    def fail(_fd: int) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(fileio.os, 'fsync', fail)
    with pytest.raises(OSError, match='disk full'):
        write_bytes_atomic(path, b'new')

    assert path.read_bytes() == b'original'
    assert [p.name for p in tmp_path.iterdir()] == ['data.json']


def test_file_mode_follows_umask(tmp_path: Path) -> None:
    previous = os.umask(0o022)
    try:
        write_bytes_atomic(tmp_path / 'data.bin', b'x')
    finally:
        os.umask(previous)
    assert (tmp_path / 'data.bin').stat().st_mode & 0o777 == 0o644
