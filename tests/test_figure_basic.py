from pathlib import Path

from pygeodata.figure import Figure


def test_figure_default_ext() -> None:
    assert Figure().get_ext() == 'png'


def test_figure_get_processed_base_dir(tmp_path: Path) -> None:
    assert Figure.get_cache_root() == tmp_path / 'figures'
