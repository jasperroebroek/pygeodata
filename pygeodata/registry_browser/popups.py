import html
import json
import logging
import re
import tempfile
from pathlib import Path

from pygeodata.ast import get_source_code
from pygeodata.graphs import plot_class_dependency_graph

from pygeodata.registries.registry import SourceRegistry
from pygeodata.registry_browser.io_utils import read_text
from pygeodata.tracked_object import TrackedObject

logger = logging.getLogger(__name__)


def _linkify_class_names(escaped_source: str, known_classes: frozenset[str], current_class: str) -> str:
    """Replace occurrences of known class names in HTML-escaped source with clickable spans."""
    sorted_names = sorted(known_classes - {current_class}, key=len, reverse=True)
    if not sorted_names:
        return escaped_source

    # Build a pattern that matches whole identifiers only
    pattern = r'\b(' + '|'.join(re.escape(n) for n in sorted_names) + r')\b'

    def replace(m: re.Match) -> str:
        name = m.group(1)
        return (
            f'<span class="src-cls-link" data-cls="{html.escape(name)}" '
            f'title="Jump to {html.escape(name)}">{html.escape(name)}</span>'
        )

    return re.sub(pattern, replace, escaped_source)


def build_json_popup(file_path: str) -> dict:
    """Return JSON file contents as parsed data for the client-side explorer."""
    path = Path(file_path)
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise FileNotFoundError(file_path) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON: {file_path}') from exc

    return {
        'title': path.name,
        'json': data,
    }


def render_source_html(source_text: str, known_classes: frozenset[str], current_class: str) -> str:
    lines = source_text.splitlines()
    rows = []
    for i, line in enumerate(lines, 1):
        escaped_line = _linkify_class_names(html.escape(line), known_classes, current_class)
        rows.append(f'<tr class="diff-ctx"><td class="diff-ln">{i}</td><td class="diff-code">{escaped_line}</td></tr>')
    return f'<table class="diff-table diff-table--inline source-table">{"".join(rows)}</table>'


def build_source_popup(class_name: str, source_path: str | None = None) -> dict[str, str]:
    cls = TrackedObject.find_object_class(class_name)

    if cls is not None:
        source = get_source_code(cls)
    elif source_path is not None:
        try:
            source = Path(source_path).read_text()
        except OSError as exc:
            raise FileNotFoundError(source_path) from exc
    else:
        logger.error('Cannot build source popup: class not in registry and no source_path: %s', class_name)
        raise KeyError(class_name)

    src = SourceRegistry()
    known_classes = frozenset(TrackedObject._registry.keys()) | frozenset(src.class_names)
    body = render_source_html(source, known_classes, class_name)

    return {
        'title': f'Source · {class_name}',
        'html': body,
    }


def _inject_graph_links(svg: str, known_classes: frozenset[str]) -> str:
    """Add data-cls attributes to graphviz node <g> elements so JS can make them clickable."""

    # Each node group looks like: <g id="nodeN" class="node">\n<title>ClassName</title>
    def replace_node(m: re.Match) -> str:
        g_attrs = m.group(1)
        title = m.group(2)
        rest = m.group(3)
        if title in known_classes:
            merged_attrs = g_attrs.replace('class="node"', 'class="node graph-node-link"')
            return f'<g {merged_attrs} data-cls="{html.escape(title)}">\n<title>{html.escape(title)}</title>{rest}'
        return m.group(0)

    pattern = re.compile(
        r'<g ([^>]*class="node"[^>]*)>\s*<title>([^<]+)</title>(.*?)</g>',
        re.DOTALL,
    )
    return pattern.sub(replace_node, svg)


def build_graph_popup(class_name: str, graph_path: str | None = None) -> dict[str, str]:
    logger.info('Building graph popup for class %s', class_name)

    cls = TrackedObject.find_object_class(class_name)

    if cls is not None:
        graph_data = cls.get_dependency_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f'{class_name}.svg'
            plot_class_dependency_graph(class_name, graph_data, path=path, view=False)
            svg = read_text(path)

        if svg is None:
            logger.error('Dependency graph SVG was not created for class %s', class_name)
            raise FileNotFoundError(class_name)

        known_classes = frozenset(TrackedObject._registry.keys())
        svg = _inject_graph_links(svg, known_classes)

        return {
            'title': f'Graph · {class_name}',
            'svg': svg,
        }

    # Class not in registry — serve the pre-rendered PDF path for the frontend to display
    if graph_path is not None:
        return {
            'title': f'Graph · {class_name}',
            'pdf_path': graph_path,
        }

    logger.error('Cannot build graph popup: class not in registry and no graph_path: %s', class_name)
    raise KeyError(class_name)
