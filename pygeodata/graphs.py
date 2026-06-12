from __future__ import annotations

import html
import warnings
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from graphviz import Digraph

    from pygeodata.graph_types import DependencyGraph, RuntimeDependencyGraph

try:
    from graphviz import Digraph as _Digraph

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

_GRAPHVIZ_WARNED = False


def _warn_graphviz_missing() -> None:
    global _GRAPHVIZ_WARNED
    if not _GRAPHVIZ_WARNED:
        _GRAPHVIZ_WARNED = True
        warnings.warn(
            'graphviz is not installed; dependency graphs will not be generated. '
            'Install it with: pip install pygeodata[viz]',
            ImportWarning,
            stacklevel=3,
        )


def _build_runtime_html_label(
    node,
    show_params: bool,
    show_inheritance: bool,
    show_calls: bool,
) -> str:
    rows = [f'<tr><td bgcolor="#d0e4fe"><b>{html.escape(node.name)}</b></td></tr>']

    meta_lines: list[str] = []

    if show_inheritance and node.inheritance_dependencies:
        inheritance_str = '<br/>'.join(html.escape(cls.__name__) for cls in node.inheritance_dependencies)
        meta_lines.append(f'<b>inherits</b>:<br/> {inheritance_str}')

    if show_calls and node.call_dependencies:
        call_str = '<br/>'.join(html.escape(cls.__name__) for cls in node.call_dependencies)
        meta_lines.append(f'<b>calls</b>:<br/> {call_str}')

    if meta_lines:
        meta_str = '<br/>'.join(meta_lines)
        rows.append(
            f'<tr><td bgcolor="#f2f2f2"><font point-size="10" color="#555555">{meta_str}</font></td></tr>',
        )

    if show_params and node.params:
        from pygeodata.formatting.html import format_html_block
        prim_params: list[str] = []

        for k, v in node.params.items():
            rendered = format_html_block(v, indent=1, nested=True)
            lines = rendered.split('\n')
            text = f'{k} = {lines[0].lstrip()}' if len(lines) == 1 else f'{k} =\n' + '\n'.join(lines)

            safe_val = html.escape(text)
            safe_val = safe_val.replace('\n', '<br align="left"/>')
            safe_val = safe_val.replace('\t', '&nbsp;&nbsp;&nbsp;')
            prim_params.append(safe_val)

        if prim_params:
            param_str = '<br align="left"/>'.join(prim_params)
            rows.append(
                '<tr><td bgcolor="#ffffff" align="left" balign="left">'
                f'<font point-size="11">{param_str}<br align="left"/></font>'
                '</td></tr>',
            )

    return f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">{"".join(rows)}</table>>'


def plot_compact_execution_graph(
    graph_data: RuntimeDependencyGraph,
    root_id: str,
    path: Path | str = 'compact_execution_graph.svg',
    show_params: bool = True,
    show_inheritance: bool = True,
    show_calls: bool = True,
    view: bool = False,
) -> Digraph | None:
    if not HAS_GRAPHVIZ:
        _warn_graphviz_missing()
        return None

    path = Path(path)

    dot = _Digraph(
        name='CompactExecutionGraph',
        graph_attr={
            'rankdir': 'LR',
            'splines': 'polyline',
            'nodesep': '0.5',
            'ranksep': '1.0',
        },
        node_attr={
            'shape': 'none',
            'fontname': 'Helvetica',
        },
    )

    reachable: set[str] = set()

    outgoing: dict[str, list] = {}
    for edge in graph_data.param_edges:
        outgoing.setdefault(edge.dst_id, []).append(edge)

    def visit(node_id: str) -> None:
        if node_id in reachable:
            return
        reachable.add(node_id)

        node = graph_data.nodes[node_id]
        dot.node(
            node_id,
            label=_build_runtime_html_label(node, show_params, show_inheritance, show_calls),
        )

        for edge in outgoing.get(node_id, []):
            visit(edge.src_id)
            dot.edge(
                edge.src_id,
                edge.dst_id,
                color='#000000',
                style='solid',
                penwidth='1.5',
                label=f'{edge.param_name}\n',
                fontcolor='#333333',
                fontsize='11',
                fontname='Helvetica',
            )

    visit(root_id)

    dot.render(outfile=path, view=view, cleanup=True)
    return dot


def plot_class_dependency_graph(
    cls_name: str,
    graph_data: DependencyGraph,
    path: Path = 'class_dependency_graph.svg',
    view: bool = False,
) -> Digraph | None:
    """
    Plots the class dependency graph and saves it in the code registry directory.
    Returns the path to the generated PNG, or None if graphviz is missing.
    """
    if not HAS_GRAPHVIZ:
        _warn_graphviz_missing()
        return None

    dot = _Digraph(comment=f'{cls_name} Dependency Graph')
    dot.attr(rankdir='LR')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    for node_cls in graph_data.nodes:
        dot.node(
            node_cls.name,
            label=node_cls.name,
            shape='box',
            style='rounded,filled',
            fillcolor=node_cls.color,
        )

    for src, dst in graph_data.inheritance_edges:
        dot.edge(
            src.name,
            dst.name,
            style='dashed',
            color='#2c3e50',
            arrowhead='empty',
            label=' inherits',
        )

    for src, dst in graph_data.call_edges:
        dot.edge(
            src.name,
            dst.name,
            style='solid',
            color='#2c3e50',
            arrowhead='normal',
            label=' calls',
        )

    dot.render(outfile=path, view=view, cleanup=True)
    return dot
