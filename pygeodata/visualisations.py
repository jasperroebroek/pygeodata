import html
from pathlib import Path
from typing import Any

from graphviz import Digraph

from pygeodata.loader import DataLoader


def _get_class_name(cls_or_str: Any) -> str:
    """Safely extracts the name from a class object or a string."""
    return cls_or_str if isinstance(cls_or_str, str) else cls_or_str.__name__


def _get_named_dependencies(params: dict) -> list[tuple[str, Any]]:
    """Extracts DataLoaders and tracks which parameter key they belong to."""
    deps = []

    def extract(val: Any, name: str):
        if isinstance(val, DataLoader):
            deps.append((name, val))
        elif isinstance(val, (list, tuple, set)):
            for i, item in enumerate(val):
                extract(item, f'{name}[{i}]')
        elif isinstance(val, dict):
            for k_sub, item in val.items():
                extract(item, f'{name}[{k_sub}]')

    for k, v in params.items():
        extract(v, k)

    return deps


def _build_html_label(current_loader: DataLoader, show_params: bool, show_inheritance: bool, show_calls: bool) -> str:
    """Builds the lowercase HTML table label for the node."""
    class_name = current_loader.get_class_name()

    # 1. Header Row
    rows = [f'<tr><td bgcolor="#d0e4fe"><b>{class_name}</b></td></tr>']

    # 2. Static Code Context Row (Grey)
    meta_lines = []
    if isinstance(current_loader, DataLoader):
        metadata = current_loader.get_source_metadata()

        if show_inheritance and metadata.get('inheritance_dependencies'):
            inh = [_get_class_name(c) for c in metadata['inheritance_dependencies']]
            inheritance_str = '<br/>'.join(inh)
            meta_lines.append(f'<b>inherits</b>:<br/> {inheritance_str}')

        if show_calls and metadata.get('call_dependencies'):
            cal = [_get_class_name(c) for c in metadata['call_dependencies']]
            call_str = '<br/>'.join(cal)
            meta_lines.append(f'<b>calls</b>:<br/> {call_str}')

    if meta_lines:
        meta_str = '<br/>'.join(meta_lines)
        rows.append(f'<tr><td bgcolor="#f2f2f2"><font point-size="10" color="#555555">{meta_str}</font></td></tr>')

    # 3. Parameters Row (White)
    if show_params:
        params = current_loader.get_params()
        prim_params = []
        for k, v in params.items():
            if not isinstance(v, DataLoader):
                safe_val = html.escape(str(v)).replace('\n', '<br align="left"/>')
                prim_params.append(f'{k}={safe_val}')

        if prim_params:
            param_str = '<br align="left"/>'.join(prim_params)
            rows.append(
                f'<tr><td bgcolor="#ffffff" align="left" balign="left">'
                f'<font point-size="11">{param_str}<br align="left"/></font>'
                f'</td></tr>',
            )

    # Wrap everything in a lowercase html table
    return f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">{"".join(rows)}</table>>'


def _add_nodes_and_edges(
    current_loader: DataLoader,
    dot: Digraph,
    visited_hashes: set[str],
    show_params: bool,
    show_inheritance: bool,
    show_calls: bool,
) -> str:
    """Recursively adds nodes and edges to the Graphviz Digraph."""
    node_id = current_loader.get_state_hash()

    if node_id in visited_hashes:
        return node_id
    visited_hashes.add(node_id)

    label = _build_html_label(current_loader, show_params, show_inheritance, show_calls)
    dot.node(node_id, label=label)

    deps = _get_named_dependencies(current_loader.get_params())
    for edge_label, dep in deps:
        dep_id = _add_nodes_and_edges(dep, dot, visited_hashes, show_params, show_inheritance, show_calls)

        padded_label = f'{edge_label}\n'

        dot.edge(
            dep_id,
            node_id,
            color='#000000',
            style='solid',
            penwidth='1.5',
            label=padded_label,
            fontcolor='#333333',
            fontsize='11',
            fontname='Helvetica',
        )

    return node_id


def plot_compact_execution_graph(
    loader: DataLoader,
    view: bool = False,
    out_path: str = 'compact_execution_graph',
    show_params: bool = True,
    show_inheritance: bool = True,
    show_calls: bool = True,
) -> Digraph:
    """
    Plots the runtime dependency data-flow graph.
    Collapses inheritance and call dependencies neatly into the node labels using lowercase HTML tables.
    Edges are labeled with the parameter names they originate from.
    """
    dot = Digraph(
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

    visited_runtime_hashes = set()

    _add_nodes_and_edges(
        current_loader=loader,
        dot=dot,
        visited_hashes=visited_runtime_hashes,
        show_params=show_params,
        show_inheritance=show_inheritance,
        show_calls=show_calls,
    )

    dot.render(out_path, format='png', view=view, cleanup=True)
    return dot


def plot_class_dependency_graph(loader: type[DataLoader], path: Path, view: bool = True) -> Digraph:
    """
    Plots the class dependency graph and saves it in the code registry directory.
    Returns the path to the generated PNG, or None if graphviz is missing.
    """
    graph_data = loader.get_dependency_graph()

    dot = Digraph(comment=f'{loader.__name__} Dependency Graph')
    dot.attr(rankdir='LR')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    for node_cls in graph_data['nodes'].values():
        dot.node(
            node_cls.__name__,
            label=node_cls.__name__,
            shape='box',
            style='rounded,filled',
            fillcolor='#f8f9fa',
        )

    for src, dst in graph_data['inheritance_edges']:
        dot.edge(
            src.__name__,
            dst.__name__,
            style='solid',
            color='#2c3e50',
            arrowhead='empty',
            label=' inherits',
        )

    for src, dst in graph_data['call_edges']:
        dot.edge(
            src.__name__,
            dst.__name__,
            style='dashed',
            color='#e74c3c',
            arrowhead='normal',
            label=' calls',
        )

    dot.render(str(path), format='png', view=view, cleanup=True)
    return dot
