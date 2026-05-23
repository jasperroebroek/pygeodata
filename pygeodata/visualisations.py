import html
import warnings
from collections.abc import Mapping, Sequence, Set
from typing import Any

from pygeodata.artifact import Artifact
from pygeodata.data import Data
from pygeodata.figure import Figure
from pygeodata.formatting import format_value_as_string
from pygeodata.tracked_object import TrackedObject

try:
    from graphviz import Digraph

    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    warnings.warn(
        'graphviz is not installed; skipping visualisation',
        ImportWarning,
        stacklevel=2,
    )


def _indent(level: int) -> str:
    return '\t' * level


def format_value_for_html(value: Any, indent: int = 0, nested: bool = True) -> str:
    if isinstance(value, Artifact):
        prefix = _indent(indent) if nested else ''
        return f'{prefix}{value.get_class_name()}'

    if isinstance(value, Mapping):
        lines: list[str] = []
        for k, v in sorted(value.items()):
            val_str = format_value_for_html(v, indent + 1, nested=True)
            first, *rest = val_str.split('\n')
            lines.append(f'{_indent(indent)}{k} = {first.lstrip()}')
            lines.extend(rest)
        return '\n'.join(lines)

    if isinstance(value, (Sequence, Set)) and not isinstance(value, (str, bytes, bytearray)):
        items = sorted(value, key=repr) if isinstance(value, Set) else value
        lines: list[str] = []
        for item in items:
            if isinstance(item, (Sequence, Set)) and not isinstance(item, (str, bytes, bytearray)):
                lines.append(f'{_indent(indent)}-')
                nested_str = format_value_for_html(item, indent + 1, nested=True)
                lines.extend(nested_str.split('\n'))
            else:
                lines.append(f'{_indent(indent)}- {format_value_as_string(item)}')
        return '\n'.join(lines)

    prefix = _indent(indent) if nested else ''
    return f'{prefix}{format_value_as_string(value)}'


def _get_class_name(cls_or_str: Any) -> str:
    """Safely extracts the name from a class object or a string."""
    return cls_or_str if isinstance(cls_or_str, str) else cls_or_str.__name__


def _get_named_dependencies(params: dict) -> list[tuple[str, Any]]:
    """Extracts Artifacts and tracks which parameter key they belong to."""
    deps = []

    def extract(val: Any, name: str):
        if isinstance(val, Artifact):
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


def _build_html_label(current_artifact: Artifact, show_params: bool, show_inheritance: bool, show_calls: bool) -> str:
    """Builds the lowercase HTML table label for the node."""
    class_name = current_artifact.get_class_name()

    # 1. Header Row
    rows = [f'<tr><td bgcolor="#d0e4fe"><b>{class_name}</b></td></tr>']

    # 2. Static Code Context Row (Grey)
    meta_lines = []
    if isinstance(current_artifact, Artifact):
        inheritance_dependencies = current_artifact.get_inheritance_dependencies()
        if show_inheritance and inheritance_dependencies:
            inh = [cls.__name__ for cls in inheritance_dependencies]
            inheritance_str = '<br/>'.join(inh)
            meta_lines.append(f'<b>inherits</b>:<br/> {inheritance_str}')

        call_dependencies = current_artifact.get_call_dependencies()
        if show_calls and call_dependencies:
            cal = [cls.__name__ for cls in call_dependencies]
            call_str = '<br/>'.join(cal)
            meta_lines.append(f'<b>calls</b>:<br/> {call_str}')

    if meta_lines:
        meta_str = '<br/>'.join(meta_lines)
        rows.append(
            f'<tr><td bgcolor="#f2f2f2"><font point-size="10" color="#555555">{meta_str}</font></td></tr>',
        )

    # 3. Parameters Row (White)
    if show_params:
        params = current_artifact.get_params()
        prim_params: list[str] = []

        for k, v in params.items():
            rendered = format_value_for_html(v, indent=1, nested=True)
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

    # Wrap everything in an HTML table
    return f'<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">{"".join(rows)}</table>>'


def _add_nodes_and_edges(
    current_artifact: Artifact,
    dot: Digraph,
    visited_hashes: set[str],
    show_params: bool,
    show_inheritance: bool,
    show_calls: bool,
) -> str:
    """Recursively adds nodes and edges to the Graphviz Digraph."""
    node_id = current_artifact.get_state_hash()

    if node_id in visited_hashes:
        return node_id
    visited_hashes.add(node_id)

    label = _build_html_label(current_artifact, show_params, show_inheritance, show_calls)
    dot.node(node_id, label=label)

    deps = _get_named_dependencies(current_artifact.get_params(exclude=False))
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
    artifact: Artifact,
    view: bool = False,
    out_path: str = 'compact_execution_graph',
    show_params: bool = True,
    show_inheritance: bool = True,
    show_calls: bool = True,
) -> Digraph | None:
    """
    Plots the runtime dependency flow graph.
    Collapses inheritance and call dependencies neatly into the node labels using lowercase HTML tables.
    Edges are labeled with the parameter names they originate from.
    """
    if not HAS_GRAPHVIZ:
        return None

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
        current_artifact=artifact,
        dot=dot,
        visited_hashes=visited_runtime_hashes,
        show_params=show_params,
        show_inheritance=show_inheritance,
        show_calls=show_calls,
    )

    dot.render(out_path, format='png', view=view, cleanup=True)
    return dot


def plot_class_dependency_graph(cls: TrackedObject, view: bool = False) -> Digraph | None:
    """
    Plots the class dependency graph and saves it in the code registry directory.
    Returns the path to the generated PNG, or None if graphviz is missing.
    """
    if not HAS_GRAPHVIZ:
        return None

    fillcolors = {
        TrackedObject: '#f8f9fa',
        Artifact: '#f8f9fa',
        Data: '#d0dceb',
        Figure: '#bda0bc',
    }

    graph_data = cls.get_dependency_graph()

    dot = Digraph(comment=f'{cls.get_class_name()} Dependency Graph')
    dot.attr(rankdir='LR')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    for node_cls in graph_data['nodes'].values():
        dot.node(
            node_cls.__name__,
            label=node_cls.__name__,
            shape='box',
            style='rounded,filled',
            fillcolor=fillcolors[node_cls.object_type],
        )

    for src, dst in graph_data['inheritance_edges']:
        dot.edge(
            src.__name__,
            dst.__name__,
            style='dashed',
            color='#2c3e50',
            arrowhead='empty',
            label=' inherits',
        )

    for src, dst in graph_data['call_edges']:
        dot.edge(
            src.__name__,
            dst.__name__,
            style='solid',
            color='#2c3e50',
            arrowhead='normal',
            label=' calls',
        )

    dir = cls.get_source_registry_dir()
    dir.mkdir(parents=True, exist_ok=True)
    dot.render(str(dir / 'dependency_graph'), format='png', view=view, cleanup=True)
    return dot
