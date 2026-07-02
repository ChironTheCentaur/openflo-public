"""Gating-tree diagram renderer for OpenFlo.

Standalone, GUI-free utilities to turn a sample's gate dict into a nested
tree structure and to render that hierarchy as a tidy top-down tree figure.

A sample's gates are a dict mapping gate-id -> gate dict, e.g.::

    {
        'g1': {'kind': 'rect', 'channel': 'FSC-A', 'parent_id': None,
               'color': '#1f77b4', 'enabled': True, 'id': 'g1'},
        'g2': {'kind': 'category', 'channel': 'CD3', 'parent_id': 'g1',
               'color': '#ff7f0e', 'enabled': True, 'id': 'g2',
               'name': 'CD3+ T cells'},
    }

Parent/child links are via ``parent_id`` (``None`` => root). Labels prefer
the gate's ``name`` and otherwise fall back to ``"<kind> on <channel>"``.

matplotlib is imported lazily so importing this module never pulls in a GUI
backend; rendering is Agg-safe and never calls ``show()``.
"""
from __future__ import annotations

from typing import Any

__all__ = ['build_tree', 'gate_tree_figure']

_DEFAULT_COLOR = '#888888'


def _gate_label(gate: dict[str, Any], gid: str) -> str:
    """Human-readable label for a gate.

    Prefers an explicit ``name``; otherwise builds ``"<kind> on <channel>"``,
    degrading gracefully when either field is missing.
    """
    name = gate.get('name')
    if isinstance(name, str) and name.strip():
        return name.strip()

    kind = gate.get('kind')
    channel = gate.get('channel')
    kind_s = str(kind).strip() if kind not in (None, '') else 'gate'
    if channel not in (None, ''):
        return f'{kind_s} on {channel}'
    return kind_s if kind_s != 'gate' else str(gid)


def build_tree(gates: dict[str, dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Build a nested tree from a per-sample gates dict.

    Returns a list of root nodes; each node is
    ``{'id': gid, 'label': str, 'children': [<node>, ...]}``.

    Defensive against:
      * orphans (``parent_id`` pointing at a missing gate) -> treated as root;
      * cycles (a gate reachable from itself via ``parent_id``) -> the
        back-edge is broken so every reachable gate appears exactly once;
      * a ``None``/empty input -> returns ``[]``.
    """
    if not gates:
        return []

    # Pre-build nodes keyed by id (preserving insertion order of `gates`).
    nodes: dict[str, dict[str, Any]] = {
        gid: {'id': gid, 'label': _gate_label(gate or {}, gid), 'children': []}
        for gid, gate in gates.items()
    }

    roots: list[dict[str, Any]] = []

    def _is_ancestor(candidate: str, of_node: str) -> bool:
        """True if `candidate` is `of_node` or already an ancestor of it,
        walking up the parent_id chain (bounded by the gate count)."""
        cur: str | None = of_node
        steps = 0
        limit = len(gates) + 1
        while cur is not None and steps <= limit:
            if cur == candidate:
                return True
            parent = gates.get(cur, {}).get('parent_id')
            cur = parent if isinstance(parent, str) else None
            steps += 1
        return False

    for gid, gate in gates.items():
        parent_id = (gate or {}).get('parent_id')
        # Root if no parent, parent missing (orphan), or attaching would
        # create a cycle (parent is gid itself or a descendant of gid).
        if (not isinstance(parent_id, str)
                or parent_id not in nodes
                or parent_id == gid
                or _is_ancestor(gid, parent_id)):
            roots.append(nodes[gid])
        else:
            nodes[parent_id]['children'].append(nodes[gid])

    return roots


def _layout(roots: list[dict[str, Any]]) -> tuple[dict[str, tuple[float, float]], int, int]:
    """Assign (x, y) positions to every node for a top-down tidy tree.

    Leaves are spread evenly along x (left->right in traversal order);
    internal nodes are centered over their children. y is the depth
    (0 at the top). Returns (positions_by_id, leaf_count, max_depth).
    """
    positions: dict[str, tuple[float, float]] = {}
    leaf_counter = [0]
    max_depth = [0]

    def visit(node: dict[str, Any], depth: int) -> float:
        max_depth[0] = max(max_depth[0], depth)
        children = node['children']
        if not children:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
        else:
            xs = [visit(c, depth + 1) for c in children]
            x = sum(xs) / len(xs)
        positions[node['id']] = (x, float(depth))
        return x

    for r in roots:
        visit(r, 0)

    return positions, leaf_counter[0], max_depth[0]


def _color_for(gates: dict[str, dict[str, Any]], gid: str) -> str:
    """Return a usable hex color for a gate, falling back to a neutral gray."""
    color = (gates.get(gid) or {}).get('color')
    if isinstance(color, str) and color.strip():
        return color.strip()
    return _DEFAULT_COLOR


def gate_tree_figure(
    gates: dict[str, dict[str, Any]] | None,
    sample_name: str = '',
):
    """Render a sample's gating hierarchy as a tidy top-down tree.

    Returns a ``matplotlib.figure.Figure`` (the caller saves/embeds it; this
    function never calls ``show()``). Boxes are drawn per population, lines
    connect parent -> child, and each box is colored by the gate's ``color``.
    Lays out reasonably for up to ~30 nodes.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import FancyBboxPatch

    gate_dict = gates or {}
    roots = build_tree(gate_dict)
    positions, n_leaves, max_depth = _layout(roots)

    n_cols = max(n_leaves, 1)
    n_rows = max_depth + 1

    # Size the canvas to the tree shape, within sane bounds.
    fig_w = min(max(n_cols * 1.9, 4.0), 22.0)
    fig_h = min(max(n_rows * 1.25, 2.5), 16.0)
    fig = Figure(figsize=(fig_w, fig_h), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    title = f'Gating tree — {sample_name}' if sample_name else 'Gating tree'
    ax.set_title(title, fontsize=11, fontweight='bold')

    if not positions:
        ax.text(0.5, 0.5, 'No gates', ha='center', va='center',
                transform=ax.transAxes, fontsize=11, color='gray')
        return fig

    # World coords: x = column index, y = depth. Flip y so depth 0 is on top.
    def to_xy(gid: str) -> tuple[float, float]:
        x, depth = positions[gid]
        return x, float(max_depth) - depth

    box_w, box_h = 0.82, 0.5
    label_map = {n['id']: n['label'] for n in _iter_nodes(roots)}

    # Edges first (so boxes sit on top).
    for node in _iter_nodes(roots):
        px, py = to_xy(node['id'])
        for child in node['children']:
            cx, cy = to_xy(child['id'])
            ax.plot([px, cx], [py - box_h / 2, cy + box_h / 2],
                    color='#555555', linewidth=1.0, zorder=1)

    # Boxes + labels.
    for gid in positions:
        cx, cy = to_xy(gid)
        color = _color_for(gate_dict, gid)
        box = FancyBboxPatch(
            (cx - box_w / 2, cy - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.02,rounding_size=0.08',
            linewidth=1.2, edgecolor=color,
            facecolor=color, alpha=0.22, zorder=2,
        )
        ax.add_patch(box)
        ax.text(cx, cy, label_map.get(gid, gid), ha='center', va='center',
                fontsize=8, zorder=3, wrap=True,
                clip_on=True)

    ax.set_xlim(-0.7, n_cols - 0.3)
    ax.set_ylim(-0.7, max_depth + 0.7)
    fig.tight_layout()
    return fig


def _iter_nodes(roots: list[dict[str, Any]]):
    """Depth-first iteration over every node in the tree."""
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node['children']))
