"""FlowSOM star-tree window (ui_flowsom_tree.FlowSOMTreeWindow), opened the way
the Analyze menu opens it (``editor._open_flowsom_tree``).

It was 2-3% covered: nothing ever built the window. These check what it
draws against what run_flowsom computed: one star per SOM node, the n-1 edges
of the minimum spanning tree, node event counts that add up to the sample,
each node coloured by the metacluster run_flowsom gave it, and the
"run FlowSOM first" guard.
"""

import numpy as np
import pandas as pd

from tests.conftest import editor_or_skip


def _editor_or_skip():
    return editor_or_skip(accept_confirms=False)


def _flowsom_sample():
    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.5, (300, 3)) for c in (0.0, 6.0, 12.0)])
    df = pd.DataFrame(X, columns=['CD1', 'CD2', 'CD3'])
    s = FlowSample.from_dataframe(df, name='s1')
    s.run_flowsom(channels=['CD1', 'CD2', 'CD3'], grid=(5, 5),
                  n_metaclusters=3, iters=4, seed=0)
    return s


def _tree_windows(ed):
    from openflo.ui_flowsom_tree import FlowSOMTreeWindow
    return [w for w in ed.winfo_children() if isinstance(w, FlowSOMTreeWindow)]


def test_tree_window_draws_the_som_it_was_given():
    from openflo.flowsom import som_assign, som_metacluster
    root, ed = _editor_or_skip()
    try:
        s = _flowsom_sample()
        ed._samples['s1'] = s
        ed._sample_order.append('s1')
        ed._active_sample = 's1'
        ed._open_flowsom_tree()
        (win,) = _tree_windows(ed)
        W = s.flowsom_result['weights']
        n = len(W)
        assert n == 25
        ax = win._fig.axes[0]
        edges = [ln for ln in ax.lines if ln.get_color() == '#cccccc']
        assert len(edges) == n - 1                      # a spanning tree
        assert len(ax.patches) == n                     # one star per node
        nodes = som_assign(s.data[['CD1', 'CD2', 'CD3']].to_numpy(), W)
        assert np.array_equal(win._counts, np.bincount(nodes, minlength=n))
        assert int(win._counts.sum()) == len(s.data)
        meta_of_node = som_metacluster(W, 3)
        occupied = win._counts > 0
        assert np.array_equal(win._node_meta[occupied], meta_of_node[occupied])
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend == ['mc 0', 'mc 1', 'mc 2']
    finally:
        root.destroy()


def test_tree_window_refuses_without_a_flowsom_run():
    from openflo.pipeline import FlowSample
    root, ed = _editor_or_skip()
    try:
        df = pd.DataFrame({'CD1': [1.0, 2.0, 3.0]})
        ed._samples['s1'] = FlowSample.from_dataframe(df, name='s1')
        ed._sample_order.append('s1')
        ed._active_sample = 's1'
        ed._open_flowsom_tree()
        assert _tree_windows(ed) == []
        assert 'Run FlowSOM first' in ed.status_var.get()
    finally:
        root.destroy()
