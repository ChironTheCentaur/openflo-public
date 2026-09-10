"""Regression tests for two FlowJo-import gating bugs found in the Fable review.

Both live in WspReader gate parsing:
  * a max-only 1-D RectangleGate (``x < hi``, no min) used to be dropped, which
    silently re-parented its children and loosened every descendant population;
  * a QuadrantGate's four rects used strict ``<``/``>`` on both sides of each
    divider, so an event exactly on a divider fell into NO quadrant.
"""
import pandas as pd

import openflo.pipeline as fp


def test_max_only_rectanglegate_kept_as_interval_and_child_preserved(tmp_path):
    """A max-only 1-D RectangleGate imports as an interval (sentinel lo) and
    keeps its children parented to it — not dropped + re-parented upward."""
    # Build a valid parent->child hierarchy via the writer (both min-only
    # thresholds), then flip ONLY the parent's dimension min->max so it becomes
    # the max-only case the reader used to discard.
    w = fp.WspWriter(cytometer='t')
    gates = [
        {'kind': 'threshold', 'channel': 'CD3-A', 'value': 1234.0,
         'id': 'P', 'parent_id': None, 'name': 'parent'},
        {'kind': 'threshold', 'channel': 'CD4-A', 'value': 50.0,
         'id': 'C', 'parent_id': 'P', 'name': 'child'},
    ]
    w.add_sample('S', fcs_path='', channels=['CD3-A', 'CD4-A'], gates=gates)
    out = tmp_path / 'maxonly.wsp'
    w.write(str(out))
    xml = out.read_text(encoding='utf-8').replace(
        'gating:min="1234.0"', 'gating:max="1234.0"')      # parent -> max-only
    out.write_text(xml, encoding='utf-8')

    read = fp.WspReader(str(out)).extract_gates()
    parent = next((g for g in read if g.get('kind') == 'interval'
                   and g.get('channel') == 'CD3-A'), None)
    assert parent is not None, 'max-only RectangleGate was dropped on import'
    assert parent['hi'] == 1234.0 and parent['lo'] < -1e11
    child = next(g for g in read if g.get('channel') == 'CD4-A')
    assert child['parent_id'] == parent['_import_id'], \
        'child was re-parented off the max-only gate instead of kept under it'


def test_quadrant_import_is_a_true_partition_on_the_divider(tmp_path):
    """Every event — including one exactly on both dividers — lands in exactly
    one of the four expanded quadrant rects."""
    quad = []
    for lbl, x0, x1, y0, y1 in [('Q++', 100, 1e12, 200, 1e12),
                                ('Q+-', 100, 1e12, -1e12, 200),
                                ('Q-+', -1e12, 100, 200, 1e12),
                                ('Q--', -1e12, 100, -1e12, 200)]:
        quad.append({'kind': 'rect', 'x_channel': 'CD3-A', 'y_channel': 'CD4-A',
                     'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1, 'label': lbl,
                     'quad_set': 'qs1', 'quad_origin_x': 100.0,
                     'quad_origin_y': 200.0, 'id': lbl, 'parent_id': None})
    w = fp.WspWriter(cytometer='t')
    w.add_sample('S', fcs_path='', channels=['CD3-A', 'CD4-A'], gates=quad)
    out = tmp_path / 'quad.wsp'
    w.write(str(out))
    rects = [g for g in fp.WspReader(str(out)).extract_gates()
             if g.get('quad_set')]
    assert len(rects) == 4
    # Row 0 is exactly on both dividers (the regression); rows 1-2 sit inside.
    df = pd.DataFrame({'CD3-A': [100.0, 150.0, 50.0],
                       'CD4-A': [200.0, 250.0, 150.0]})
    for i in range(len(df)):
        n = sum(1 for g in rects if bool(fp.gate_to_mask(g, df)[i]))
        assert n == 1, f'event {i} is in {n} quadrants, expected exactly 1'
