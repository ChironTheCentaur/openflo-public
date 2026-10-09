"""Tools -> Compare workspace: the dialog reports the same counts as the CLI.

ui_compare.CompareWspDialog was only constructed and destroyed by the suite
(test_ui_compare); _run/_worker/_on_done/_export never ran. A synthetic .wsp +
FCS give FlowJo-declared counts that are right by construction:

    run1.fcs: 1000 events, FSC-A = 0..999, Time = tick 0..999, $TIMESTEP 0.01 s
    'Big'    FSC-A min 700           -> 300 events (700..999: Gating-ML's min is inclusive)
    'Early'  Time [0, 5) SECONDS     -> ticks 0..499 -> 500 events

The .wsp's DataSet uri points at a folder that no longer exists, so the FCS
directory picked in the dialog is the only way to the file.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(__file__))

from test_gui_smoke import _editor_or_skip  # noqa: E402

_NS = ('xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating" '
       'xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"')


def _pop(name, count, dim, lo, hi=None):
    hi_attr = f' gating:max="{hi}"' if hi is not None else ''
    return (f'<Population name="{name}" count="{count}"><Gate>'
            f'<gating:RectangleGate {_NS}><gating:dimension gating:min="{lo}"'
            f'{hi_attr}><data-type:fcs-dimension data-type:name="{dim}"/>'
            f'</gating:dimension></gating:RectangleGate></Gate></Population>')


def _workspace(tmp_path, pops):
    import flowio
    d = tmp_path / 'fcs'
    d.mkdir()
    n = 1000
    ev = np.column_stack([np.arange(n, dtype=float), np.full(n, 5e4),
                          np.arange(n, dtype=float)]).astype(np.float32)
    with open(d / 'run1.fcs', 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ['FSC-A', 'SSC-A', 'Time'],
                          metadata_dict={'$TIMESTEP': '0.01'})
    wsp = tmp_path / 'exp.wsp'
    wsp.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><Workspace><SampleList><Sample>'
        '<DataSet uri="file:///Z:/moved_away/run1.fcs" sampleID="1"/>'
        '<SampleNode name="run1.fcs" count="1000" sampleID="1"><Subpopulations>'
        + ''.join(pops) +
        '</Subpopulations></SampleNode></Sample></SampleList></Workspace>',
        encoding='utf-8')
    return str(wsp), str(d)


def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        r = work()
    except Exception as exc:                      # noqa: BLE001
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(r)
    finally:
        if on_finally:
            on_finally()


def _run_dialog(monkeypatch, wsp, fcs_dir):
    import openflo.ui_compare as uc
    monkeypatch.setattr(uc, 'run_async', _inline)
    root, ed, gui = _editor_or_skip()
    monkeypatch.setattr(gui.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(gui.messagebox, 'showerror', lambda *a, **k: None)
    d = uc.CompareWspDialog(ed)
    d.withdraw()
    d._wsp_var.set(wsp)
    d._dir_var.set(fcs_dir)
    d._run()
    return root, d


def test_dialog_counts_and_csv_export(monkeypatch, tmp_path):
    import openflo.ui_compare as uc
    wsp, fcs_dir = _workspace(tmp_path, [_pop('Big', 300, 'FSC-A', 700),
                                         _pop('Off', 250, 'FSC-A', 700)])
    root, d = _run_dialog(monkeypatch, wsp, fcs_dir)
    try:
        rows = {d._tree.item(i, 'values')[1]: d._tree.item(i, 'values')
                for i in d._tree.get_children()}
        assert rows['Big'][2:5] == ('300', '300', '+0')
        assert rows['Off'][2:6] == ('250', '300', '+50', '+20.00%')
        # >5 % off is 'bad', 1-5 % 'warn' (judged on rel_delta alone).
        assert uc.CompareWspDialog._row_tag({'rel_delta': 50 / 250}) == 'bad'
        assert uc.CompareWspDialog._row_tag({'rel_delta': 0.02}) == 'warn'
        assert uc.CompareWspDialog._row_tag({'rel_delta': 0.0}) == ''
        assert '2 population row(s)' in d._summary.cget('text')
        assert str(d._export_btn['state']) == 'normal'
        out = tmp_path / 'out.csv'
        monkeypatch.setattr(uc.filedialog, 'asksaveasfilename', lambda **k: str(out))
        d._export()
        got = {r['population']: r for r in csv.DictReader(open(out, encoding='utf-8'))}
        assert got['Off']['openflo_count'] == '300' and got['Off']['delta'] == '50'
        assert float(got['Off']['rel_delta']) == pytest.approx(50 / 250)
    finally:
        root.destroy()


def test_dialog_scales_time_gates_like_the_cli(monkeypatch, tmp_path):
    """The .wsp uri is stale, so only the chosen FCS directory leads to the
    file's $TIMESTEP; the dialog must pass it on as the CLI does."""
    wsp, fcs_dir = _workspace(tmp_path, [_pop('Early', 500, 'Time', 0, 5)])
    from openflo.compare import (
        WspReader,
        _compare_one_sample,
        _per_sample_inventory,
        _resolve_fcs_uri,
    )
    reader = WspReader(wsp)
    (name, uri, pops), = _per_sample_inventory(reader, fcs_dir)   # CLI path
    cli = _compare_one_sample(name, _resolve_fcs_uri(uri, fcs_dir), pops)[0]
    assert cli['openflo_count'] == 500
    root, d = _run_dialog(monkeypatch, wsp, fcs_dir)
    try:
        vals = d._tree.item(d._tree.get_children()[0], 'values')
        assert vals[3] == '500', f'dialog counted {vals[3]} where FlowJo and the CLI count 500'
    finally:
        root.destroy()
