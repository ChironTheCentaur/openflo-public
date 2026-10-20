"""Voltage optimization (Tools -> Voltage optimization…) — IO + dialog path.

tests/test_voltage.py covers the pure metric layer and tests/test_ui_voltage.py
only constructs the dialog. VoltageTitration.analyze (FCS + $PnV) and the
dialog's folder -> channel -> Run -> recommendation path were untested.

Ground truth: 7 files at 400..700 V, each with negatives ~ N(1000, 100) and
positives ~ N(1000 + 200*SI, 100), so the Stain Index is SI by construction:
[2, 5, 9, 12, 12.4, 12.5, 12.5]. Lowest V with SI >= 95% of max -> 550 V.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import gui_unavailable

VOLTS = [400, 450, 500, 550, 600, 650, 700]
SI = [2.0, 5.0, 9.0, 12.0, 12.4, 12.5, 12.5]


def _write_series(folder, seed=0, n=3000):
    import flowio
    rng = np.random.default_rng(seed)
    for v, si in zip(VOLTS, SI, strict=True):
        pe = np.concatenate([rng.normal(1000, 100, n),
                             rng.normal(1000 + 200 * si, 100, n)])
        df = pd.DataFrame({'FSC-A': rng.uniform(5e4, 1e5, 2 * n),
                           'SSC-A': rng.uniform(5e4, 1e5, 2 * n), 'FL5-A': pe})
        with open(os.path.join(folder, f'tube_{v}V.fcs'), 'wb') as fh:
            flowio.create_fcs(fh, df.to_numpy().flatten().tolist(),
                              list(df.columns),
                              metadata_dict={'P3V': str(v)})


def test_analyze_recovers_constructed_stain_index_and_plateau(tmp_path):
    from openflo.voltage import VoltageTitration
    _write_series(str(tmp_path))
    res = VoltageTitration.analyze(sorted(str(p) for p in
                                          tmp_path.glob('*.fcs')),
                                   channels=['FL5-A'])
    assert res['order'] == ['FL5-A']
    r = res['results']['FL5-A']
    assert [row['voltage'] for row in r['rows']] == [float(v) for v in VOLTS]
    np.testing.assert_allclose([row['si'] for row in r['rows']], SI,
                               rtol=0.05)
    assert r['recommended_voltage'] == 550.0


def _dialog_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    root._begin_busy = lambda *a, **k: None
    root._end_busy = lambda *a, **k: None
    return root


def _inline(widget, work, on_done=None, on_error=None, on_finally=None):
    try:
        out = work()
    except Exception as exc:                        # noqa: BLE001
        if on_error:
            on_error(exc)
    else:
        if on_done:
            on_done(out)
    finally:
        if on_finally:
            on_finally()


def test_voltage_dialog_folder_run_recommends_plateau(tmp_path, monkeypatch):
    import openflo.ui_voltage as uv
    data = tmp_path / 'series'
    data.mkdir()
    _write_series(str(data))
    root = _dialog_or_skip()
    try:
        monkeypatch.setattr(uv, 'run_async', _inline)
        monkeypatch.setattr(uv.filedialog, 'askdirectory',
                            lambda *a, **k: str(data))
        dlg = uv.VoltageDialog(root)
        dlg._pick_folder()
        # *.fcs and *.FCS match the same files on Windows — counted once
        assert dlg._folder_var.get() == 'series (7 FCS)'
        assert dlg._channel_var.get() == 'FL5-A'          # fluor first
        dlg._run()
        assert dlg._rec_var.get() == (
            'FL5-A: recommended 550 V (>= 95% of max SI, 7 voltage point(s))')
        out = tmp_path / 'si.png'
        monkeypatch.setattr(uv.filedialog, 'asksaveasfilename',
                            lambda *a, **k: str(out))
        dlg._export()
        assert out.read_bytes()[:4] == b'\x89PNG'
        dlg._channel_var.set('FL2-A')                    # not in the files
        dlg._run()
        assert dlg._rec_var.get() == ''
        assert "'FL2-A' was not found" in dlg._status_var.get()
    finally:
        root.destroy()


@pytest.mark.parametrize('volts', [[400, 500]])
def test_voltage_dialog_without_pnv_says_why(tmp_path, monkeypatch, volts):
    """Files with no $PnV keyword: no recommendation, and the text says to
    check $PnV rather than inventing a voltage."""
    import openflo.ui_voltage as uv
    from openflo.fcs_export import write_fcs
    rng = np.random.default_rng(1)
    for v in volts:
        pe = np.concatenate([rng.normal(1000, 100, 500),
                             rng.normal(3000, 100, 500)])
        write_fcs(pd.DataFrame({'FSC-A': np.full(1000, 7e4), 'FL5-A': pe}),
                  str(tmp_path / f't{v}.fcs'))
    root = _dialog_or_skip()
    try:
        monkeypatch.setattr(uv, 'run_async', _inline)
        monkeypatch.setattr(uv.filedialog, 'askdirectory',
                            lambda *a, **k: str(tmp_path))
        dlg = uv.VoltageDialog(root)
        dlg._pick_folder()
        dlg._run()
        assert 'no recommendation' in dlg._rec_var.get()
        assert '$PnV' in dlg._rec_var.get()
    finally:
        root.destroy()
