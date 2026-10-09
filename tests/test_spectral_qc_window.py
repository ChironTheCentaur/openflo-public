"""Spectral QC window (opened after Unmix) — summary, header and exports.

The QC maths (spectral.unmixing_qc) is tested in tests/test_spectral.py and
tests/test_spectral_degenerate_reference.py; the window that presents it to
the user (ui_spectral_qc.SpectralQCWindow) had ~3-8% coverage.

Ground truth: hand-built reference spectra whose problems are known — a
near-duplicate pair (cosine 0.9999), a near-collinear 2-fluor panel
(condition number > 100) and an all-zero (degenerate) spectrum.
"""
from __future__ import annotations

import os
import tkinter as tk

import numpy as np

from tests.conftest import gui_unavailable

S_SIM = np.array([[0.8, 0.5, 0.1, 0.00],
                  [0.3, 0.9, 0.2, 0.05],
                  [0.31, 0.9, 0.21, 0.05],
                  [0.1, 0.2, 0.9, 0.30]])
FL_SIM = ['FL4', 'FL5', 'PEdim', 'FL2']
S_DEG = np.array([[0.8, 0.5, 0.1, 0.00],
                  [0.3, 0.9, 0.2, 0.05],
                  [0.0, 0.0, 0.0, 0.00]])
FL_DEG = ['FL4', 'FL5', 'FL2']


def _root_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    return root


def _header(win):
    for frame in win.winfo_children():
        for w in frame.winfo_children():
            try:
                t = str(w.cget('text'))
            except tk.TclError:
                continue
            if 'condition number' in t:
                return t
    raise AssertionError('no header label')


def test_spectral_qc_window_lists_similar_pair_and_exports(tmp_path):
    from openflo import ui_spectral_qc as mod
    from openflo.spectral import unmixing_qc
    root = _root_or_skip()
    try:
        qc = unmixing_qc({}, S_SIM, FL_SIM)
        audits = []
        win = mod.SpectralQCWindow(root, qc,
                                   audit=lambda ev, **k: audits.append(ev))
        assert '4 fluors' in _header(win) and '1 similar pair(s)' in _header(win)
        summary = win._summary_text()
        assert summary.startswith('Spectrally-similar pairs')
        assert '• FL5 ~ PEdim  (cosine 1.000)' in summary
        md = win._markdown()
        sim = float(qc['similar_pairs'][0]['similarity'])
        assert f'| FL5 | PEdim | {sim:.4f} |' in md
        saved = (mod.filedialog.asksaveasfilename, mod.messagebox.showinfo)
        mod.messagebox.showinfo = lambda *a, **k: None
        try:
            mod.filedialog.asksaveasfilename = lambda *a, **k: str(
                tmp_path / 'qc.md')
            win._export_md()
            mod.filedialog.asksaveasfilename = lambda *a, **k: str(
                tmp_path / 'qc.png')
            win._export_png()
        finally:
            mod.filedialog.asksaveasfilename, mod.messagebox.showinfo = saved
        text = (tmp_path / 'qc.md').read_text(encoding='utf-8')
        assert f'| FL5 | PEdim | {sim:.4f} |' in text
        assert (tmp_path / 'qc.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'
        assert audits == ['spectral.qc.export', 'spectral.qc.export']
    finally:
        root.destroy()


def test_spectral_qc_header_marks_ill_conditioned_panels():
    from openflo.spectral import unmixing_qc
    from openflo.ui_spectral_qc import SpectralQCWindow
    root = _root_or_skip()
    try:
        near = unmixing_qc({}, np.array([[1.0, 0.0], [1.0, 1e-3]]), ['A', 'B'])
        assert 100 < near['condition_number'] < np.inf
        assert '[ill-conditioned]' in _header(SpectralQCWindow(root, near))
        good = unmixing_qc({}, np.eye(3), ['A', 'B', 'C'])
        hdr = _header(SpectralQCWindow(root, good))
        assert 'condition number 1.0' in hdr and 'ill-conditioned' not in hdr
        deg = unmixing_qc({}, S_DEG, FL_DEG)
        assert '∞  [ill-conditioned]' in _header(SpectralQCWindow(root, deg))
    finally:
        root.destroy()


def test_spectral_qc_names_a_degenerate_fluor():
    """unmixing_qc names an unusable reference spectrum in
    'degenerate_fluors', but the window never read that key: it told the
    user 'spectra are well separated' and the exported report never named
    the fluor."""
    from openflo.spectral import unmixing_qc
    from openflo.ui_spectral_qc import SpectralQCWindow
    root = _root_or_skip()
    try:
        win = SpectralQCWindow(root, unmixing_qc({}, S_DEG, FL_DEG))
        summary = win._summary_text()
        assert 'well separated' not in summary
        assert 'FL2' in summary
        assert 'FL2' in win._markdown()
    finally:
        root.destroy()


def test_ssm_heatmap_puts_the_stain_on_the_rows_and_says_so():
    """qc['ssm'][i, j] is the spread INTO i FROM stain j; the usual SSM table
    has the stain on the rows. The heatmap had no axis labels, so the matrix
    read the wrong way round."""
    from openflo.ui_spectral_qc import SpectralQCWindow
    root = _root_or_skip()
    try:
        ssm = np.array([[0.0, 0.9, 0.1],
                        [0.2, 0.0, 0.3],
                        [0.4, 0.5, 0.0]])
        qc = {'fluors': ['A', 'B', 'C'], 'similarity': np.eye(3), 'ssm': ssm,
              'condition_number': 1.0, 'similar_pairs': [],
              'worst_spread': [], 'degenerate_fluors': []}
        win = SpectralQCWindow(root, qc)
        ax = win._fig.axes[2]            # similarity, its colorbar, then SSM
        shown = np.asarray(ax.get_images()[0].get_array())
        np.testing.assert_array_equal(shown, ssm.T)
        assert shown[1, 0] == 0.9        # row B (from) -> column A (into)
        assert 'spread from' in ax.get_ylabel()
        assert 'spread into' in ax.get_xlabel()
    finally:
        root.destroy()


def test_spectral_qc_shows_how_each_spectrum_was_built_and_the_dim_ones():
    """A control too dim to define a spectrum, or one whose spectrum was
    re-estimated against matched autofluorescence, plotted like any other:
    the window now says how each was built and names the dim ones (header,
    summary and the exported Markdown). See tests/test_spectral_matched_af.py.
    """
    from openflo.spectral import unmixing_qc
    from openflo.ui_spectral_qc import SpectralQCWindow
    ref = [{'fluor': 'FL4', 'method': 'total', 'requested_method': 'auto',
            'reason': 'total and matched estimates agree (cosine 0.9995); '
                      'kept total', 'primary_detector': 'B2-A',
            'separation_index': 41.2, 'n_events': 5000, 'n_positives': 500,
            'agreement': 0.9995, 'af_cosine': 0.31, 'warning': None},
           {'fluor': 'FL5', 'method': 'matched', 'requested_method': 'auto',
            'reason': 'total-signal estimate disagrees with the matched-'
                      'autofluorescence one (cosine 0.5486)',
            'primary_detector': 'YG1-A', 'separation_index': 2.1,
            'n_events': 5000, 'n_positives': 480, 'agreement': 0.5486,
            'af_cosine': 0.2, 'warning': None},
           {'fluor': 'FL2', 'method': 'matched', 'requested_method': 'auto',
            'reason': 'x', 'primary_detector': 'R1-A',
            'separation_index': 0.1, 'n_events': 5000, 'n_positives': 30,
            'agreement': 0.6, 'af_cosine': 0.8,
            'warning': 'the spectrum is not reproducible'}]
    root = _root_or_skip()
    try:
        qc = unmixing_qc({}, S_SIM[[0, 1, 3]], ['FL4', 'FL5', 'FL2'],
                         reference_spectra=ref)
        win = SpectralQCWindow(root, qc)
        assert '[!] 1 control(s) too dim' in _header(win)
        summary = win._summary_text()
        assert 'Reference spectra ([!] too dim: FL2)' in summary
        assert '• FL5: matched autofluorescence (primary YG1-A' in summary
        assert '• FL4: total signal (primary B2-A' in summary
        assert 'Use beads or a brighter fluor' in summary
        md = win._markdown()
        assert '## Reference spectra' in md
        assert '| FL5 | matched | YG1-A | 2.10 | 480 | 0.5486 |' in md
        assert '- **FL2**: the spectrum is not reproducible' in md
        # Without a report nothing is added.
        plain = SpectralQCWindow(root, unmixing_qc({}, S_SIM, FL_SIM))
        assert 'Reference spectra' not in plain._summary_text()
        assert 'too dim' not in _header(plain)
    finally:
        root.destroy()
