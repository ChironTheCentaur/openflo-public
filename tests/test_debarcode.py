"""Barcode deconvolution (debarcode.py) and Tools > Debarcode... (ui_debarcode).

Ground truth is built in: every pooled sample is synthetic, each event drawn
from one known sample's code (a 3-of-6 scheme with all 20 codes, or a mixed
key), on an arcsinh-like scale where a positive channel sits near 4 and a
negative one near 1. Separations are checked against values worked by hand.
"""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import pytest

from openflo import debarcode as dbc
from tests.test_transform_records import _new_editor, _pump, capture_warnings, isolate_home

BC = [f'Mass{101 + j}' for j in range(6)]
CODES_3OF6 = [c for c in itertools.product((0, 1), repeat=6) if sum(c) == 3]
SAMPLES = [f'Sample{i + 1:02d}' for i in range(20)]


def _key():
    return dbc.BarcodeKey(tuple(SAMPLES), tuple(BC), np.array(CODES_3OF6))


def _pool(n_per=300, seed=0, codes=None, pos=4.0, neg=1.0, sd=0.35,
          dim=None):
    """(values events x channels, truth sample index per event). ``dim``
    ({sample index: factor}) scales that sample's positive level."""
    codes = np.array(CODES_3OF6 if codes is None else codes)
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(len(codes)), n_per)
    level = np.full(len(codes), pos)
    for i, f in (dim or {}).items():
        level[i] = neg + (pos - neg) * f
    on = codes[truth] == 1
    x = np.where(on, level[truth][:, None], neg) + rng.normal(0, sd,
                                                             on.shape)
    perm = rng.permutation(len(truth))
    return x[perm], truth[perm]


# ── Key ─────────────────────────────────────────────────────────────────────

def _key_text(rows, head=('Sample', *BC[:3])):
    return '\n'.join([','.join(head)] + [','.join(r) for r in rows]) + '\n'


def test_read_key_reads_names_channels_and_codes(tmp_path):
    p = tmp_path / 'key.csv'
    p.write_text(_key_text([('SampleA', '1', '1', '0'), ('SampleB', '0', '1',
                                                         '1'), ('', '', '', '')]))
    key = dbc.read_key(str(p))
    assert key.samples == ('SampleA', 'SampleB')
    assert key.channels == tuple(BC[:3])
    assert key.codes.tolist() == [[1, 1, 0], [0, 1, 1]]
    assert key.uniform_k == 2
    assert key.code_text(1) == '011'


@pytest.mark.parametrize('rows, head, msg', [
    ([('SampleA', '1', '1', '0'), ('SampleB', '1', '1', '0')], None,
     'share a barcode code'),
    ([('SampleA', '1', '2', '0')], None, "'2' is not 0 or 1"),
    ([('SampleA', '1', 'x', '0')], None, "'x' is not 0 or 1"),
    ([('SampleA', '1', '1', '0'), ('SampleA', '0', '1', '1')], None,
     'used twice'),
    ([('SampleA', '1', '0')], None, 'has 3 column(s)'),
    ([('Unassigned', '1', '1', '0')], None, 'kept for the events'),
    ([(' ', '1', '1', '0')], None, 'Sample name missing'),
    ([('SampleA', '1')], ('Sample', 'Mass101'), 'at least two'),
    ([('SampleA', '1', '1', '0')], ('Sample', 'Mass101', 'Mass101',
                                    'Mass102'), 'listed twice'),
    ([], None, 'header row'),
])
def test_a_bad_key_is_refused_with_a_clear_reason(rows, head, msg):
    text = _key_text(rows, head or ('Sample', *BC[:3]))
    with pytest.raises(dbc.BarcodeKeyError, match=None) as ei:
        dbc.read_key(text)
    assert msg in str(ei.value), str(ei.value)


def test_a_key_channel_missing_from_the_data_is_named():
    key = _key()
    x, _ = _pool(n_per=5)
    df = pd.DataFrame(x[:, :5], columns=BC[:5])
    with pytest.raises(dbc.BarcodeKeyError, match='Mass106'):
        dbc.debarcode(df, key)


def test_an_unreadable_key_file_says_so(tmp_path):
    with pytest.raises(dbc.BarcodeKeyError, match='could not be read'):
        dbc.read_key(str(tmp_path / 'missing.csv'))


# ── Rescaling and separation, worked by hand ────────────────────────────────

def test_rescale_maps_the_1st_to_99th_percentile_onto_0_to_1():
    col = np.linspace(0.0, 100.0, 101)          # p1 = 1, p99 = 99
    out = dbc.rescale(np.column_stack([col, 2 * col]))
    assert out[10, 0] == pytest.approx((10 - 1) / 98)
    assert out[0, 0] == 0.0 and out[100, 0] == 1.0         # clipped
    assert out[10, 1] == pytest.approx((20 - 2) / 196)


def test_separation_in_a_k_of_n_scheme_is_the_gap_after_the_kth():
    x = np.array([[0.9, 0.2, 0.8, 0.1, 0.7, 0.0],
                  [0.5, 0.5, 0.45, 0.4, 0.0, 0.1]])
    pos, sep = dbc.separation(x, {3})
    assert pos[0].tolist() == [True, False, True, False, True, False]
    assert sep[0] == pytest.approx(0.7 - 0.2)
    assert pos[1].tolist() == [True, True, True, False, False, False]
    assert sep[1] == pytest.approx(0.45 - 0.4)


def test_separation_with_mixed_k_takes_the_largest_gap_among_the_keys_k():
    x = np.array([[0.9, 0.85, 0.1, 0.05],      # k=1: .05, k=2: .75
                  [0.95, 0.3, 0.2, 0.1],       # k=1: .65, k=2: .10
                  [0.9, 0.8, 0.75, 0.7]])      # k=1: .10, k=2: .05, k=4: .70
    pos, sep = dbc.separation(x, {1, 2})
    assert pos[0].tolist() == [True, True, False, False]
    assert sep[0] == pytest.approx(0.75)
    assert pos[1].tolist() == [True, False, False, False]
    assert sep[1] == pytest.approx(0.65)
    pos, sep = dbc.separation(x, {1, 2, 4})       # k = n: the gap down to 0
    assert pos[2].all() and sep[2] == pytest.approx(0.7)


def test_an_event_with_a_missing_value_is_unassigned():
    x, _ = _pool(n_per=20)
    x[3, 2] = np.nan
    res = dbc.debarcode(x, _key())
    assert np.isnan(res.separation[3])
    assert res.assigned[3] == -1 and res.labels()[3] == dbc.UNASSIGNED
    assert np.isfinite(np.delete(res.separation, 3)).all()


def test_a_barcode_channel_with_no_spread_is_refused():
    x, _ = _pool(n_per=20)
    x[:, 4] = 2.0
    with pytest.raises(ValueError, match='Mass105'):
        dbc.debarcode(pd.DataFrame(x, columns=BC), _key())


# ── Assignment against the truth ────────────────────────────────────────────

@pytest.mark.parametrize('second_pass', [False, True])
def test_a_3_of_6_pool_of_20_samples_is_split_correctly(second_pass):
    x, truth = _pool()
    res = dbc.debarcode(pd.DataFrame(x, columns=BC), _key(),
                        second_pass=second_pass)
    a = res.assigned
    assert (a == truth).mean() > 0.95
    assert ((a >= 0) & (a != truth)).sum() <= 2       # almost never wrong
    y = res.yields()
    assert list(y) == SAMPLES + [dbc.UNASSIGNED]
    for i, smp in enumerate(SAMPLES):
        assert y[smp] == int(((a == i)).sum())
        assert 270 <= y[smp] <= 300
    assert sum(y.values()) == len(truth)


def test_an_event_whose_code_is_in_no_sample_is_unassigned():
    codes = [(1, 1, 0, 0), (0, 0, 1, 1)]
    key = dbc.BarcodeKey(('SampleA', 'SampleB'), tuple(BC[:4]),
                         np.array(codes))
    x, truth = _pool(n_per=200, seed=11, codes=codes + [(1, 0, 1, 0)])
    res = dbc.debarcode(x, key, cutoff=0.0)
    stray = truth == 2
    assert (res.match[stray] == -1).all()
    assert (res.assigned[stray] == -1).all()
    assert (res.assigned[~stray] == truth[~stray]).mean() > 0.97
    assert res.yields()[dbc.UNASSIGNED] >= 200


def test_doublets_of_two_samples_are_left_unassigned():
    x, truth = _pool(n_per=100, seed=3)
    rng = np.random.default_rng(4)
    a_i, b_i = rng.integers(0, len(x), 300), rng.integers(0, len(x), 300)
    keep = truth[a_i] != truth[b_i]
    dbl = np.maximum(x[a_i[keep]], x[b_i[keep]])  # 4 to 6 positives
    res = dbc.debarcode(np.vstack([x, dbl]), _key(), second_pass=False)
    assert (res.assigned[len(x):] == -1).mean() > 0.9
    assert (res.assigned[:len(x)] == truth).mean() > 0.95


def test_a_mixed_key_is_split_correctly():
    codes = [(1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1),
             (1, 0, 1, 0)]
    key = dbc.BarcodeKey(tuple(SAMPLES[:5]), tuple(BC[:4]), np.array(codes))
    assert key.uniform_k is None
    x, truth = _pool(n_per=400, seed=5, codes=codes)
    res = dbc.debarcode(x, key)
    assert (res.assigned == truth).mean() > 0.95
    assert ((res.assigned >= 0) & (res.assigned != truth)).sum() <= 4


def test_the_cutoff_keeps_only_events_that_separate_at_least_that_well():
    x, truth = _pool(sd=0.8, seed=7)
    res = dbc.debarcode(x, _key(), cutoff=0.0)
    sep, match = res.separation, res.match
    for c in (0.0, 0.2, 0.3, 0.5, 0.8):
        r = res.with_cutoff(c)
        want = np.where(sep >= c, match, -1)
        np.testing.assert_array_equal(r.assigned, want)
        assert r.yields()[dbc.UNASSIGNED] == int((want == -1).sum())
    assert res.with_cutoff(1.01).yields()[dbc.UNASSIGNED] == len(x)
    # Raising the cutoff trades yield for purity.
    lo, hi = res.with_cutoff(0.0).assigned, res.with_cutoff(0.3).assigned
    assert (hi >= 0).sum() < (lo >= 0).sum()
    err = [((a >= 0) & (a != truth)).sum() / max((a >= 0).sum(), 1)
           for a in (lo, hi)]
    assert err[0] > 0.01 and err[1] < err[0] / 10


def test_yield_curve_counts_each_sample_at_each_cutoff():
    x, _ = _pool(sd=0.6, seed=8)
    res = dbc.debarcode(x, _key())
    cuts = [0.0, 0.25, 0.5]
    curve = res.yield_curve(cuts)
    assert list(curve.index) == cuts
    assert list(curve.columns) == SAMPLES + [dbc.UNASSIGNED]
    for c in cuts:
        for i, smp in enumerate(SAMPLES):
            assert curve.loc[c, smp] == int(((res.match == i)
                                             & (res.separation >= c)).sum())
    assert (curve.sum(axis=1) == len(x)).all()
    default = res.yield_curve()
    assert default.index[0] == 0.0 and default.index[-1] == 1.0


def test_rescaling_per_sample_recovers_a_dim_sample():
    """Sample 1 stained at 40 % of the others' intensity: on the global
    scale its positives sit low and many of its events fall under the
    cutoff; scaled by its own positives they separate like the rest."""
    x, truth = _pool(seed=9, dim={0: 0.4})
    one = dbc.debarcode(x, _key(), second_pass=False)
    two = dbc.debarcode(x, _key(), second_pass=True)
    n_one = int((one.assigned == 0).sum())
    n_two = int((two.assigned == 0).sum())
    assert n_two > n_one + 50, (n_one, n_two)
    assert ((two.assigned >= 0) & (two.assigned != truth)).sum() <= 3
    # The bright samples keep their yield.
    assert (two.assigned[truth == 5] == 5).mean() > 0.9


def test_the_methods_paragraph_names_debarcoding():
    from openflo.provenance import methods_paragraph
    text = methods_paragraph([{'action': 'debarcode', 'details': {
        'n_samples': 20, 'channels': BC, 'cutoff': 0.35,
        'second_pass': True}}])
    assert ('single-cell debarcoding (Zunder et al., 2015) into 20 samples '
            'on 6 barcode channels, with a separation cutoff of 0.35 after '
            'per-sample rescaling.') in text
    assert 'debarcoding' not in methods_paragraph([])


# ── The window ──────────────────────────────────────────────────────────────

def _fcs(path, seed=0, n_per=150):
    """A pooled FCS: FSC-A uniform 0..1000, six barcode channels (linear
    values whose arcsinh(x/5) is the pool's scale). Returns the truth."""
    import flowio
    x, truth = _pool(n_per=n_per, seed=seed)
    fsc = np.random.default_rng(seed + 1).uniform(0, 1000, len(truth))
    ev = np.column_stack([fsc, np.sinh(x) * 5.0])
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.astype(np.float32).flatten().tolist(),
                          ['FSC-A'] + BC,
                          opt_channel_names=['', 'BC1', '', '', '', '', ''])
    return truth


def _loaded(path):
    from openflo.pipeline import FlowSample
    s = FlowSample(path)
    s.apply_transform()
    return s


def _write_key(path, codes=CODES_3OF6, samples=SAMPLES, head=None):
    rows = [','.join(['Sample'] + list(head or BC))]
    rows += [','.join([smp] + [str(v) for v in c])
             for smp, c in zip(samples, codes, strict=True)]
    path.write_text('\n'.join(rows) + '\n')
    return str(path)


@pytest.fixture
def env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    yield root, _new_editor(root), tmp_path
    root.destroy()


def _pooled_editor(env, monkeypatch, key_path=None, head=None):
    """Editor holding the pooled sample 'pool' with a parent gate 'Big'
    (FSC-A >= 500), and the Debarcode window with the key loaded."""
    from tkinter import filedialog
    root, ed, tmp = env
    truth = _fcs(tmp / 'pool.fcs')
    s = _loaded(str(tmp / 'pool.fcs'))
    ed._on_loaded('pool', s)
    assert _pump(root, lambda: 'pool' in ed._samples)
    gid = ed._next_gate_id_for('pool')
    ed._sample_gates['pool'][gid] = {
        'kind': 'threshold', 'channel': 'FSC-A',
        'value': float(np.quantile(s.data['FSC-A'], 0.5)), 'name': 'Big',
        'parent_id': None, 'enabled': True, 'color': '#1f77b4'}
    ed._sample_gate_order['pool'].append(gid)
    kp = key_path or _write_key(tmp / 'key.csv', head=head)
    monkeypatch.setattr(filedialog, 'askopenfilename', lambda **k: kp)
    ed._open_debarcode_window()
    w = next(c for c in ed.winfo_children()
             if getattr(c, 'title', lambda: '')() == 'Debarcode')
    w._load_key()
    path = next(p for p, g in w._population_gids('pool').items() if g == gid)
    w.pop_var.set(path)
    return w, s, gid, truth


def _counts(ed, name):
    from openflo.pipeline import cumulative_gate_mask
    gates = ed._sample_gates[name]
    return {g['value']: int(np.asarray(cumulative_gate_mask(
        gates, gid, ed._samples[name].data)).sum())
        for gid, g in gates.items() if g.get('channel') == 'debarcode'}


def test_window_adds_each_barcoded_sample_under_the_chosen_population(
        env, monkeypatch):
    root, ed, _tmp = env
    w, s, parent, truth = _pooled_editor(env, monkeypatch)
    assert [v.get() for v in w._map_vars] == [ed._fmt_channel(c) for c in BC]
    assert w.cutoff_var.get() == '0.3'
    w._preview()
    rows = [w.tv.item(i, 'values') for i in w.tv.get_children()]
    assert len(rows) == 21 and rows[-1][0] == dbc.UNASSIGNED
    assert rows[0][1] == ''.join(map(str, CODES_3OF6[0]))
    # Nothing changed by Preview.
    assert 'debarcode' not in s.data.columns
    gids = w._apply()
    assert gids and len(gids) == 21
    gates = ed._sample_gates['pool']
    assert all(gates[g]['parent_id'] == parent for g in gids)
    assert [gates[g]['name'] for g in gids] == (
        [f'{x} (barcode)' for x in SAMPLES] + ['Unassigned (barcode)'])
    # Independently: debarcode the parent's events directly.
    big = s.data['FSC-A'].to_numpy() >= gates[parent]['value']
    ref = dbc.debarcode(s.data.loc[big, BC], _key()).yields()
    assert _counts(ed, 'pool') == ref
    assert int(rows[0][2].replace(',', '')) == ref[SAMPLES[0]]
    # Against the truth: the parent's events land in their own sample.
    lab = s.data['debarcode'].to_numpy()
    named = np.array(SAMPLES, dtype=object)[truth]
    assert (lab[big] == named[big]).mean() > 0.95
    assert (lab[~big] == '').all()
    # Statistics list them.
    rows, _cols = ed._collect_stats_rows({'Count'})
    stat = {r['Population'].rsplit('/', 1)[-1]: r['Count'] for r in rows
            if r['Sample'] == 'pool'}
    assert stat[f'{SAMPLES[3]} (barcode)'] == ref[SAMPLES[3]]
    # Audited for the Methods paragraph.
    from openflo.provenance import methods_paragraph
    assert 'cutoff of 0.3' in methods_paragraph(ed._audit_log)


def test_window_cutoff_and_reapply_update_the_populations(env, monkeypatch):
    root, ed, tmp = env
    w, s, parent, _truth = _pooled_editor(env, monkeypatch)
    w._apply()
    before = _counts(ed, 'pool')
    w.cutoff_var.set('0.6')
    w._cutoff_changed()
    first = w.tv.item(w.tv.get_children()[0], 'values')
    big = s.data['FSC-A'].to_numpy() >= ed._sample_gates['pool'][parent][
        'value']
    ref = dbc.debarcode(s.data.loc[big, BC], _key(), cutoff=0.6).yields()
    assert int(first[2].replace(',', '')) == ref[SAMPLES[0]]
    # Re-apply with fewer samples in the key and no Unassigned population.
    from tkinter import filedialog
    kp = _write_key(tmp / 'key2.csv', CODES_3OF6[:4], SAMPLES[:4])
    monkeypatch.setattr(filedialog, 'askopenfilename', lambda **k: kp)
    w._load_key()
    w.unassigned_var.set(False)
    gids = w._apply()
    assert len(gids) == 4
    after = _counts(ed, 'pool')
    assert set(after) == set(SAMPLES[:4])
    k4 = dbc.BarcodeKey(tuple(SAMPLES[:4]), tuple(BC),
                        np.array(CODES_3OF6[:4]))
    ref4 = dbc.debarcode(s.data.loc[big, BC], k4, cutoff=0.6).yields()
    assert after == {k: ref4[k] for k in SAMPLES[:4]}
    assert after[SAMPLES[0]] < before[SAMPLES[0]]      # higher cutoff


def test_window_maps_a_key_column_by_marker_label_and_asks_for_a_missing_one(
        env, monkeypatch):
    root, ed, tmp = env
    head = ['BC1'] + BC[1:5] + ['Mass999']
    w, s, _parent, _truth = _pooled_editor(env, monkeypatch, head=head)
    got = [v.get() for v in w._map_vars]
    assert got[0] == ed._fmt_channel('Mass101')
    assert got[5] == ''
    w._preview()
    assert "Pick the data channel for key column 'Mass999'" in \
        w.status_var.get()
    assert w._result is None
    w._map_vars[5].set(ed._fmt_channel('Mass106'))
    w._preview()
    assert w._result is not None
    w._map_vars[5].set(ed._fmt_channel('Mass105'))
    w._preview()
    assert 'same data channel' in w.status_var.get()


def test_window_refuses_a_bad_key_and_a_bad_cutoff(env, monkeypatch):
    root, ed, tmp = env
    bad = tmp / 'bad.csv'
    bad.write_text('Sample,Mass101,Mass102\nSampleA,1,0\nSampleB,1,0\n')
    w, _s, _p, _t = _pooled_editor(env, monkeypatch, key_path=str(bad))
    assert 'share a barcode code' in w.status_var.get()
    assert w._key is None
    from tkinter import filedialog
    kp = _write_key(tmp / 'key.csv')
    monkeypatch.setattr(filedialog, 'askopenfilename', lambda **k: kp)
    w._load_key()
    w.cutoff_var.set('1.5')
    assert w._apply() is None
    assert 'from 0 to 1' in w.status_var.get()
    assert 'debarcode' not in ed._samples['pool'].data.columns


def test_barcoded_populations_survive_a_session_round_trip(env, monkeypatch):
    from tests.test_derived_kinetics import _restore_session
    root, ed, tmp = env
    w, s, _parent, _truth = _pooled_editor(env, monkeypatch)
    assert not ed._has_computed_columns(s)
    w._apply()
    want = _counts(ed, 'pool')
    labels = s.data['debarcode'].to_numpy(copy=True)
    path = tmp / 'pool.flowsession'
    data = ed._write_session(str(path))
    assert data['samples'][0].get('processed_data')
    w.destroy()
    ed2, _jobs = _restore_session(root, path)
    s2 = ed2._samples['pool']
    np.testing.assert_array_equal(s2.data['debarcode'].to_numpy(), labels)
    assert _counts(ed2, 'pool') == want
    names = {g['name'] for g in ed2._sample_gates['pool'].values()}
    assert f'{SAMPLES[7]} (barcode)' in names
    assert (s2.result_stamps.get('debarcode') or {}).get('step') == \
        'debarcode'


def test_save_each_as_fcs_writes_one_file_per_sample(env, monkeypatch):
    from tkinter import filedialog

    import flowio

    from openflo.async_task import wait_for_job
    root, ed, tmp = env
    w, s, _parent, _truth = _pooled_editor(env, monkeypatch)
    out = tmp / 'split'
    out.mkdir()
    monkeypatch.setattr(filedialog, 'askdirectory', lambda **k: str(out))
    w._save_fcs()
    assert wait_for_job(ed._job_strip)
    assert _pump(root, lambda: 'Saved' in w.status_var.get())
    want = _counts(ed, 'pool')
    files = sorted(p.name for p in out.iterdir())
    assert len(files) == 21
    for smp in (SAMPLES[0], SAMPLES[19], dbc.UNASSIGNED):
        f = flowio.FlowData(str(out / f'{smp}.fcs'))
        assert f.event_count == want[smp]
    assert 'Not exported' not in w.status_var.get()


def test_debarcode_is_on_the_tools_menu_and_needs_a_sample(env):
    root, ed, _tmp = env
    tools = next(b.nametowidget(b.cget('menu')) for b in ed._menubar_buttons
                 if b.cget('text') == 'Tools')
    labels = [tools.entrycget(i, 'label') for i in range(tools.index('end') + 1)
              if tools.type(i) == 'command']
    assert 'Debarcode…' in labels
    opened = []
    ed._show_or_raise = lambda key, make: opened.append(key)
    ed._open_debarcode_window()
    assert opened == [] and 'Load the pooled sample' in ed.status_var.get()
