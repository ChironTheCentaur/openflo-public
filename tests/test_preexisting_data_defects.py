"""Wrong-data defects that predate the linear-values branch (2.6.1 has them
too), each pinned by a test that failed before its fix."""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_editor_tools import _loaded
from tests.test_transform_records import (
    capture_warnings,
    editor_with_samples,
    isolate_home,
)


@pytest.fixture
def ed_env(tmp_path, monkeypatch):
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)       # never a real (blocking) modal
    yield from editor_with_samples(tmp_path)


def _run_events(ed, tmp_path, name='expt_s1', **cfg_over):
    """A real workspace run for `name`; returns its _events.csv path."""
    from openflo import workspace as W
    cfg = dict(W.default_run_cfg(), method='flowsom', n_metaclusters=2,
               umap=False, trimap=False, max_events=0, **cfg_over)
    prep = W.prepare_run(ed, {'sample': name, 'trial': 'T'}, cfg)
    res = W.compute_run(prep, cfg, str(tmp_path / 'run'))
    assert res['ok'], res
    return res['events']


# ── (d) workspace source tags are not detectors ─────────────────────────────

def test_statistics_of_a_workspace_imported_sample(ed_env, tmp_path):
    """'Load in editor' of a run's _events.csv brings its __group__ /
    __sample__ source tags. They were classified as fluor channels, so the
    Statistics window asked for their median: "ValueError: could not convert
    string to float: '(ungrouped)'"."""
    root, ed, _ = ed_env
    s = ed._samples[ed._import_processed_csv(_run_events(ed, tmp_path))]
    assert '__group__' in s.data.columns and '__sample__' in s.data.columns
    assert '__group__' not in s.fluor_channels
    assert '__sample__' not in s.fluor_channels
    rows, cols = ed._collect_stats_rows({'Count', 'Median'},
                                        samples=[s.name])
    assert not any('__' in c for c in cols), cols


# ── (a) population FCS export: compensated linear values ────────────────────

CH = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
SPILL = '3,FL1-A,FL2-A,FL3-A,1,0.1,0,0,1,0.05,0,0,1'


def _wide_range_fcs(path, n=600):
    """Compensated on load ($SPILL), with FL1-A on a 2^20 range ($P3R) so
    the export has a non-default $PnR to carry."""
    import flowio
    rng = np.random.default_rng(11)
    ev = np.column_stack([
        rng.lognormal(10, .2, n), rng.lognormal(9, .2, n),
        rng.choice([150., 200000.], n) + rng.normal(0, 40, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.normal(800, 100, n)]).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), CH,
                          opt_channel_names=['', '', 'CD901b', 'CD902', 'CD903'],
                          metadata_dict={'SPILL': SPILL, 'P3R': '1048576'})
    return str(path)


def _read_fcs(path):
    import flowio
    f = flowio.FlowData(str(path))
    names = [f.text[f'p{i + 1}n'] for i in range(f.channel_count)]
    ev = np.reshape(np.asarray(f.events, dtype=float), (-1, f.channel_count))
    ranges = {names[i]: f.text.get(f'p{i + 1}r') for i in range(len(names))}
    return dict(zip(names, ev.T, strict=True)), ranges


@pytest.mark.parametrize('how', ['bulk', 'single'])
def test_population_fcs_holds_compensated_linear_values(
        ed_env, tmp_path, monkeypatch, how):
    """Both population exports wrote what the editor DISPLAYS (bulk:
    s.data, logicle coordinates; single: s.data, or the UNcompensated raw
    detectors with no $SPILL), so a re-import transformed them again. They
    must hold the compensated linear values, each parameter's $PnR from the
    source file, and re-import to the same display values."""
    from openflo import editor_populations
    from openflo.pipeline import (
        FlowSample,
        linear_values,
        transform_values,
    )
    root, ed, _ = ed_env
    s = _loaded(_wide_range_fcs(tmp_path / 'wide.fcs'))
    s.data['cell_cycle'] = 'G1'                  # text: has no FCS parameter
    ed._on_loaded('wide', s)
    ed._set_active_sample('wide')
    cut = float(transform_values(np.array([2000.0]), method='logicle')[0])
    gid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                        'value': cut, 'op': '>=', 'parent_id': None,
                        'name': 'CD901b+', 'enabled': True})
    mask = s.data['FL1-A'].to_numpy(float) >= cut
    out = tmp_path / 'export'
    out.mkdir()
    monkeypatch.setattr(editor_populations.messagebox, 'showerror',
                        lambda *a, **k: pytest.fail(f'export failed: {a}'))
    if how == 'bulk':
        monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                            lambda **k: str(out))
        ed._export_populations_fcs()
        path, = out.glob('*.fcs')
    else:
        path = out / 'one.fcs'
        monkeypatch.setattr(editor_populations.filedialog,
                            'asksaveasfilename', lambda **k: str(path))
        ed._export_population_fcs('wide', gid)
    cols, ranges = _read_fcs(path)
    assert 'cell_cycle' not in cols
    assert ranges['FL1-A'] == '1048576' and ranges['FL2-A'] == '262144'
    for ch in ('FL1-A', 'FL2-A', 'FL3-A', 'FSC-A'):
        np.testing.assert_allclose(cols[ch], linear_values(s, ch)[mask],
                                   rtol=1e-6, atol=1e-3, err_msg=ch)
    # The source has a $SPILL; the export must not, or every re-import
    # compensates the compensated values again.
    import flowio
    text = flowio.FlowData(str(path)).text
    assert 'spill' in {k.lower().lstrip('$') for k in s.metadata}
    assert not [k for k in text
                if any(w in k.lower() for w in ('spill', 'comp'))], text
    again = FlowSample(str(path))                # what a re-import shows,
    again.auto_compensate()                      # compensation included
    again.apply_transform()
    for ch in ('FL1-A', 'FL2-A', 'FL3-A'):
        np.testing.assert_allclose(again.data[ch].to_numpy(float),
                                   s.data[ch].to_numpy(float)[mask],
                                   rtol=1e-5, atol=1e-6, err_msg=ch)


def test_each_bulk_exported_population_gets_its_own_ranges(ed_env, tmp_path,
                                                           monkeypatch):
    """A column with no source $PnR gets one from its data. Bulk export
    handed every file the LAST population's: measured, Bright.fcs carried
    MESF:CD902 $PnR 262144 with values up to 3.0e6 (single export: 4194304)."""
    from openflo import editor_populations
    from openflo.pipeline import transform_values
    root, ed, _ = ed_env
    s = _loaded(_wide_range_fcs(tmp_path / 'wide.fcs'))
    cut = float(transform_values(np.array([2000.0]), method='logicle')[0])
    bright = s.data['FL1-A'].to_numpy(float) > cut
    s.data['MESF:CD902'] = np.where(bright, 3.0e6, 100.0)  # derived, linear
    ed._on_loaded('wide', s)
    ed._set_active_sample('wide')
    ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A', 'value': cut,
                  'parent_id': None, 'name': 'Bright', 'enabled': True})
    # Another channel: the editor keeps one threshold/interval per channel
    # and parent, so a second FL1-A gate would replace 'Bright'.
    ed._add_gate({'kind': 'interval', 'channel': 'MESF:CD902', 'lo': 0.0,
                  'hi': 1000.0, 'parent_id': None, 'name': 'Dim',
                  'enabled': True})                   # exported LAST
    out = tmp_path / 'bulk'
    out.mkdir()
    monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                        lambda **k: str(out))
    ed._export_populations_fcs()
    for label, top in (('Bright', 3.0e6), ('Dim', 100.0)):
        cols, ranges = _read_fcs(out / f'{label}.fcs')
        assert float(np.max(cols['MESF:CD902'])) == pytest.approx(top), label
        assert float(ranges['MESF:CD902']) >= top, (label, ranges['MESF:CD902'])
    assert _read_fcs(out / 'Bright.fcs')[1]['MESF:CD902'] == '4194304'
    assert _read_fcs(out / 'Dim.fcs')[1]['MESF:CD902'] == '262144'


KEYWORDS = {'TIMESTEP': '0.01', 'DATE': '02-OCT-2026', 'BTIM': '10:00:00',
            'ETIM': '10:05:00', 'CYT': 'Synthetic cytometer'}
VOLTAGES = {'FL1-A': '400', 'FL2-A': '520'}


def _keyword_fcs(path, n=600):
    """A source with a Time channel and the acquisition keywords a
    re-import (FlowJo, FCS Express) needs: $TIMESTEP for Time, $DATE,
    $BTIM / $ETIM, $CYT and per-detector $PnV."""
    import flowio
    rng = np.random.default_rng(13)
    ch = ['Time', 'FSC-A', 'SSC-A', 'FL1-A', 'FL2-A']
    ev = np.column_stack([
        np.linspace(0, 30000, n), rng.lognormal(10, .2, n),
        rng.lognormal(9, .2, n),
        rng.choice([150., 20000.], n) + rng.normal(0, 40, n),
        rng.normal(800, 100, n)]).astype(np.float32)
    md = dict(KEYWORDS)
    md.update({f'P{ch.index(c) + 1}V': v for c, v in VOLTAGES.items()})
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.flatten().tolist(), ch, metadata_dict=md)
    return str(path)


def _keywords(path):
    """{keyword: value} and {PnN: $PnV} as an FCS reader sees them."""
    import flowio
    t = flowio.FlowData(str(path)).text
    names = {i: t[f'p{i}n'] for i in range(1, int(t['par']) + 1)}
    return ({k: t.get(k.lower()) for k in KEYWORDS},
            {names[i]: t.get(f'p{i}v') for i in names})


@pytest.mark.parametrize('how', ['bulk', 'single'])
def test_population_fcs_carries_the_acquisition_keywords(
        ed_env, tmp_path, monkeypatch, how):
    """An exported population lost $TIMESTEP (so its Time channel had no
    unit), $DATE, $BTIM/$ETIM, $CYT and every $PnV. They come from the
    source file, where it has them; $SPILL must NOT (the values are
    already compensated)."""
    from openflo import editor_populations
    from openflo.pipeline import transform_values
    root, ed, _ = ed_env
    s = _loaded(_keyword_fcs(tmp_path / 'kw.fcs'))
    assert 'Time' in s.data.columns
    ed._on_loaded('kw', s)
    ed._set_active_sample('kw')
    cut = float(transform_values(np.array([2000.0]), method='logicle')[0])
    gid = ed._add_gate({'kind': 'threshold', 'channel': 'FL1-A',
                        'value': cut, 'parent_id': None, 'name': 'Pos',
                        'enabled': True})
    out = tmp_path / 'kw_out'
    out.mkdir()
    if how == 'bulk':
        monkeypatch.setattr(editor_populations.filedialog, 'askdirectory',
                            lambda **k: str(out))
        ed._export_populations_fcs()
        path, = out.glob('*.fcs')
    else:
        path = out / 'one.fcs'
        monkeypatch.setattr(editor_populations.filedialog,
                            'asksaveasfilename', lambda **k: str(path))
        ed._export_population_fcs('kw', gid)
    kw, volts = _keywords(path)
    assert kw == KEYWORDS, kw
    assert {c: volts.get(c) for c in VOLTAGES} == VOLTAGES, volts
    assert volts.get('FSC-A') is None              # none in the source
    import flowio
    assert not [k for k in flowio.FlowData(str(path)).text
                if 'spill' in k.lower()]


# ── (b) a member reloaded from file is gated on the editor's transforms ─────

def test_a_reloaded_member_is_gated_on_the_editors_transforms(ed_env,
                                                               tmp_path):
    """A workspace item whose sample is not loaded is reloaded from its FCS,
    which the loader puts on logicle. Its gate was drawn in the editor's
    coordinates -- asinh here -- and was evaluated on the logicle values
    BEFORE prepare_unit aligned the transforms, so it selected the wrong
    events."""
    from openflo import workspace as W
    from openflo.pipeline import FlowSample
    from tests.test_editor_tools import FL, SPILL_M, _fcs
    root, ed, _ = ed_env
    ed._apply_channel_transforms({'FL1-A': 'asinh'})
    p3 = _fcs(tmp_path / 'expt_s3.fcs', 3)                 # never loaded
    cut = float(np.arcsinh(2000 / 150))                     # asinh coords
    gates = {'g1': {'kind': 'threshold', 'channel': 'FL1-A', 'value': cut,
                    'op': '>=', 'parent_id': None, 'name': 'CD901b+',
                    'enabled': True}}
    item = {'sample': 'expt_s3', 'path': p3, 'gate_id': 'g1', 'gates': gates,
            'comp_matrix': SPILL_M, 'comp_channels': FL}
    # The truth: the same reload, on the editor's transforms, then gated.
    want = FlowSample(p3)
    want.manual_compensate(SPILL_M, FL)
    want.apply_transform()
    want.retransform(['FL1-A'], method='asinh')
    n_true = int((want.data['FL1-A'].to_numpy(float) >= cut).sum())
    s, note = W.resolve_run_sample(ed, item)
    assert 0 < n_true < len(want.data)
    assert len(s.data) == n_true, (len(s.data), n_true, note)
    assert s.data_transforms['FL1-A']['method'] == 'asinh'


# ── (c) differential expression on linear values ────────────────────────────

def _loader_like(seed, cd901b_median, cd3_median, n=2000):
    """A sample as the editor holds it: linear values, then the loader's
    logicle on the fluor channels (recorded)."""
    import pandas as pd

    from openflo.pipeline import FlowSample
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'FSC-A': rng.normal(8e4, 8e3, n),
        'CD901b-A': cd901b_median * rng.lognormal(0.0, 0.3, n),
        'CD3-A': cd3_median + rng.normal(0.0, 20.0, n)})   # a dim, negative one
    s = FlowSample.from_dataframe(df, name=f's{seed}')
    s.apply_transform(channels=['CD901b-A', 'CD3-A'])
    return s


def test_marker_fold_change_is_on_linear_values():
    """log2FC of a marker between groups: per-sample medians of the DISPLAY
    values gave 0.51 for a 10x difference (log2 10 = 3.32). On linear
    values it is right; a negative linear median has no fold change (NaN,
    not a log of negatives that looks like a result)."""
    from openflo.diffexp import differential_test, marker_expression
    a = [_loader_like(i, 20000.0, -50.0) for i in (1, 2, 3)]
    b = [_loader_like(i, 2000.0, -500.0) for i in (4, 5, 6)]
    ga, gb = marker_expression(a, b, ['CD901b-A', 'CD3-A'])
    rows = {r['feature']: r for r in differential_test(ga, gb)}
    # B over A (2000 / 20000): the direction of every OpenFlo fold change.
    assert rows['CD901b-A']['log2fc'] == pytest.approx(-np.log2(10.0), abs=0.05)
    assert rows['CD901b-A']['mean_a'] == pytest.approx(20000.0, rel=0.05)
    assert np.isnan(rows['CD3-A']['log2fc'])
    assert rows['CD3-A']['mean_a'] == pytest.approx(-50.0, abs=5.0)
