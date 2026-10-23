"""After a FlowJo import, the FlowJo-form GeoMean is taken on the workspace's
axis, and the GeoMean column says which axis it used.

FlowJo's GeoMean is the mean of the events on the channel's display axis,
converted back to intensity, so it depends on that axis. OpenFlo loads every
sample on its own default logicle (W=0.5, M=4.5); a workspace often declares
another one per channel (here W=1, M=4.4185). On the default axis a dim
population's GeoMean read well below the workspace's.

Synthetic data and channel names only. The expected values are computed
independently of OpenFlo: logicle from its published definition (Parks et
al. 2006; Moore & Parks 2012) in numpy, inverse in closed form and forward
by bisection, and a plain mean.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from openflo import pipeline as fp
from openflo.gating import population_stats

T = 262144.0
WSP_AXIS = {'kind': 'logicle', 'T': T, 'W': 1.0, 'M': 4.4185, 'A': 0.0}
STAT_CHAN = ('Median', 'Mean', 'GeoMean', 'CV', 'rCV')
FL = ['FL1-A', 'FL2-A']
N = 20000


# ── an independent logicle ──────────────────────────────────────────────────

def _coef(t, w_, m, a_):
    w = w_ / (m + a_)
    x2 = a_ / (m + a_)
    x1, x0 = x2 + w, x2 + 2 * w
    b = (m + a_) * np.log(10.0)
    d = brentq(lambda d: 2 * (np.log(d) - np.log(b)) + w * (b + d), 1e-12, b)
    ca = np.exp(x0 * (b + d))
    mfa = np.exp(b * x1) - ca / np.exp(d * x1)
    a = t / ((np.exp(b) - mfa) - ca / np.exp(d))
    return a, b, ca * a, d, -mfa * a, x1


def _inv(y, t, w, m, a_):
    a, b, c, d, f, x1 = _coef(t, w, m, a_)
    y = np.asarray(y, float)

    def s(v):
        return a * np.exp(b * v) - c * np.exp(-d * v) + f
    return np.where(y >= x1, s(y), -s(2 * x1 - y))


def _fwd(x, t, w, m, a_):
    x = np.asarray(x, float)
    lo, hi = np.full(x.shape, -2.0), np.full(x.shape, 2.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = _inv(mid, t, w, m, a_) < x
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def _geomean_on(x, t, w, m, a_):
    """FlowJo's GeoMean of `x` on a logicle axis: mean there, back."""
    return float(_inv(np.mean(_fwd(x, t, w, m, a_)), t, w, m, a_))


def _dim(seed=0, n=N):
    """A dim population that compensation spreads around zero, and a bright
    one."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.normal(150.0, 250.0, n),
                            rng.lognormal(np.log(20000.0), 0.5, n)])


# ── population_stats ─────────────────────────────────────────────────────────

def _stored(lin):
    spec = fp.transform_spec('logicle')
    df = pd.DataFrame({c: fp.transform_values(lin[:, i], **spec)
                       for i, c in enumerate(FL)})
    return df, {c: spec for c in FL}


ALL = {'g1': {'id': 'g1', 'kind': 'threshold', 'parent_id': None,
              'channel': 'FL1-A', 'op': '>', 'value': -np.inf,
              'name': 'all', 'enabled': True}}


def _row(df, transforms, **kw):
    return population_stats('s', df, ALL, ['g1'], {}, FL, {'GeoMean'},
                            STAT_CHAN, transforms=transforms,
                            stat_forms={'geomean': 'display'}, **kw)[0]


def test_geomean_uses_the_workspace_axis_for_the_channels_it_declares():
    lin = _dim()
    df, tr = _stored(lin)
    row = _row(df, tr, stat_axes={'FL1-A': WSP_AXIS})
    want = _geomean_on(lin[:, 0], T, 1.0, 4.4185, 0.0)
    default = _geomean_on(lin[:, 0], T, 0.5, 4.5, 0.0)
    assert row['GeoMean FL1-A'] == pytest.approx(want, rel=1e-6)
    # The axis matters: the default axis gives another number.
    assert abs(want - default) / want > 0.10, (want, default)
    # FL2-A declares no axis: OpenFlo's own, as before.
    assert row['GeoMean FL2-A'] == pytest.approx(
        _geomean_on(lin[:, 1], T, 0.5, 4.5, 0.0), rel=1e-6)
    axes = row['__geomean_axis__']
    assert axes['FL1-A'] == 'FlowJo axis: logicle W=1 M=4.42'
    assert axes['FL2-A'] == 'OpenFlo display axis'


def test_a_sample_without_a_workspace_axis_is_unchanged():
    lin = _dim(1)
    df, tr = _stored(lin)
    row = _row(df, tr)
    for i, c in enumerate(FL):
        assert row[f'GeoMean {c}'] == pytest.approx(
            _geomean_on(lin[:, i], T, 0.5, 4.5, 0.0), rel=1e-6)
    assert set(row['__geomean_axis__'].values()) == {'OpenFlo display axis'}


def test_the_textbook_form_ignores_the_axis():
    lin = _dim(2)
    df, tr = _stored(lin)
    row = population_stats('s', df, ALL, ['g1'], {}, FL, {'GeoMean'},
                           STAT_CHAN, transforms=tr,
                           stat_forms={'geomean': 'log'},
                           stat_axes={'FL1-A': WSP_AXIS})[0]
    pos = lin[:, 0][lin[:, 0] > 0]
    assert row['GeoMean FL1-A'] == pytest.approx(
        float(np.exp(np.mean(np.log(pos)))), rel=1e-6)
    assert '__geomean_axis__' not in row


@pytest.mark.parametrize('spec, text', [
    (WSP_AXIS, 'FlowJo axis: logicle W=1 M=4.42'),
    ({'kind': 'logicle', 'T': 1e5, 'W': 0.5, 'M': 4.5, 'A': 1.0},
     'FlowJo axis: logicle T=100000 W=0.5 M=4.5 A=1'),
    ({'kind': 'biex', 'neg': 0.0, 'width': -10.0, 'pos': 4.42,
      'maxRange': T}, 'FlowJo axis: biex width=-10 pos=4.42'),
    ({'kind': 'log', 'offset': 1.0, 'decades': 5.0},
     'FlowJo axis: log 5 decades'),
    ({'kind': 'fasinh', 'T': T, 'M': 4.0, 'A': 0.0},
     'FlowJo axis: arcsinh T=262144 M=4 A=0'),
])
def test_the_axis_is_named(spec, text):
    from openflo.gating import geomean_axis_text
    assert geomean_axis_text(spec) == text


# ── reading the axes from a workspace ───────────────────────────────────────

def _axis_xml(param, kind='logicle', w=1.0, m=4.4185):
    if kind == 'logicle':
        return (f'<transforms:logicle transforms:length="256" '
                f'transforms:T="{T:g}" transforms:W="{w}" transforms:M="{m}" '
                f'transforms:A="0"><data-type:parameter '
                f'data-type:name="{param}"/></transforms:logicle>')
    return ('<transforms:linear transforms:minRange="0" '
            'transforms:maxRange="262144" gain="1"><data-type:parameter '
            f'data-type:name="{param}"/></transforms:linear>')


def _wsp(path, fcs, axes):
    """A FlowJo 10 style workspace for `fcs` with one gate on FL1-A and the
    <Transformations> entries `axes`."""
    path.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<Workspace version="20.0" flowJoVersion="10.8.1"
  xmlns:gating="http://www.isac-net.org/std/Gating-ML/v2.0/gating"
  xmlns:transforms="http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
  xmlns:data-type="http://www.isac-net.org/std/Gating-ML/v2.0/datatypes">
 <SampleList><Sample>
  <DataSet uri="{fp._wsp_dataset_uri(fcs)}" sampleID="1"/>
  <Transformations>{''.join(axes)}</Transformations>
  <SampleNode name="s1" sampleID="1" count="{N}"><Subpopulations>
   <Population name="Any" count="0">
    <Gate gating:id="ID1"><gating:RectangleGate>
     <gating:dimension gating:min="-100000"><data-type:fcs-dimension data-type:name="FL1-A"/></gating:dimension>
    </gating:RectangleGate></Gate>
   </Population>
  </Subpopulations></SampleNode>
 </Sample></SampleList>
</Workspace>''', encoding='utf-8')
    return str(path)


def _fcs(path, seed=0):
    import flowio
    ev = _dim(seed).astype(np.float32)
    with open(path, 'wb') as fh:
        flowio.create_fcs(fh, ev.ravel().tolist(), FL)
    return str(path), ev.astype(float)


def _sample_elem(wsp):
    return next(fp.WspReader(wsp).root.iter('Sample'))


def test_the_workspace_axes_are_read_per_channel(tmp_path):
    fcs, _ev = _fcs(tmp_path / 'a.fcs')
    wsp = _wsp(tmp_path / 'a.wsp', fcs, [
        _axis_xml('FL1-A', w=0.5, m=4.5),          # uncompensated parameter
        _axis_xml('Comp-FL1-A'),                   # the data column's
        _axis_xml('Comp-FL2-A', kind='linear')])
    axes = fp.wsp_stat_axes(_sample_elem(wsp))
    assert axes == {'FL1-A': WSP_AXIS}             # linear: OpenFlo's own


def test_a_wsp_template_brings_its_axes(tmp_path):
    fcs, _ev = _fcs(tmp_path / 'a.fcs')
    wsp = _wsp(tmp_path / 'a.wsp', fcs, [_axis_xml('Comp-FL1-A')])
    assert fp.wsp_template_stat_axes(wsp) == {'FL1-A': WSP_AXIS}


# ── the editor: import, Statistics heading, session round trip ──────────────

@pytest.fixture
def env(tmp_path, monkeypatch):
    from tests.test_transform_records import (
        _new_editor,
        capture_warnings,
        isolate_home,
    )
    isolate_home(monkeypatch, tmp_path)
    capture_warnings(monkeypatch)
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    import openflo.prefs as prefs
    monkeypatch.setattr(prefs, 'load_stat_forms',
                        lambda: {'geomean': 'display', 'rcv': 'percentile'})
    yield root, _new_editor(root), tmp_path
    root.destroy()


def _geomeans(ed, name):
    """The editor's statistics row (_sample_rows, as the Statistics window
    collects them) of `name`'s one population, which holds every event."""
    if not ed._sample_gates.get(name):
        ed._sample_gates[name] = {k: dict(g) for k, g in ALL.items()}
        ed._sample_gate_order[name] = list(ALL)
    row, = ed._sample_rows(name, {'GeoMean'})
    return row


def test_editor_wsp_import_uses_the_workspace_axis_and_keeps_it(env):
    from tests.test_session_restore_transforms import _restore
    from tests.test_transform_records import settle
    root, ed, tmp = env
    fcs, ev = _fcs(tmp / 'wsp.fcs')
    plain, ev_plain = _fcs(tmp / 'plain.fcs', seed=3)
    wsp = _wsp(tmp / 'w.wsp', fcs, [_axis_xml('Comp-FL1-A')])
    queued = []
    ed._queue_fcs_loads = lambda paths, **_k: queued.extend(paths)
    ed._ingest_wsp(wsp)
    assert queued == [fcs]
    for p in (fcs, plain):
        ed._load_worker(ed._sample_name_for(p), p)
        root.update()
    a, b = ed._sample_name_for(fcs), ed._sample_name_for(plain)
    assert ed._samples[a].stat_axes == {'FL1-A': WSP_AXIS}
    assert not getattr(ed._samples[b], 'stat_axes', None)

    want = _geomean_on(ev[:, 0], T, 1.0, 4.4185, 0.0)
    ra = _geomeans(ed, a)
    assert ra['GeoMean FL1-A'] == pytest.approx(want, rel=1e-6)
    assert ra['GeoMean FL2-A'] == pytest.approx(
        _geomean_on(ev[:, 1], T, 0.5, 4.5, 0.0), rel=1e-6)
    rb = _geomeans(ed, b)
    assert rb['GeoMean FL1-A'] == pytest.approx(
        _geomean_on(ev_plain[:, 0], T, 0.5, 4.5, 0.0), rel=1e-6)

    # The Statistics window and its export name the axis.
    from openflo.ui_statistics import StatisticsWindow
    win = StatisticsWindow.__new__(StatisticsWindow)
    win.editor = ed
    win._rows = [ra]
    assert win._heading('GeoMean FL1-A') == (
        'GeoMean FL1-A (linear, FlowJo axis: logicle W=1 M=4.42)')
    assert win._heading('GeoMean FL2-A') == 'GeoMean FL2-A (linear)'
    win._rows = [ra, rb]
    assert win._heading('GeoMean FL1-A') == (
        'GeoMean FL1-A (linear, axis differs by sample)')

    # Saved and reopened: the axis and the number come back.
    path = tmp / 'geo.flowsession'
    ed._write_session(str(path))
    ed2 = _restore(root, path, raw_first=True)
    settle(root, 0.5)
    assert ed2._samples[a].stat_axes == {'FL1-A': WSP_AXIS}
    assert not getattr(ed2._samples[b], 'stat_axes', None)
    assert _geomeans(ed2, a)['GeoMean FL1-A'] == pytest.approx(want, rel=1e-6)


def test_an_older_session_without_axes_still_opens(env):
    from tests.test_session_restore_transforms import _restore
    root, ed, tmp = env
    plain, ev = _fcs(tmp / 'plain.fcs', seed=4)
    ed._load_worker(ed._sample_name_for(plain), plain)
    root.update()
    path = tmp / 'old.flowsession'
    data = ed._write_session(str(path))
    for e in data['samples']:
        e.pop('stat_axes', None)
    path.write_text(json.dumps(data), encoding='utf-8')
    ed2 = _restore(root, path, raw_first=True)
    s = ed2._samples[ed2._sample_name_for(plain)]
    assert not getattr(s, 'stat_axes', None)
