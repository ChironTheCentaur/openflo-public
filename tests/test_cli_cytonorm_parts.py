"""cli_cytonorm.fit_cytonorm's parts: what each helper does, and the order
and arguments the fit passes them.

fit_cytonorm checks the control token, takes each batch's FCS paths
(_batch_paths), stops with fewer than two batches, loads every sample
(_load_batches), stops without two shared channels and two loaded batches,
picks the channels, takes the controls-mode references
(_control_references), fits the model and reports its score (_report_qc).

Synthetic FCS files only: scatter, four FL*-A channels with made-up marker
names, and for some files a Time channel with a flow-rate burst and
saturated events (acquisition QC removes some) and a $SPILLOVER matrix
(compensation changes the data).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openflo.batch_normalize as batch_normalize
import openflo.cli_cytonorm as cli_cytonorm
from openflo.cytonorm import cap_events_per_batch
from openflo.flow_sample import FlowSample
from tests._fcs_writer import write_float_fcs

CH = ['FSC-A', 'FSC-H', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']
MK = ['', '', '', 'mk1', 'mk2', 'mk3', 'mk4']
SPILL = ('4,FL1-A,FL2-A,FL3-A,FL4-A,1,0.2,0.05,0,0.1,1,0.2,0.05,0,0.1,1,'
         '0.2,0.02,0,0.1,1')


def _events(seed, n, gain=1.0):
    rng = np.random.default_rng(seed)
    a = rng.normal(1e5, 5e3, n)
    n1 = int(n * .6)
    ln = rng.lognormal
    f1 = np.r_[ln(np.log(20000), .5, n1), ln(np.log(150), .5, n - n1)]
    f2 = np.r_[ln(np.log(150), .5, n1), ln(np.log(20000), .5, n - n1)]
    fl = np.column_stack([f1, f2, ln(np.log(800), .5, n),
                          ln(np.log(3000), .7, n)]) * gain
    return np.column_stack([a, a / 1.2, rng.normal(9e3, 500, n), fl])


def _write(path, seed=0, n=600, gain=1.0, n_fluor=4, rich=False):
    """An FCS file at `path` with the first `n_fluor` FL channels; `rich`
    adds a Time channel QC trims and a $SPILLOVER matrix."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ev = _events(seed, n, gain)[:, :3 + n_fluor]
    ch, mk, kw = CH[:3 + n_fluor], MK[:3 + n_fluor], {}
    if rich:
        rng = np.random.default_rng(seed + 1000)
        ev[:n // 30, 3] = 262143.0
        t = np.sort(rng.uniform(0, 100, n))
        t[n // 2:n // 2 + n // 7] = t[n // 2]
        ev = np.column_stack([ev, t])
        ch, mk = ch + ['Time'], mk + ['']
        kw = {'metadata_dict': {'SPILLOVER': SPILL}}
    write_float_fcs(path, ev, ch, opt_channel_names=mk, **kw)
    return str(path)


def _bad(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'not an fcs file' * 10)
    return str(path)


def _expected_frame(path, qc=False):
    s = FlowSample(path)
    if qc:
        s.run_qc()
    s.auto_compensate()
    s.apply_transform()
    return s


# ── _batch_paths ──────────────────────────────────────────────────────────

def test_batch_paths_keeps_found_samples_per_batch_in_order(tmp_path):
    d0, d3 = tmp_path / 'day0', tmp_path / 'day3'
    p = {nm: _write(d / f'{nm}.fcs') for d, nms in ((d0, ['a1', 'a2']),
                                                    (d3, ['b1']))
         for nm in nms}
    _write(tmp_path / 'top.fcs')
    groups = [
        {'trial_dir': str(d3), 'samples': ['b1', 'gone']},
        {'trial_dir': str(d0),
         'samples': ['', None, {'name': ''}, {'other': 1}, {'name': 'a2'},
                     'a1']},
        {'samples': ['top']},                     # no trial_dir: the default
        {'trial_dir': '', 'samples': ['top']},    # empty: the default too
    ]
    got = cli_cytonorm._batch_paths(groups, str(tmp_path), 'goal', '', '')
    assert list(got) == [str(d3), str(d0), str(tmp_path)]
    assert got[str(d3)] == [p['b1']]
    assert got[str(d0)] == [p['a2'], p['a1']]
    assert [os.path.basename(x) for x in got[str(tmp_path)]] == ['top.fcs'] * 2


def test_batch_paths_takes_names_that_are_not_strings(tmp_path):
    p = _write(tmp_path / 'b' / '7.fcs')
    got = cli_cytonorm._batch_paths(
        [{'trial_dir': str(tmp_path / 'b'), 'samples': [7]}], '.',
        'controls', '7', '7')
    assert got == {str(tmp_path / 'b'): [p]}


def test_batch_paths_skips_a_falsy_name_even_with_a_file_for_it(tmp_path):
    _write(tmp_path / 'b' / '0.fcs')
    p = _write(tmp_path / 'b' / '1.fcs')
    got = cli_cytonorm._batch_paths(
        [{'trial_dir': str(tmp_path / 'b'),
          'samples': [0, {'name': 0}, 1]}], '.', 'goal', '', '')
    assert got == {str(tmp_path / 'b'): [p]}


def test_batch_paths_controls_keep_only_names_with_the_token(tmp_path):
    d0, d3 = tmp_path / 'day0', tmp_path / 'day3'
    paths = {nm: _write(d / f'{nm}.fcs') for d, nms in (
        (d0, ['Ctl0', 'a2']), (d3, ['b1', 'xCTLx', 'ctl3']))
        for nm in nms}
    groups = [{'trial_dir': str(d0), 'samples': ['Ctl0', 'a2']},
              {'trial_dir': str(d3), 'samples': ['b1', 'xCTLx', 'ctl3']}]
    got = cli_cytonorm._batch_paths(groups, '.', 'controls', 'ctl', ' CTL ')
    assert got == {str(d0): [paths['Ctl0']],
                   str(d3): [paths['xCTLx'], paths['ctl3']]}
    # The token is matched as given (the caller lower-cases it), against
    # the lower-cased name, so an upper-case token matches nothing.
    with pytest.raises(ValueError, match=r"'CTL' in batch\(es\) day3, day0;"):
        cli_cytonorm._batch_paths(
            [{'trial_dir': str(d3), 'samples': ['xCTLx']},
             {'trial_dir': str(d0), 'samples': ['Ctl0']}],
            '.', 'controls', 'CTL', 'CTL')


def test_batch_paths_controls_needs_one_in_every_batch_with_samples(tmp_path):
    d0, d3, d5 = tmp_path / 'day0', tmp_path / 'day3', tmp_path / 'day5'
    _write(d0 / 'ctl0.fcs')
    _write(d3 / 'b1.fcs')
    _write(d5 / 'c1.fcs')
    groups = [{'trial_dir': str(d0), 'samples': ['ctl0']},
              {'trial_dir': str(d3), 'samples': ['b1']},
              {'trial_dir': str(d5), 'samples': ['c1']},
              {'trial_dir': str(tmp_path / 'empty'), 'samples': ['x']}]
    with pytest.raises(ValueError) as e:
        cli_cytonorm._batch_paths(groups, '.', 'controls', 'ctl', ' ctl ')
    assert str(e.value) == (
        "--cytonorm-mode controls: no sample containing 'ctl' in batch(es) "
        "day3, day5; 'controls' mode needs the reference run in every batch")
    # Goal mode keeps every batch, so none lacks a sample.
    got = cli_cytonorm._batch_paths(groups, '.', 'goal', 'ctl', 'ctl')
    assert list(got) == [str(d0), str(d3), str(d5)]


def test_batch_paths_names_a_batch_by_its_path_when_it_has_no_basename(
        tmp_path):
    _write(tmp_path / 'b1.fcs')
    root = str(tmp_path) + os.sep
    with pytest.raises(ValueError, match=f"batch\\(es\\) {root}; "):
        cli_cytonorm._batch_paths([{'samples': ['b1']}], root, 'controls',
                                  'ctl', 'ctl')


# ── _load_batches ─────────────────────────────────────────────────────────

def test_load_batches_compensates_then_transforms_each_sample(tmp_path):
    a = _write(tmp_path / 'day0' / 'a1.fcs', seed=1, rich=True)
    b = _write(tmp_path / 'day3' / 'b1.fcs', seed=2, gain=1.4, rich=True)
    said = []
    frames, batch_of, shared, ref = cli_cytonorm._load_batches(
        {str(tmp_path / 'day0'): [a], str(tmp_path / 'day3'): [b]}, False,
        said.append)
    ka = f'{tmp_path / "day0"}{os.sep}a1.fcs'
    kb = f'{tmp_path / "day3"}{os.sep}b1.fcs'
    assert list(frames) == [ka, kb]
    assert batch_of == {ka: str(tmp_path / 'day0'), kb: str(tmp_path / 'day3')}
    for k, p in ((ka, a), (kb, b)):
        pd.testing.assert_frame_equal(frames[k], _expected_frame(p).data,
                                      check_exact=True)
    assert shared == {'FL1-A', 'FL2-A', 'FL3-A', 'FL4-A'}
    assert ref.name == FlowSample(a).name and len(ref.data) == len(frames[ka])
    assert said == []


def test_load_batches_runs_qc_only_when_asked(tmp_path, monkeypatch):
    a = _write(tmp_path / 'd' / 'a1.fcs', n=900, rich=True)
    seen = []
    real = FlowSample.run_qc

    def spy(self, *args, **kw):
        seen.append(self.name)
        return real(self, *args, **kw)
    monkeypatch.setattr(FlowSample, 'run_qc', spy)
    key = f'{tmp_path / "d"}{os.sep}a1.fcs'
    plain = cli_cytonorm._load_batches({str(tmp_path / 'd'): [a]}, False,
                                       print)[0][key]
    assert seen == []
    qcd = cli_cytonorm._load_batches({str(tmp_path / 'd'): [a]}, True,
                                     print)[0][key]
    assert len(seen) == 1
    assert len(qcd) < len(plain) == 900
    pd.testing.assert_frame_equal(qcd, _expected_frame(a, qc=True).data,
                                  check_exact=True)


def test_load_batches_reports_and_skips_a_bad_file(tmp_path):
    bad = _bad(tmp_path / 'd0' / 'bad.fcs')
    ok = _write(tmp_path / 'd0' / 'ok.fcs')
    said = []
    frames, _batch_of, shared, ref = cli_cytonorm._load_batches(
        {str(tmp_path / 'd0'): [bad, ok]}, False, said.append)
    assert len(said) == 1
    assert said[0].startswith('[batch-correct] skip bad.fcs: ')
    assert list(frames) == [f'{tmp_path / "d0"}{os.sep}ok.fcs']
    assert ref is not None and len(shared) == 4
    assert cli_cytonorm._load_batches({'x': [bad]}, False, said.append) == (
        {}, {}, None, None)


def test_load_batches_shares_only_the_channels_every_sample_has(tmp_path):
    a = _write(tmp_path / 'd0' / 'a.fcs', n_fluor=4)
    b = _write(tmp_path / 'd3' / 'b.fcs', n_fluor=2)
    c = _write(tmp_path / 'd3' / 'c.fcs', n_fluor=3)
    _f, batch_of, shared, ref = cli_cytonorm._load_batches(
        {Path(tmp_path / 'd3'): [b, c], Path(tmp_path / 'd0'): [a]}, False,
        print)
    assert shared == {'FL1-A', 'FL2-A'}
    assert list(ref.data.columns) == list(FlowSample(b).data.columns)
    # A batch given as a Path is recorded as its string.
    assert set(batch_of.values()) == {str(tmp_path / 'd3'), str(tmp_path / 'd0')}
    assert all(type(v) is str for v in batch_of.values())


# ── _control_references ───────────────────────────────────────────────────

def test_control_references_group_keys_by_batch_in_controls_mode_only():
    batch_of = {'x/1': 'x', 'y/1': 'y', 'x/2': 'x'}
    assert cli_cytonorm._control_references('goal', batch_of) is None
    assert cli_cytonorm._control_references('controls', batch_of) == {
        'x': ['x/1', 'x/2'], 'y': ['y/1']}
    assert cli_cytonorm._control_references('controls', {}) == {}


# ── _report_qc ────────────────────────────────────────────────────────────

class _FakeModel:
    def __init__(self, batches, means, seed=7):
        self.batches, self.seed, self.means = batches, seed, means
        self.qc_got = self.means_got = None

    def qc(self, events_by_batch):
        self.qc_got = events_by_batch
        return {'scores': 1}

    def qc_means(self, scores):
        self.means_got = scores
        return self.means


def _frames():
    rng = np.random.default_rng(3)

    def f(n):
        return pd.DataFrame(rng.normal(size=(n, 3)), columns=['p', 'q', 'r'])
    frames = {'x/1': f(50), 'y/1': f(40), 'x/2': f(30)}
    return frames, {'x/1': 'x', 'y/1': 'y', 'x/2': 'x'}


def test_report_qc_scores_the_capped_pooled_events_it_was_fitted_on():
    frames, batch_of = _frames()
    cn = _FakeModel(['y', 'x'], (0.5, 0.25, 0))
    said = []
    cli_cytonorm._report_qc(cn, frames, batch_of, ['r', 'p'], 45, 'goal',
                            said.append)
    want = cap_events_per_batch(
        {'y': frames['y/1'][['r', 'p']],
         'x': pd.concat([frames['x/1'][['r', 'p']], frames['x/2'][['r', 'p']]],
                        ignore_index=True)}, 45, seed=7)
    assert list(cn.qc_got) == ['y', 'x']
    for b in want:
        pd.testing.assert_frame_equal(cn.qc_got[b], want[b], check_exact=True)
    assert len(cn.qc_got['x']) == 45 and len(cn.qc_got['y']) == 40
    assert cn.means_got == {'scores': 1}
    assert said == ['[batch-correct] CytoNorm (goal): 2 markers, 2 batches, '
                    'mean batch→goal 0.500 → 0.250.']


@pytest.mark.parametrize('means, mode, tail', [
    ((1.23456, 0.0004, 2), 'controls',
     '1.235 → 0.000, 2 marker(s) not evaluable.'),
    ((float('nan'), float('nan'), 3), 'goal',
     'not evaluable, 3 marker(s) not evaluable.'),
    ((np.float64(2.0), np.float64(1.5), 0), 'goal', '2.000 → 1.500.'),
    ((1.0, float('nan'), 0), 'goal', '1.000 → nan.'),     # only `before` read
])
def test_report_qc_says_the_scores(means, mode, tail):
    frames, batch_of = _frames()
    said = []
    cli_cytonorm._report_qc(_FakeModel(['x'], means), frames, batch_of,
                            ['p'], 0, mode, said.append)
    assert said == [f'[batch-correct] CytoNorm ({mode}): 1 markers, 1 '
                    f'batches, mean batch→goal {tail}']


# ── fit_cytonorm: order, arguments and exits ──────────────────────────────

class _Rec:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, name, ret):
        def f(*args, **kw):
            self.calls.append((name, args, kw))
            if isinstance(ret, Exception):
                raise ret
            return ret
        return f


@pytest.fixture
def stub(monkeypatch):
    """Replace every helper and fit_batch_model with recorders."""
    calls = []
    rec = _Rec(calls)
    ref = type('S', (), {'fluor_channels': ['FL3-A', 'FL1-H', 'FL1-A',
                                            'FL2-A', 'FL4-A']})()
    model = type('M', (), {'to_dict': lambda self: {'model': 1}})()
    state = {
        '_batch_paths': {'b1': ['p1'], 'b2': ['p2']},
        '_load_batches': ({'k1': 'f1', 'k2': 'f2'}, {'k1': 'b1', 'k2': 'b2'},
                          {'FL1-A', 'FL3-A', 'FL4-A'}, ref),
        '_control_references': {'refs': 1},
        '_report_qc': None,
    }

    def install(**over):
        for name, ret in {**state, **over}.items():
            monkeypatch.setattr(cli_cytonorm, name, rec(name, ret))
        monkeypatch.setattr(batch_normalize, 'fit_batch_model',
                            rec('fit_batch_model', model))
    install()
    return calls, install, model


def test_fit_calls_its_parts_in_order_with_their_arguments(stub):
    calls, _install, model = stub
    say = []
    out = cli_cytonorm.fit_cytonorm(
        ['groups'], 'trial', mode='controls', n_metaclusters=4,
        control_token='  RefX ', max_events_per_batch=123, report=say.append,
        qc='qc-flag')
    assert out == ({'model': 1}, ['FL3-A', 'FL1-A', 'FL4-A'])
    names = [c[0] for c in calls]
    assert names == ['_batch_paths', '_load_batches', '_control_references',
                     'fit_batch_model', '_report_qc']
    assert calls[0][1:] == ((['groups'], 'trial', 'controls', 'refx',
                             '  RefX '), {})
    frames, batch_of = {'k1': 'f1', 'k2': 'f2'}, {'k1': 'b1', 'k2': 'b2'}
    assert calls[1][1][:2] == ({'b1': ['p1'], 'b2': ['p2']}, 'qc-flag')
    assert calls[1][1][2] == say.append and calls[1][2] == {}
    assert calls[2][1:] == (('controls', batch_of), {})
    assert calls[3][1:] == (
        (frames, batch_of, ['FL3-A', 'FL1-A', 'FL4-A']),
        {'mode': 'controls', 'references': {'refs': 1}, 'n_metaclusters': 4,
         'max_events_per_batch': 123})
    assert calls[4][1][:7] == (model, frames, batch_of,
                               ['FL3-A', 'FL1-A', 'FL4-A'], 123, 'controls',
                               say.append)
    assert say == []


def test_fit_prints_when_there_is_no_report(stub, capsys):
    calls, install, _model = stub
    install(_batch_paths={'b1': ['p1']})
    assert cli_cytonorm.fit_cytonorm([], 't') == (None, None)
    assert capsys.readouterr().out == (
        '[batch-correct] need ≥2 batches with samples (got 1) — skipped.\n')
    # The default report reaches the helpers too.
    install()
    cli_cytonorm.fit_cytonorm([], 't')
    say = [c for c in calls if c[0] == '_report_qc'][-1][1][-1]
    say('hello')
    assert capsys.readouterr().out == 'hello\n'


@pytest.mark.parametrize('token', ['', '   ', None])
def test_fit_controls_without_a_token_raises_before_reading(stub, token):
    calls, _install, _model = stub
    with pytest.raises(ValueError, match='needs --cytonorm-control'):
        cli_cytonorm.fit_cytonorm([], 't', mode='controls',
                                  control_token=token)
    assert calls == []


def test_fit_goal_mode_takes_no_token_and_passes_an_empty_one(stub):
    calls, _install, _model = stub
    cli_cytonorm.fit_cytonorm([], 't', control_token=None)
    assert calls[0][1] == ([], 't', 'goal', '', None)
    # goal mode reaches every part as goal mode
    assert calls[2][1] == ('goal', {'k1': 'b1', 'k2': 'b2'})
    assert calls[3][2]['mode'] == 'goal' and calls[4][1][5] == 'goal'


def test_fit_passes_on_the_batch_paths_error(stub):
    calls, install, _model = stub
    install(_batch_paths=ValueError('no control'))
    with pytest.raises(ValueError, match='no control'):
        cli_cytonorm.fit_cytonorm([], 't', mode='controls', control_token='c')
    assert [c[0] for c in calls] == ['_batch_paths']


@pytest.mark.parametrize('n', [0, 1])
def test_fit_stops_with_fewer_than_two_batches(stub, n):
    calls, install, _model = stub
    install(_batch_paths={f'b{i}': ['p'] for i in range(n)})
    said = []
    assert cli_cytonorm.fit_cytonorm([], 't', report=said.append) == (None, None)
    assert said == [f'[batch-correct] need ≥2 batches with samples (got {n}) '
                    '— skipped.']
    assert [c[0] for c in calls] == ['_batch_paths']


S = type('S', (), {'fluor_channels': ['FL1-A', 'FL2-A']})()
TWO = {'k1': 'b1', 'k2': 'b2'}


@pytest.mark.parametrize('loaded', [
    ({}, {}, None, None),                                 # nothing loaded
    ({}, TWO, {'FL1-A', 'FL2-A'}, None),                  # no reference
    ({}, TWO, None, S),                                   # no shared set
    ({}, TWO, {'FL1-A'}, S),                              # one shared channel
    ({}, {'k1': 'b1', 'k2': 'b1'}, {'FL1-A', 'FL2-A'}, S),  # one batch loaded
])
def test_fit_stops_without_two_shared_channels_and_batches(stub, loaded):
    calls, install, _model = stub
    install(_load_batches=loaded)
    said = []
    assert cli_cytonorm.fit_cytonorm([], 't', report=said.append) == (None, None)
    assert said == ['[batch-correct] insufficient shared markers / batches '
                    '— skipped.']
    assert [c[0] for c in calls] == ['_batch_paths', '_load_batches']


def test_fit_goes_on_with_exactly_two_channels_and_batches(stub):
    calls, install, _model = stub
    install(_load_batches=({}, TWO, {'FL1-A', 'FL2-A'}, S))
    assert cli_cytonorm.fit_cytonorm([], 't')[1] == ['FL1-A', 'FL2-A']
    assert calls[-1][0] == '_report_qc'


# ── fit_cytonorm end to end ───────────────────────────────────────────────

def _two_batches(root, rich=False, names=(('a1', 'a2'), ('b1', 'b2'))):
    groups = []
    for bi, (d, nms) in enumerate((('day0', names[0]), ('day3', names[1]))):
        for j, nm in enumerate(nms):
            _write(root / d / f'{nm}.fcs', seed=10 * bi + j, n=700,
                   gain=1.0 + .5 * bi, rich=rich)
        groups.append({'name': d, 'samples': list(nms),
                       'trial_dir': str(root / d)})
    return groups


@pytest.mark.parametrize('mode, token, qc', [
    ('goal', '', False), ('goal', '', True), ('controls', 'CTL', True)])
def test_fit_is_fit_batch_model_on_the_loaded_samples(tmp_path, mode, token,
                                                      qc):
    names = (('ctl0', 'a2'), ('b1', 'ctl3'))
    groups = _two_batches(tmp_path, rich=True, names=names)
    said = []
    model, channels = cli_cytonorm.fit_cytonorm(
        groups, str(tmp_path), mode=mode, n_metaclusters=2,
        control_token=token, max_events_per_batch=900, report=said.append,
        qc=qc)
    assert channels == ['FL1-A', 'FL2-A', 'FL3-A', 'FL4-A']
    frames, batch_of = {}, {}
    for g, nms in zip(groups, names, strict=True):
        for nm in nms:
            if mode == 'controls' and 'ctl' not in nm:
                continue
            key = f'{g["trial_dir"]}{os.sep}{nm}.fcs'
            frames[key] = _expected_frame(key, qc=qc).data
            batch_of[key] = g['trial_dir']
    refs = None
    if mode == 'controls':
        refs = {}
        for k, b in batch_of.items():
            refs.setdefault(b, []).append(k)
    app = batch_normalize.fit_batch_model(
        frames, batch_of, channels, mode=mode, references=refs,
        n_metaclusters=2, max_events_per_batch=900)
    assert model == app.to_dict()
    assert len(said) == 1
    assert said[0].startswith(f'[batch-correct] CytoNorm ({mode}): 4 markers, '
                              '2 batches, mean batch→goal ')
