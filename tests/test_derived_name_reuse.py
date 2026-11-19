"""Adding a derived channel under a name already in use (scope audit D3).

"Add to all samples" with an existing name and a new formula replaced the
channel in every sample with no word, and gates drawn on it then selected
other events. Remove refused while a gate used the channel; Add did not.
Now Add asks first (default No) when the name exists with another formula,
and refuses while a gate is drawn on it in a sample it would change, in the
live gates or in another tab's own gates. Re-adding the same formula (to
reach samples loaded since) is unchanged and asks nothing. Remove's gate
check now looks in every tab too.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import derived
from openflo.view_tabs import ViewTab
from tests.conftest import editor_or_skip

COLS = ('FL1-A', 'FL2-A')


def _sample(name, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(100, 1000, (60, 2)), columns=list(COLS))
    return SimpleNamespace(name=name, path=f'{name}.fcs', data=df,
                           data_transforms={}, fluor_channels=list(COLS),
                           channel_labels={c: c for c in COLS})


def _samples(*names):
    return {n: _sample(n, k) for k, n in enumerate(names)}


# ── derived.replaced_by ─────────────────────────────────────────────────────

def test_replaced_by_lists_the_samples_another_formula_would_change():
    s = _samples('SampleA', 'SampleB', 'SampleC')
    derived.apply_to_samples({k: s[k] for k in ('SampleA', 'SampleB')},
                             'ratio', '[FL1-A] / [FL2-A]')
    col, names = derived.replaced_by(s, 'ratio', '[FL2-A] - [FL1-A]')
    assert col == 'D:ratio'
    assert names == ['SampleA', 'SampleB']      # SampleC has no such column
    assert derived.replaced_by(s, 'ratio', '[FL1-A] / [FL2-A]') == (
        'D:ratio', [])
    assert derived.replaced_by(s, 'other', '[FL1-A]') == ('D:other', [])


def test_replaced_by_counts_a_column_with_no_formula_recorded():
    s = _samples('SampleA')
    s['SampleA'].data['D:ratio'] = 1.0
    assert derived.replaced_by(s, 'ratio', '[FL1-A]')[1] == ['SampleA']


def test_replaced_by_checks_the_formula_and_name_first():
    s = _samples('SampleA')
    with pytest.raises(derived.FormulaError):
        derived.replaced_by(s, 'ratio', '[FL1-A] /')
    with pytest.raises(derived.FormulaError):
        derived.replaced_by(s, '', '[FL1-A]')


# ── the window ──────────────────────────────────────────────────────────────

class _Ask:
    """Stands in for messagebox.askyesno: records the call, gives `answer`."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.answer


@pytest.fixture
def win(monkeypatch):
    from tkinter import messagebox

    from openflo.ui_derived import DerivedChannelsWindow
    root, ed = editor_or_skip()
    for name, s in _samples('SampleA', 'SampleB').items():
        ed._samples[name] = s
        ed._sample_order.append(name)
        ed._sample_gates.setdefault(name, {})
        ed._sample_gate_order.setdefault(name, [])
    ed._channels = list(COLS)
    ed._channel_labels = {c: c for c in COLS}
    w = DerivedChannelsWindow(ed)
    w.name_var.set('ratio')
    w.formula_var.set('[FL1-A] / [FL2-A]')
    w._add()
    assert 'D:ratio' in ed._samples['SampleA'].data.columns
    ask = _Ask(False)
    monkeypatch.setattr(messagebox, 'askyesno', ask)
    w.ask = ask
    yield w
    w.destroy()
    root.destroy()


def _values(w, name='SampleA'):
    return w.editor._samples[name].data['D:ratio'].to_numpy().copy()


def _gate_on_ratio():
    return {'g1': {'kind': 'threshold', 'channel': 'D:ratio', 'op': '>',
                   'value': 1.0, 'parent_id': None}}


def test_a_new_formula_under_a_used_name_asks_first_default_no(win):
    before = _values(win)
    win.formula_var.set('[FL2-A] - [FL1-A]')
    win._add()
    assert len(win.ask.calls) == 1
    (_title, text), kw = win.ask.calls[0]
    assert "'D:ratio' already exists in 2 sample(s)" in text
    assert '[FL1-A] / [FL2-A]' in text           # the formula it replaces
    assert kw.get('default') == 'no'
    np.testing.assert_array_equal(_values(win), before)
    assert win.editor._samples['SampleA'].derived == {
        'D:ratio': '[FL1-A] / [FL2-A]'}
    assert 'kept' in win.status_var.get()


def test_yes_replaces_it_in_every_sample(win):
    win.ask.answer = True
    win.formula_var.set('[FL2-A] - [FL1-A]')
    win._add()
    for name in ('SampleA', 'SampleB'):
        d = win.editor._samples[name].data
        np.testing.assert_allclose(d['D:ratio'], d['FL2-A'] - d['FL1-A'])
    assert "added to 2 sample(s)" in win.status_var.get()


def test_the_same_formula_again_asks_nothing(win):
    """Re-adding to reach a sample loaded since is the old, quiet path."""
    ed = win.editor
    ed._samples['SampleC'] = _sample('SampleC', 9)
    ed._sample_order.append('SampleC')
    win._add()
    assert win.ask.calls == []
    assert 'D:ratio' in ed._samples['SampleC'].data.columns
    assert "added to 3 sample(s)" in win.status_var.get()


def test_refused_while_a_gate_is_drawn_on_it(win):
    win.ask.answer = True
    win.editor._sample_gates['SampleB'] = _gate_on_ratio()
    before = _values(win, 'SampleB')
    win.formula_var.set('[FL2-A] - [FL1-A]')
    win._add()
    assert win.ask.calls == []
    np.testing.assert_array_equal(_values(win, 'SampleB'), before)
    msg = win.status_var.get()
    assert 'SampleB' in msg and 'gates' in msg


def test_refused_while_another_tab_holds_a_gate_on_it(win):
    """Tabs keep their own gates: one only a hidden tab holds still counts."""
    win.ask.answer = True
    tab = ViewTab('Tab 2', members={'SampleA': None})
    tab.gates, tab.gate_order = {'SampleA': _gate_on_ratio()}, {
        'SampleA': ['g1']}
    win.editor._tabs().tabs.append(tab)
    before = _values(win)
    win.formula_var.set('[FL2-A] - [FL1-A]')
    win._add()
    np.testing.assert_array_equal(_values(win), before)
    assert "SampleA (tab 'Tab 2')" in win.status_var.get()


def test_a_gate_on_a_sample_it_would_not_change_does_not_block(win):
    """Only the samples whose values would change are checked for gates."""
    ed = win.editor
    derived.add_to_sample(ed._samples['SampleB'], 'ratio', '[FL2-A]')
    ed._sample_gates['SampleA'] = _gate_on_ratio()
    win.ask.answer = True
    win._add()                     # SampleA's formula is this one already
    assert len(win.ask.calls) == 1
    assert '1 sample(s) (SampleB)' in win.ask.calls[0][0][1]
    d = ed._samples['SampleB'].data
    np.testing.assert_allclose(d['D:ratio'], d['FL1-A'] / d['FL2-A'])


def test_remove_refuses_while_another_tab_holds_a_gate_on_it(win):
    win.ask.answer = True
    tab = ViewTab('Tab 2', members={'SampleB': None})
    tab.gates, tab.gate_order = {'SampleB': _gate_on_ratio()}, {
        'SampleB': ['g1']}
    win.editor._tabs().tabs.append(tab)
    win.tv.selection_set('D:ratio')
    win._remove()
    assert 'D:ratio' in win.editor._samples['SampleB'].data.columns
    assert "SampleB (tab 'Tab 2')" in win.status_var.get()
