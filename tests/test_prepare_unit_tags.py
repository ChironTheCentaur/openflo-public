"""Source tags are categorical, and mean exactly what they meant before.

`prepare_unit` tags every event with its source group and sample. Those are one
short string repeated across a whole member, and assigning them as object-dtype
columns materialised millions of Python string references: measured on a
12-member x 250k-event unit (3M rows), that ONE step cost 415 ms of the
function's 531 ms and 300 MB of the frame — which is then pickled to hand to
the child process, so the cost was paid twice.

As a categorical the same information is int32 codes plus a handful of strings:
prepare_unit 604 -> 111 ms on the Tk thread, 730 -> 414 MB in memory, and the
job.pkl write 3044 -> 803 ms.

This file pins that the VALUES did not change, because the point of the tags is
downstream: `color_by` names one of these columns, the embedding plot colours
by it, and a cluster-by-source crosstab is written from it.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openflo import workspace as W

COLS = ['FSC-A', 'CD3', 'CD19']


def _sample(name, n, seed=0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(
        name=name,
        data=pd.DataFrame({c: rng.normal(1000.0, 200.0, n) for c in COLS}),
        fluor_channels=['CD3', 'CD19'], scatter_channels=['FSC-A'],
        channel_labels={c: c for c in COLS})


@pytest.fixture
def editor(monkeypatch):
    samples = {f's{i}': _sample(f's{i}', 400 + 100 * i, seed=i)
               for i in range(6)}
    monkeypatch.setattr(W, 'resolve_run_sample',
                        lambda ed, item: (ed._samples[item['sample']], ''))
    return SimpleNamespace(_samples=samples)


def _members(n=6, groups=3):
    return [{'item': {'sample': f's{i}'}, 'group': f'g{i % groups}',
             'sample': f's{i}'} for i in range(n)]


def _expected_tags(editor, members):
    """What the object-dtype implementation produced, built independently."""
    gtags, stags = [], []
    for m in members:
        n = len(editor._samples[m['sample']].data)
        gtags.append(np.repeat(m['group'] or '(ungrouped)', n))
        stags.append(np.repeat(f"{m['group']} / {m['sample']}"
                               if m['group'] else m['sample'], n))
    return np.concatenate(gtags), np.concatenate(stags)


def test_the_tag_values_are_unchanged(editor):
    members = _members()
    prep = W.prepare_unit(editor, members, 'unit', {'seed': 42})
    want_g, want_s = _expected_tags(editor, members)

    assert list(prep['data']['__group__'].astype(str)) == list(want_g)
    assert list(prep['data']['__sample__'].astype(str)) == list(want_s)


def test_the_tags_are_categorical(editor):
    prep = W.prepare_unit(editor, _members(), 'unit', {'seed': 42})
    for col in ('__group__', '__sample__'):
        assert isinstance(prep['data'][col].dtype, pd.CategoricalDtype), (
            f'{col} is {prep["data"][col].dtype} — the object-dtype cost is '
            'back')


def test_each_member_contributes_its_own_events(editor):
    """The codes are built from run lengths, so an off-by-one would silently
    mislabel whole blocks of events."""
    members = _members()
    prep = W.prepare_unit(editor, members, 'unit', {'seed': 42})
    counts = prep['data']['__sample__'].astype(str).value_counts()
    for m in members:
        tag = f"{m['group']} / {m['sample']}"
        assert counts[tag] == len(editor._samples[m['sample']].data)


def test_colour_by_still_picks_the_right_column(editor):
    """Multiple groups -> colour by group; one group -> by sample; a single
    member -> neither."""
    multi = W.prepare_unit(editor, _members(6, groups=3), 'u', {'seed': 42})
    assert multi['color_by'] == '__group__'

    one_group = [{'item': {'sample': f's{i}'}, 'group': 'g0',
                  'sample': f's{i}'} for i in range(3)]
    assert W.prepare_unit(editor, one_group, 'u',
                          {'seed': 42})['color_by'] == '__sample__'

    single = [{'item': {'sample': 's0'}, 'group': 'g0', 'sample': 's0'}]
    assert W.prepare_unit(editor, single, 'u', {'seed': 42})['color_by'] is None


def test_subsampling_drops_categories_nobody_has(editor):
    """A categorical keeps every category after `.sample()`. The downstream
    cluster-by-source crosstab would then gain a zero column for a member that
    lost all its events — which the object dtype never produced."""
    members = _members()
    prep = W.prepare_unit(editor, members, 'unit',
                          {'seed': 42, 'max_events': 50})
    assert prep['n_events'] == 50
    for col in ('__group__', '__sample__'):
        present = set(prep['data'][col].astype(str).unique())
        declared = set(prep['data'][col].cat.categories)
        assert declared == present, (
            f'{col} declares categories nobody has: {declared - present}')


def test_the_crosstab_downstream_is_unaffected(editor):
    """The actual consumer: a cluster-by-source table. Its columns must be the
    sources that are present, exactly as before."""
    prep = W.prepare_unit(editor, _members(), 'unit', {'seed': 42})
    data = prep['data']
    rng = np.random.default_rng(0)
    data['cluster'] = rng.integers(0, 3, len(data))

    table = pd.crosstab(data['cluster'], data[prep['color_by']])
    assert list(table.columns) == sorted(
        data[prep['color_by']].astype(str).unique())
    assert int(table.to_numpy().sum()) == len(data)


def test_it_survives_a_pickle_round_trip(editor):
    """The frame is pickled to hand to the child process, so the tags have to
    mean the same thing on the other side."""
    import pickle

    prep = W.prepare_unit(editor, _members(), 'unit', {'seed': 42})
    back = pickle.loads(pickle.dumps(prep))
    for col in ('__group__', '__sample__'):
        assert list(back['data'][col].astype(str)) == list(
            prep['data'][col].astype(str))
        assert isinstance(back['data'][col].dtype, pd.CategoricalDtype)


def test_an_ungrouped_member_is_labelled_as_such(editor):
    members = [{'item': {'sample': 's0'}, 'group': None, 'sample': 's0'},
               {'item': {'sample': 's1'}, 'group': None, 'sample': 's1'}]
    prep = W.prepare_unit(editor, members, 'unit', {'seed': 42})
    assert set(prep['data']['__group__'].astype(str)) == {'(ungrouped)'}
    assert set(prep['data']['__sample__'].astype(str)) == {'s0', 's1'}


def test_a_member_with_no_events_does_not_shift_the_labels(editor):
    """A zero-length run length must consume no codes, or every later member's
    events would carry the wrong source."""
    editor._samples['empty'] = _sample('empty', 0)
    members = [{'item': {'sample': 's0'}, 'group': 'g0', 'sample': 's0'},
               {'item': {'sample': 'empty'}, 'group': 'g1', 'sample': 'empty'},
               {'item': {'sample': 's1'}, 'group': 'g2', 'sample': 's1'}]
    prep = W.prepare_unit(editor, members, 'unit', {'seed': 42})
    counts = prep['data']['__group__'].astype(str).value_counts()
    assert counts.get('g0', 0) == len(editor._samples['s0'].data)
    assert counts.get('g2', 0) == len(editor._samples['s1'].data)
    assert 'g1' not in set(prep['data']['__group__'].astype(str))
