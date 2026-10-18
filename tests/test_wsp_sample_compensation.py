"""WspWriter puts each sample's OWN spillover matrix inside that <Sample>.

A FlowJo-written workspace carries each compensation matrix twice: once in
the top-level <Matrices> list, and once as a copy inside every <Sample> that
uses it, directly after <DataSet>, with the same transforms:id. FlowJo 10.10.2
applies a per-sample matrix as given (measured), and a sample with no
matrix gets FlowJo's "Acquisition-defined" one from its own FCS SPILL.

So the copy inside a sample must be THAT sample's matrix. Copying one
workspace-wide matrix into every sample would override every other file's own
SPILL: real runs mix files whose SPILL differs (some carry an identity SPILL).
"""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import openflo.pipeline as fp

T = 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations'
D = 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes'
SM = f'{{{T}}}spilloverMatrix'
CHANNELS = ['FSC-A', 'SSC-A', 'FL1-A', 'FL2-A', 'FL3-A']
COMP = ['FL1-A', 'FL2-A', 'FL3-A']
SPILL = np.array([[1.0, 0.12, 0.003],
                  [0.04, 1.0, 0.25],
                  [0.0, 0.015, 1.0]])
OTHER = np.array([[1.0, 0.3, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
GATES = [{'kind': 'interval', 'channel': 'FL1-A', 'lo': 0.0, 'hi': 1e5,
          'id': 'g1', 'parent_id': None, 'name': 'FL1 pos'}]


def _fcs(path):
    """A tiny synthetic FCS (the writer only needs a real path for DataSet)."""
    import flowio
    rng = np.random.default_rng(1)
    ev = rng.uniform(100, 5000, size=(100, len(CHANNELS))).astype(np.float32)
    with open(path, 'wb') as f:
        flowio.create_fcs(f, ev.flatten().tolist(), CHANNELS)
    return str(path)


def _matrix_values(m):
    """Matrix element -> (channels, ndarray) from the Gating-ML coefficients."""
    chans = [p.get(f'{{{D}}}name') for p in m.find(f'{{{D}}}parameters')]
    idx = {c: i for i, c in enumerate(chans)}
    out = np.full((len(chans), len(chans)), np.nan)
    for sp in m.findall(f'{{{T}}}spillover'):
        i = idx[sp.get(f'{{{D}}}parameter')]
        for c in sp.findall(f'{{{T}}}coefficient'):
            out[i, idx[c.get(f'{{{D}}}parameter')]] = float(c.get(f'{{{T}}}value'))
    return chans, out


def _write(tmp_path, own, workspace=None):
    """`own[k]` is sample k's (channels, matrix) or None."""
    w = fp.WspWriter(cytometer='Generic')
    if workspace is not None:
        w.set_compensation(*workspace)
    for k, comp in enumerate(own):
        w.add_sample(f's{k}.fcs', _fcs(tmp_path / f's{k}.fcs'), CHANNELS,
                     GATES, compensation=comp)
    out = tmp_path / 'out.wsp'
    w.write(str(out))
    return out, ET.parse(out).getroot()


def test_sample_carries_its_own_matrix_right_after_its_dataset(tmp_path):
    out, root = _write(tmp_path, [(COMP, SPILL), (COMP, SPILL)])
    top = root.findall('Matrices/' + SM)
    assert len(top) == 1, 'two samples with the same matrix share one entry'
    top_id = top[0].get(f'{{{T}}}id')
    for s in root.findall('SampleList/Sample'):
        kids = [c.tag for c in s]
        assert kids == ['DataSet', SM, 'Keywords', 'SampleNode'], kids   # FlowJo's order
        m = s.find(SM)
        assert m.get(f'{{{T}}}id') == top_id
        assert m.get('prefix') == 'Comp-' and m.get('suffix') == ''
        chans, vals = _matrix_values(m)
        assert chans == COMP
        np.testing.assert_array_equal(vals, SPILL)
        infix = [p.get('userProvidedCompInfix') for p in m.find(f'{{{D}}}parameters')]
        assert infix == [f'Comp-{c}' for c in COMP]
    chans_back, mat_back = fp.read_compensation_matrix(str(out))
    assert chans_back == COMP
    np.testing.assert_array_equal(mat_back, SPILL)


def test_samples_with_different_matrices_each_keep_their_own(tmp_path):
    # The mixed case real runs have: an identity SPILL next to a real one.
    ident = np.eye(3)
    _, root = _write(tmp_path, [(COMP, SPILL), (COMP, ident), (COMP, OTHER)],
                     workspace=(COMP, SPILL))
    top = {m.get(f'{{{T}}}id'): m for m in root.findall('Matrices/' + SM)}
    assert len(top) == 3
    assert len({m.get('name') for m in top.values()}) == 3
    got = []
    for s in root.findall('SampleList/Sample'):
        m = s.find(SM)
        assert m.get(f'{{{T}}}id') in top
        got.append(_matrix_values(m)[1])
    np.testing.assert_array_equal(got[0], SPILL)
    np.testing.assert_array_equal(got[1], ident)
    np.testing.assert_array_equal(got[2], OTHER)


def test_workspace_matrix_is_never_copied_into_a_sample(tmp_path):
    # set_compensation alone: listed in <Matrices>, but NO sample carries it,
    # so FlowJo applies each file's own SPILL instead of this one.
    _, root = _write(tmp_path, [None, (COMP, OTHER)], workspace=(COMP, SPILL))
    assert len(root.findall('Matrices/' + SM)) == 2
    s0, s1 = root.findall('SampleList/Sample')
    assert [c.tag for c in s0] == ['DataSet', 'Keywords', 'SampleNode']
    np.testing.assert_array_equal(_matrix_values(s1.find(SM))[1], OTHER)


def test_no_matrix_means_no_sample_matrix(tmp_path):
    _, root = _write(tmp_path, [None])
    assert root.findall('Matrices/' + SM) == []
    assert [c.tag for c in root.find('SampleList/Sample')] == ['DataSet', 'Keywords', 'SampleNode']


def test_per_sample_matrix_shape_is_checked(tmp_path):
    w = fp.WspWriter(cytometer='Generic')
    with pytest.raises(ValueError):
        w.add_sample('s0.fcs', _fcs(tmp_path / 's0.fcs'), CHANNELS, GATES,
                     compensation=(COMP, np.eye(2)))
