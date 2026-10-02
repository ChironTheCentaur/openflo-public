"""WspWriter sample structure must match the form FlowJo itself writes.

Three properties, each taken from workspaces FlowJo wrote:

* ``<DataSet uri>`` is ``file:/`` + the absolute path with forward slashes,
  spaces as ``%20`` and commas literal (``file:/C:/a%20b,c/x.fcs``).
* A ``<Population owningGroup="G">`` only ever names a group whose
  ``<GroupNode>`` carries a population at the same name-path. A population
  that exists only on the sample is written ``owningGroup=""``.
* The "All Samples" ``<Group>`` lists every sample in ``<SampleRefs>``.

OpenFlo's own uri reader must keep accepting both the old writer form
(``file:C:/a b/x.fcs``) and the new one.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import pytest

import openflo.pipeline as fp
from openflo.compare import _resolve_fcs_uri

GATES = [
    {'id': 'r', 'parent_id': None, 'name': 'cells', 'kind': 'rect',
     'x_channel': 'FSC-A', 'y_channel': 'SSC-A',
     'x0': 1e3, 'x1': 1e6, 'y0': 1e3, 'y1': 1e6},
    {'id': 't', 'parent_id': 'r', 'name': 'big', 'kind': 'threshold',
     'channel': 'FSC-A', 'value': 2e4},
    {'id': 'i', 'parent_id': None, 'name': 'side', 'kind': 'interval',
     'channel': 'SSC-A', 'lo': 0.0, 'hi': 1e6},
]


@pytest.fixture
def spaced_fcs(tmp_path, synthetic_fcs):
    """Two copies of the synthetic FCS in a folder whose name needs
    escaping in a URI (a space and a comma)."""
    folder = tmp_path / 'run a,b'
    folder.mkdir()
    out = []
    for k in (1, 2):
        p = folder / f'tube {k}.fcs'
        shutil.copy(synthetic_fcs, p)
        out.append(str(p))
    return out


def _write(tmp_path, fcs_paths, name='w.wsp'):
    w = fp.WspWriter()
    for p in fcs_paths:
        w.add_sample(os.path.basename(p), p, ['FSC-A', 'SSC-A'], GATES)
    out = tmp_path / name
    w.write(str(out))
    return ET.parse(str(out)).getroot(), str(out)


def _pop_paths(node, prefix=()):
    """{name-path tuple: Population element} below `node`."""
    found = {}
    sub = node.find('Subpopulations')
    if sub is None:
        return found
    for pop in sub.findall('Population'):
        key = prefix + (pop.get('name'),)
        found[key] = pop
        found.update(_pop_paths(pop, key))
    return found


def test_dataset_uri_is_flowjo_file_form(tmp_path, spaced_fcs):
    root, _ = _write(tmp_path, spaced_fcs)
    uris = [ds.get('uri') for ds in root.iter('DataSet')]
    assert len(uris) == len(spaced_fcs)
    for uri, path in zip(uris, spaced_fcs, strict=True):
        assert uri.startswith('file:/') and not uri.startswith('file://'), uri
        assert ' ' not in uri and '\\' not in uri, uri
        assert uri.endswith(f'/run%20a,b/{os.path.basename(path).replace(" ", "%20")}'), uri
        if sys.platform == 'win32':
            assert re.match(r'^file:/[A-Za-z]:/', uri), uri
        decoded = unquote(uri[len('file:'):])
        if re.match(r'^/[A-Za-z]:/', decoded):
            decoded = decoded[1:]
        assert os.path.normcase(os.path.normpath(decoded)) == \
            os.path.normcase(os.path.abspath(path))


def test_dataset_uri_from_relative_path_is_absolute(tmp_path, spaced_fcs,
                                                    monkeypatch):
    """A relative fcs_path (the CLI joins a possibly relative trial dir) must
    still produce an absolute uri that resolves from any working directory."""
    monkeypatch.chdir(tmp_path)
    rel = os.path.relpath(spaced_fcs[0], str(tmp_path))
    root, _ = _write(tmp_path, [rel])
    uri = root.find('SampleList/Sample/DataSet').get('uri')
    assert uri.startswith('file:/'), uri
    monkeypatch.chdir(os.path.dirname(spaced_fcs[0]))   # cwd must not matter
    resolved = _resolve_fcs_uri(uri)
    assert resolved is not None, f'{uri!r} resolves only from the export cwd'
    assert os.path.samefile(resolved, spaced_fcs[0])


@pytest.mark.parametrize('form', ['old', 'new'])
def test_uri_reader_accepts_old_and_new_forms(tmp_path, spaced_fcs, form):
    path = spaced_fcs[0]
    if form == 'old':                       # what WspWriter wrote before
        uri = 'file:' + path.replace('\\', '/')
    else:
        root, _ = _write(tmp_path, [path])
        uri = root.find('SampleList/Sample/DataSet').get('uri')
    resolved = _resolve_fcs_uri(uri)
    assert resolved is not None and os.path.samefile(resolved, path), uri


def test_compare_inventory_resolves_written_workspace(tmp_path, spaced_fcs):
    from openflo.compare import _per_sample_inventory
    _, out = _write(tmp_path, spaced_fcs)
    inv = _per_sample_inventory(fp.WspReader(out))
    assert [os.path.basename(name) for name, _, _ in inv] == \
        [os.path.basename(p) for p in spaced_fcs]
    for (_, uri, _), path in zip(inv, spaced_fcs, strict=True):
        assert os.path.samefile(_resolve_fcs_uri(uri), path)
    # Gates still round-trip through the reader.
    assert len(fp.WspReader(out).extract_gates()) == len(GATES) * len(spaced_fcs)


def test_population_owning_group_has_the_gate(tmp_path, spaced_fcs):
    """owningGroup="G" is only valid when GroupNode G carries the same
    population path; otherwise the population is the sample's own ("")."""
    root, _ = _write(tmp_path, spaced_fcs)
    group_paths = {gn.get('name'): set(_pop_paths(gn))
                   for gn in root.iter('GroupNode')}
    bad = []
    for sn in root.find('SampleList').iter('SampleNode'):
        for key, pop in _pop_paths(sn).items():
            owner = pop.get('owningGroup')
            if owner and key not in group_paths.get(owner, set()):
                bad.append((sn.get('name'), '/'.join(key), owner))
    assert not bad, f'populations claim a group that holds no such gate: {bad}'


def test_all_samples_group_lists_every_sample(tmp_path, spaced_fcs):
    root, _ = _write(tmp_path, spaced_fcs)
    gn = [g for g in root.iter('GroupNode') if g.get('name') == 'All Samples']
    assert len(gn) == 1
    refs = [r.get('sampleID')
            for r in gn[0].findall('Group/SampleRefs/SampleRef')]
    ids = [sn.get('sampleID') for sn in root.iter('SampleNode')]
    ds_ids = [ds.get('sampleID') for ds in root.iter('DataSet')]
    assert ids == ds_ids
    assert sorted(refs) == sorted(ids) and len(refs) == len(set(refs))
