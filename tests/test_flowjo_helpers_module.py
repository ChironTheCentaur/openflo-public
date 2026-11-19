"""The last functions of pipeline.py live in their own core modules:
placeholder_booleans and the status-bar note (flowjo_placeholder_note, once
pipeline._flowjo_placeholder_note) in flowjo_placeholders, and
fcs_values_as_flowjo_gates_them in flowjo_events. pipeline.py now defines no
function of its own but its module ``__getattr__``.

These tests pin the move:

* the new modules do not import pipeline (or Tk);
* pipeline.placeholder_booleans and fcs_values_as_flowjo_gates_them are the
  moved functions, without a warning; _flowjo_placeholder_note resolves
  through pipeline._MOVED with exactly one warning;
* no module imports anything from pipeline, so FlowSample and the rest are
  read from their homes;
* the moved functions still do what they did, on generated gates and a
  generated FCS file.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

from openflo import flow_sample, flowjo_events, flowjo_placeholders, pipeline
from openflo.fcs_write import write_fcs

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'


def _ellipse(gid, name):
    return {'id': gid, 'kind': 'flowjo_ellipse', 'name': name}


def _boolean(gid, name, operands):
    return {'id': gid, 'kind': 'boolean', 'name': name, 'op': 'and',
            'operands': operands}


def _rect(gid, name):
    return {'id': gid, 'kind': 'rectangle', 'name': name, 'x': 'FL1-A',
            'y': 'FL2-A', 'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.flowjo_placeholders, openflo.flowjo_events; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_pipeline_defines_no_function_but_its_getattr_hook():
    tree = ast.parse((PACKAGE / 'pipeline.py').read_text(encoding='utf-8'))
    defined = [n.name for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef))]
    assert defined == ['__getattr__']


def test_pipeline_serves_the_public_names_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert (pipeline.placeholder_booleans
                is flowjo_placeholders.placeholder_booleans)
        assert (pipeline.fcs_values_as_flowjo_gates_them
                is flowjo_events.fcs_values_as_flowjo_gates_them)
        assert pipeline.FlowSample is flow_sample.FlowSample


def test_the_old_private_note_resolves_with_exactly_one_warning():
    assert '_flowjo_placeholder_note' not in vars(pipeline)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        obj = pipeline._flowjo_placeholder_note
    assert obj is flowjo_placeholders.flowjo_placeholder_note
    assert [w.category for w in caught] == [DeprecationWarning]
    assert 'flowjo_placeholders.flowjo_placeholder_note' in str(caught[0].message)


# The one module that still imports pipeline, as a probe that the data
# dependencies are installed; it reads nothing from it.
PIPELINE_PROBES = {'editor_session.py'}


def _pipeline_imports(tree):
    """(line, what) for every import of pipeline or of a name from it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.level and node.module == 'pipeline') or (
                    node.module == 'openflo.pipeline'):
                yield from ((node.lineno, a.name) for a in node.names)
            elif (node.level and node.module is None) or node.module == 'openflo':
                yield from ((node.lineno, a.name) for a in node.names
                            if a.name == 'pipeline')
        elif isinstance(node, ast.Import):
            yield from ((node.lineno, a.name) for a in node.names
                        if a.name == 'openflo.pipeline')


def test_no_module_imports_anything_from_pipeline():
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path.name == 'pipeline.py':
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        found = list(_pipeline_imports(tree))
        if path.name in PIPELINE_PROBES:
            assert found and all(w == 'pipeline' for _, w in found), path.name
            continue
        hits.extend(f'{path.name}:{line} {what}' for line, what in found)
    assert hits == []


def test_the_guard_sees_every_way_to_import_pipeline():
    for code in ('from .pipeline import FlowSample', 'from . import pipeline',
                 'from . import pipeline as _p', 'import openflo.pipeline',
                 'from openflo import pipeline',
                 'from openflo.pipeline import FlowSample'):
        assert list(_pipeline_imports(ast.parse(code))), code
    assert not list(_pipeline_imports(ast.parse(
        'from . import flow_sample\nimport openflo.flow_sample')))


def test_the_public_lazy_names_resolve_from_their_homes():
    import openflo
    for name in ('FlowSample', 'set_default_palette', 'WspReader',
                 'WspWriter', 'OpenFloError', 'GateError'):
        assert openflo._PUBLIC[name] != 'openflo.pipeline', name
        assert getattr(openflo, name) is getattr(pipeline, name), name


def test_placeholder_booleans_names_booleans_resting_on_a_placeholder():
    gates = [_ellipse('e1', 'Ell'), _rect('r1', 'Rect'),
             _boolean('b1', 'OnEll', ['e1', 'r1']),
             _boolean('b2', 'OnRect', ['r1']),
             _boolean('b3', 'OnOnEll', ['b1']),
             _boolean('b4', '', ['e1'])]
    names = flowjo_placeholders.placeholder_booleans(gates)
    assert names[:2] == ['OnEll', 'OnOnEll']
    assert len(names) == 3 and names[2]          # unnamed: described instead
    assert 'OnRect' not in names


def test_placeholder_booleans_reads_import_ids_first():
    gates = [{'_import_id': 'x9', 'id': 'other', 'kind': 'flowjo_ellipse',
              'name': 'Ell'},
             _boolean('b1', 'OnEll', ['x9'])]
    assert flowjo_placeholders.placeholder_booleans(gates) == ['OnEll']


def test_a_gate_without_an_id_is_never_an_operand():
    # A boolean whose operand is missing (None) must not find an id-less
    # placeholder under the key None.
    gates = [{'kind': 'flowjo_ellipse', 'name': 'NoId'},
             _boolean('b1', 'Dangling', [None])]
    assert flowjo_placeholders.placeholder_booleans(gates) == []


def test_the_note_is_empty_without_a_placeholder():
    gates = [_rect('r1', 'Rect'), _boolean('b1', 'B', ['r1'])]
    assert flowjo_placeholders.flowjo_placeholder_note(gates) == ''


def test_the_note_names_three_placeholders_and_counts_the_rest():
    gates = [_ellipse(f'e{i}', f'E{i}') for i in range(5)]
    gates[0]['label'] = 'Labelled'
    note = flowjo_placeholders.flowjo_placeholder_note(gates)
    assert note == (" — 5 FlowJo ellipse(s) not converted (Labelled, E1, E2 "
                    "(+2)): they and their subpopulations count no events "
                    "(see log)")


def test_the_note_names_the_booleans_found_or_given_once_each():
    gates = [_ellipse('e1', 'Ell')] + [
        _boolean(f'b{i}', f'B{i}', ['e1']) for i in range(4)]
    found = flowjo_placeholders.flowjo_placeholder_note(gates)
    assert found.endswith("; so do the boolean gate(s) built on them: "
                          "'B0', 'B1', 'B2' (+1)")
    given = flowjo_placeholders.flowjo_placeholder_note(
        gates, booleans=['Z', 'Z', 'Y'])
    assert given.endswith("built on them: 'Z', 'Y'")
    none = flowjo_placeholders.flowjo_placeholder_note(gates, booleans=[])
    assert 'boolean' not in none and none.startswith(' — 1 FlowJo ellipse(s)')


def _fcs(tmp_path, n=2000, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'FSC-A': rng.uniform(2e4, 2e5, n), 'SSC-A': rng.uniform(1e4, 1.5e5, n),
        'FL1-A': rng.lognormal(7, 1, n), 'FL2-A': rng.lognormal(6, 1, n),
        'Time': np.sort(rng.uniform(0, 1000, n))})
    # Events stuck at the top of a detector, which QC's margin check drops.
    df.loc[rng.choice(n, n // 20, replace=False), 'FL1-A'] = 262143.0
    path = tmp_path / 'synthetic.fcs'
    write_fcs(df, str(path))
    return str(path)


def test_the_events_are_the_file_s_without_qc_or_compensation(tmp_path):
    path = _fcs(tmp_path)
    got = flowjo_events.fcs_values_as_flowjo_gates_them(path)
    pd.testing.assert_frame_equal(got, flow_sample.FlowSample(path).data)


def test_the_events_are_compensated_with_the_given_matrix(tmp_path):
    path = _fcs(tmp_path)
    spill = np.array([[1.0, 0.2], [0.05, 1.0]])
    got = flowjo_events.fcs_values_as_flowjo_gates_them(
        path, compensation=(('FL1-A', 'FL2-A'), spill))
    raw = flow_sample.FlowSample(path).data
    want = raw[['FL1-A', 'FL2-A']].to_numpy() @ np.linalg.inv(spill)
    np.testing.assert_allclose(got[['FL1-A', 'FL2-A']].to_numpy(), want,
                               rtol=1e-9)
    pd.testing.assert_series_equal(got['FSC-A'], raw['FSC-A'])


def test_the_events_are_the_qc_kept_ones_when_qc_ran(tmp_path):
    path = _fcs(tmp_path)
    ref = flow_sample.FlowSample(path)
    ref.run_qc()
    assert len(ref.data) < 2000                  # QC did drop events
    got = flowjo_events.fcs_values_as_flowjo_gates_them(path, qc=True)
    pd.testing.assert_frame_equal(got, ref.data)


def test_workspace_reload_warnings_still_go_to_the_pipeline_logger(
        tmp_path, monkeypatch, caplog):
    # workspace_run logged these through pipeline.log; it now names the
    # same logger itself instead of importing pipeline.
    import logging
    import types

    from openflo import workspace_run

    def fail(self, *a, **k):
        raise RuntimeError('planted QC failure')
    monkeypatch.setattr(flow_sample.FlowSample, 'run_qc', fail)
    editor = types.SimpleNamespace(_samples={})
    item = {'sample': 'synthetic', 'path': _fcs(tmp_path), 'qc_applied': True}
    with caplog.at_level(logging.WARNING, logger='openflo.pipeline'):
        workspace_run.resolve_run_sample(editor, item)
    assert any(r.name == 'openflo.pipeline'
               and 'QC on reload failed' in r.getMessage()
               for r in caplog.records), caplog.records
