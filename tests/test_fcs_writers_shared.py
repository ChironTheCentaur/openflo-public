"""Tests write their synthetic FCS files through tests/_fcs_writer.py.

Sixty tests each opened a file and called ``flowio.create_fcs`` with their
own flatten / ravel / astype spelling, so a change to how test files are
written (a keyword every file needs, a flowio signature change) meant
sixty edits. They now call ``write_float_fcs``, and these tests stop new
copies coming back. tests/test_fcs_speed.py keeps its own call on purpose:
it reproduces the app's former export path to time it.
"""
from __future__ import annotations

import ast
import pathlib

import flowio
import numpy as np

from tests._fcs_writer import write_float_fcs

TESTS = pathlib.Path(__file__).resolve().parent

# Files that may call flowio.create_fcs directly. This list may only shrink.
_MAY_CALL_CREATE_FCS = {
    '_fcs_writer.py',     # the shared writer itself
    'test_fcs_speed.py',  # times the app's former export, call for call
}


def _create_fcs_calls(tree):
    """Line numbers of calls to ``create_fcs``, as ``flowio.create_fcs(...)``,
    ``<alias>.create_fcs(...)`` or a bare ``create_fcs(...)`` after a
    from-import."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if ((isinstance(f, ast.Attribute) and f.attr == 'create_fcs')
                or (isinstance(f, ast.Name) and f.id == 'create_fcs')):
            found.append(node.lineno)
    return found


def test_no_test_file_writes_fcs_through_its_own_create_fcs_call():
    bad = []
    for path in sorted(TESTS.glob('*.py')):
        if path.name in _MAY_CALL_CREATE_FCS:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        bad += [f'{path.name}:{ln}' for ln in _create_fcs_calls(tree)]
    assert not bad, (
        f'{bad} call flowio.create_fcs directly; use '
        'tests._fcs_writer.write_float_fcs (or write_fcs for integer data '
        'or $PnE / $PnG / $PnR)')


def test_the_allowed_list_only_names_files_that_still_need_it():
    stale = [name for name in sorted(_MAY_CALL_CREATE_FCS)
             if not _create_fcs_calls(ast.parse(
                 (TESTS / name).read_text(encoding='utf-8')))]
    assert not stale, f'{stale} no longer call create_fcs; drop them from the list'


def test_the_scan_sees_a_planted_copy():
    planted = ast.parse(
        "import flowio\n"
        "import flowio as fio\n"
        "from flowio import create_fcs\n"
        "def f(fh, ev):\n"
        "    flowio.create_fcs(fh, ev, ['A'])\n"
        "    fio.create_fcs(fh, ev, ['A'])\n"
        "    create_fcs(fh, ev, ['A'])\n"
        "def left_alone(path, ev):\n"
        "    write_float_fcs(path, ev, ['A'])\n"
        "    flowio.FlowData(path)\n")
    assert _create_fcs_calls(planted) == [5, 6, 7]


def test_write_float_fcs_round_trips_events_names_markers_and_keywords(tmp_path):
    rng = np.random.default_rng(7)
    ev = np.column_stack([rng.uniform(1e3, 1e5, 40), rng.normal(0, 50, 40),
                          np.arange(40) * 0.5])          # float64, negatives
    path = write_float_fcs(tmp_path / 'x.fcs', ev, ['FSC-A', 'FL1-A', 'Time'],
                           opt_channel_names=['', 'CD901', ''],
                           metadata_dict={'TIMESTEP': '0.5'})
    assert path == str(tmp_path / 'x.fcs')
    fd = flowio.FlowData(path)
    assert (fd.event_count, fd.channel_count) == (40, 3)
    got = np.reshape(np.asarray(fd.events, dtype=np.float64), (40, 3))
    np.testing.assert_array_equal(got, ev.astype(np.float32))
    assert fd.pnn_labels == ['FSC-A', 'FL1-A', 'Time']
    assert fd.pns_labels == ['', 'CD901', '']
    assert fd.text['timestep'] == '0.5'
    assert fd.text['datatype'] == 'F'


def test_write_float_fcs_keeps_event_order(tmp_path):
    """Rows are events: a transposed write would swap channels and events."""
    ev = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = write_float_fcs(tmp_path / 'o.fcs', ev, ['A', 'B', 'C'])
    fd = flowio.FlowData(path)
    got = np.reshape(np.asarray(fd.events, dtype=np.float64), (-1, fd.channel_count))
    np.testing.assert_array_equal(got, ev)
