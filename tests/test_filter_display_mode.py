"""'Filter to gated' has one source of truth: the display mode.

The editor used to keep a second variable, ``apply_gates_var``, that every
display-mode handler had to set by hand. Code that changed ``gate_display_var``
any other way left the filter reading stale. These tests pin the single
source: setting the display mode alone filters, and the old name still works
as a deprecated view onto the display mode.
"""
import ast
import pathlib
import warnings

import pytest

from tests.test_gui_ux import _editor_or_skip, _load_sample

SRC = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'


def _half_gate(ed):
    """Synthetic sample of 200 events spread 0..1 on every channel, and a
    rectangle that keeps the lower half on FL1-A."""
    _load_sample(ed, cols=('FSC-A', 'SSC-A', 'FL1-A', 'FL2-A'))
    ed._set_active_sample('s1')
    ed.x_combo.set('FL1-A')
    ed.y_combo.set('FL2-A')
    ed._add_gate({'kind': 'rect', 'x_channel': 'FL1-A', 'y_channel': 'FL2-A',
                  'x0': -1.0, 'x1': 0.5, 'y0': -1.0, 'y1': 2.0,
                  'name': 'low', 'parent_id': None})
    data = ed._samples['s1'].data
    return len(data), int((data['FL1-A'] <= 0.5).sum())


def test_setting_the_display_mode_alone_filters():
    root, ed = _editor_or_skip()
    try:
        total, inside = _half_gate(ed)
        assert 0 < inside < total
        for mode, expect in (('filter', inside), ('all', total),
                             ('highlight', total), ('filter', inside)):
            ed.gate_display_var.set(mode)      # no handler runs
            assert ed._filters_to_gated() is (mode == 'filter')
            got = ed._get_df('s1', 'FL1-A', 'FL2-A', downsample=False)
            assert len(got) == expect, mode
    finally:
        root.destroy()


def test_apply_gates_var_is_a_deprecated_view_of_the_display_mode():
    root, ed = _editor_or_skip()
    try:
        ed.gate_display_var.set('filter')
        with pytest.warns(DeprecationWarning, match='apply_gates_var'):
            assert ed.apply_gates_var.get() is True
        with pytest.warns(DeprecationWarning):
            ed.apply_gates_var.set(False)
        assert ed.gate_display_var.get() == 'all'
        with pytest.warns(DeprecationWarning):
            ed.apply_gates_var.set(True)
        assert ed.gate_display_var.get() == 'filter'
        # Turning the filter off leaves a non-filter mode alone.
        ed.gate_display_var.set('highlight')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            assert ed.apply_gates_var.get() is False
            ed.apply_gates_var.set(False)
        assert ed.gate_display_var.get() == 'highlight'
    finally:
        root.destroy()


def test_no_app_code_reads_apply_gates_var():
    """Only the deprecated shim in editor_mode may name it."""
    users = []
    for path in sorted(SRC.glob('*.py')):
        if path.name == 'editor_mode.py':
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.Attribute) and node.attr == 'apply_gates_var':
                users.append(f'{path.name}:{node.lineno}')
    assert users == []

