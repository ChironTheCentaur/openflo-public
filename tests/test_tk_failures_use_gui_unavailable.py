"""A GUI test that cannot start Tk goes through conftest.gui_unavailable.

``pytest.skip`` in the ``except`` around ``tk.Tk()`` is fail-open: where CI
promised a GUI (OPENFLO_REQUIRE_GUI=1) a broken Tk would turn every such test
into a quiet skip and the run would still go green. gui_unavailable skips on
a headless dev box and fails there instead. Forty-six guards had the bare
skip; this test stops it coming back.
"""
from __future__ import annotations

import ast
import pathlib

TESTS = pathlib.Path(__file__).resolve().parent


def _called(nodes):
    names = set()
    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                names.add(f.attr if isinstance(f, ast.Attribute)
                          else getattr(f, 'id', ''))
    return names


def _skips_on_tk_failure(tree):
    """Line numbers of except handlers that pytest.skip when the try body
    starting Tk raised."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and 'Tk' in _called(node.body):
            found += [h.lineno for h in node.handlers
                      if 'skip' in _called(h.body)]
    return found


def test_no_tk_guard_skips_past_require_gui():
    bad = []
    for path in sorted(TESTS.glob('test_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        bad += [f'{path.name}:{ln}' for ln in _skips_on_tk_failure(tree)]
    assert not bad, (
        f'{bad} answer a Tk failure with pytest.skip, which lets a run with '
        'OPENFLO_REQUIRE_GUI=1 go green without the GUI; call '
        'tests.conftest.gui_unavailable(reason) instead')


def test_the_scan_sees_a_planted_skip():
    planted = ast.parse(
        'def a():\n'
        '    try:\n'
        '        root = tk.Tk()\n'
        '    except Exception as e:\n'
        '        pytest.skip(str(e))\n'
        'def b():\n'
        '    try:\n'
        '        root = tk.Tk()\n'
        '    except Exception as e:\n'
        '        gui_unavailable(str(e))\n'
        'def c():\n'
        '    try:\n'
        '        import h5py\n'
        '    except ImportError:\n'
        '        pytest.skip("no h5py")\n')
    assert _skips_on_tk_failure(planted) == [4]
