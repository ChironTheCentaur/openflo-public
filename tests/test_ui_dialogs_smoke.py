"""Every ui_*.py window must survive being opened and driven in a DEGENERATE
editor state.

The two crashers found in this area both fired when the editor was nearly
empty: `ui_sample_qc._markers()` indexed `names[0]` with nothing selected, and
`ui_voltage._chan_lookup` was read before it existed. Neither is reachable when
the editor is comfortably populated — which is the state a hand-written test
naturally sets up, and why 26 of 30 dialogs could be "under-covered" without
anyone noticing.

This walks every window class that takes only the editor, constructs it, and
calls each zero-argument action method (`_compute`, `_draw`, `_refresh`,
`_export…`, `_on_…`), under three states:

    full          — three enabled samples and a gate (the happy path)
    none_enabled  — samples loaded, none ticked for plotting
    no_samples    — nothing loaded at all

It is a NET, not a substitute for behavioural tests: it proves nothing about
whether a dialog computes the right answer. It was checked against the real
`ui_sample_qc` bug — with the guard removed it reports that crash, and it does
not report it with the guard in place — so a green run means something.
"""
import inspect
import os
import pkgutil
from types import SimpleNamespace

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import openflo  # noqa: E402

COLS = ['FSC-A', 'FSC-H', 'SSC-A', 'LiveDead-A', 'BV510-A', 'FITC-A', 'APC-A']
FLUOR = COLS[3:]
ACTION_PREFIXES = ('_compute', '_draw', '_refresh', '_rebuild', '_run',
                   '_apply', '_update', '_populate', '_export', '_on_')


def _editor(state):
    try:
        import tkinter as tk
    except ImportError:
        pytest.skip('tkinter not available')
    try:
        root = tk.Tk()
        root.withdraw()
    except Exception as e:                            # noqa: BLE001
        pytest.skip(str(e))
    import importlib
    gui = importlib.import_module('openflo.gui')
    for n in ('askyesno', 'showinfo', 'showwarning', 'showerror',
              'askokcancel'):
        setattr(gui.messagebox, n, lambda *a, **k: True)
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()

    rng = np.random.default_rng(0)
    names = () if state == 'no_samples' else ('s1', 's2', 's3')
    for i, nm in enumerate(names):
        df = pd.DataFrame({c: rng.random(600) * 50000 for c in COLS})
        df['cluster'] = rng.integers(-1, 5, 600)
        ed._samples[nm] = SimpleNamespace(
            name=nm, path=rf'C:\exp\{nm}.fcs', data=df,
            fluor_channels=list(FLUOR), raw=df.copy(),
            channel_labels={c: c for c in COLS})
        ed._sample_order.append(nm)
        ed._sample_colors[nm] = ['#1f77b4', '#d62728', '#2ca02c'][i]
        ed._sample_trial[nm] = 'Day1'
        ed._sample_plot_enabled[nm] = (state != 'none_enabled')
        ed._sample_gates.setdefault(nm, {})
        ed._sample_gate_order.setdefault(nm, [])
        ed._sample_gate_seq.setdefault(nm, 0)
    if names and 'Day1' not in ed._trial_order:
        ed._trial_order.append('Day1')
    ed._channels = list(COLS) + ['cluster']
    ed._channel_labels = {c: c for c in ed._channels}
    ed._populate_channel_combos()
    ed.x_combo.set('FSC-A')
    ed.y_combo.set('SSC-A')
    ed._active_sample = 's1' if 's1' in ed._samples else None
    ed._gates = ed._sample_gates.get('s1', {})
    ed._gate_id_order, ed._gate_id_seq = [], 0
    if ed._active_sample:
        ed._add_gate({'kind': 'threshold', 'channel': 'FITC-A', 'op': '>',
                      'value': 25000.0, 'name': 'FITC+'}, audit=False)
    return root, ed


def _windows():
    """Window classes constructible from the editor alone."""
    import importlib
    import tkinter as tk
    out = []
    for mod in pkgutil.iter_modules(openflo.__path__):
        if not mod.name.startswith('ui_'):
            continue
        m = importlib.import_module(f'openflo.{mod.name}')
        fd = getattr(m, 'filedialog', None)
        if fd is not None:                            # never block on a dialog
            for fn in ('asksaveasfilename', 'askopenfilename', 'askdirectory',
                       'askopenfilenames'):
                if hasattr(fd, fn):
                    setattr(fd, fn, lambda *a, **k: '')
        for cname, cls in vars(m).items():
            if not (inspect.isclass(cls) and cls.__module__ == m.__name__
                    and issubclass(cls, tk.Misc)):
                continue
            try:
                sig = inspect.signature(cls.__init__)
                required = [
                    p for p in list(sig.parameters)[1:]
                    if sig.parameters[p].default is inspect.Parameter.empty
                    and sig.parameters[p].kind not in
                    (inspect.Parameter.VAR_POSITIONAL,
                     inspect.Parameter.VAR_KEYWORD)]
            except (TypeError, ValueError):
                continue
            if len(required) <= 1:
                out.append((f'{mod.name}.{cname}', cls))
    return out


def _zero_arg_actions(w):
    for name in dir(w):
        if not name.startswith(ACTION_PREFIXES):
            continue
        fn = getattr(w, name, None)
        if not callable(fn):
            continue
        try:
            ps = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.default is inspect.Parameter.empty
               and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD)
               for p in ps.values()):
            continue
        yield name, fn


@pytest.mark.parametrize('state', ['full', 'none_enabled', 'no_samples'])
def test_dialogs_survive_construction_and_their_actions(state):
    root, ed = _editor(state)
    failures = []
    try:
        for label, cls in _windows():
            try:
                w = cls(ed)
                w.withdraw()
            except Exception as exc:                  # noqa: BLE001
                failures.append(f'{label} failed to CONSTRUCT: '
                                f'{type(exc).__name__}: {exc}')
                continue
            for mname, fn in _zero_arg_actions(w):
                try:
                    fn()
                    root.update_idletasks()
                except Exception as exc:              # noqa: BLE001
                    failures.append(f'{label}.{mname}: '
                                    f'{type(exc).__name__}: {exc}')
            try:
                w.destroy()
            except Exception:                         # noqa: BLE001
                pass
    finally:
        root.destroy()
    assert not failures, (
        f'{len(failures)} dialog action(s) raised with state={state!r}:\n  '
        + '\n  '.join(failures[:12]))


def test_the_sweep_actually_reaches_some_windows():
    """Guard the guard: if the discovery ever returns nothing, the test above
    would pass vacuously — which is the failure mode this suite exists to
    catch elsewhere."""
    found = _windows()
    assert len(found) >= 10, f'only discovered {len(found)} windows: {found}'
