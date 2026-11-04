"""One name per action (best-practices finding N9): the renamed exports,
parsers and mixins keep their behaviour, and the old public names still work
with a DeprecationWarning.

Every input here is generated: synthetic detector spectra and synthetic
channel names (FL1-A, FL2-A, ...). One test builds the voltage dialog on a
stub editor to read its export tooltip; it opens no file picker. Every other
window class is only inspected.
"""
import importlib
import inspect
import warnings

import numpy as np
import pytest

from openflo import acquisition, events_csv, session_models
from openflo.spectral_af import AF_MAX_EVENTS, af_signatures
from tests.conftest import gui_unavailable

# ── af_signatures: n_max -> max_events ──────────────────────────────────────

N_EVENTS = 4000          # bright quarter: ~1000 events
CAP = 300                # well below the bright-quarter count, so it bites


def _unstained(seed=1):
    """Two synthetic autofluorescence shapes over 8 detectors, with noise."""
    rng = np.random.default_rng(seed)
    det = np.arange(8)
    shapes = np.stack([np.exp(-0.5 * ((det - 2) / 1.2) ** 2),
                       np.exp(-0.5 * ((det - 5.5) / 1.5) ** 2)])
    which = rng.integers(0, 2, N_EVENTS)
    scale = rng.lognormal(6.0, 0.8, N_EVENTS)[:, None]
    return shapes[which] * scale + rng.normal(0, 20, (N_EVENTS, 8))


def _no_deprecation(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        return fn(*args, **kwargs)


def test_max_events_and_the_deprecated_n_max_give_the_same_signatures():
    U = _unstained()
    new = _no_deprecation(af_signatures, U, k=2, max_events=CAP)
    with pytest.warns(DeprecationWarning, match='max_events'):
        old = af_signatures(U, k=2, n_max=CAP)
    assert new.shape[1] == 8 and len(new) >= 1
    np.testing.assert_array_equal(new, old)


def test_the_event_cap_is_applied():
    U = _unstained()
    capped = af_signatures(U, k=2, max_events=CAP)
    uncapped = af_signatures(U, k=2, max_events=10 * N_EVENTS)
    # A cap at or above the bright-quarter count leaves the run unchanged...
    np.testing.assert_array_equal(
        uncapped, af_signatures(U, k=2, max_events=N_EVENTS))
    # ...while one below it clusters a subsample, so the medians move.
    assert capped.shape != uncapped.shape or not np.array_equal(capped, uncapped)


def test_max_events_is_still_the_fifth_positional_argument():
    U = _unstained()
    positional = _no_deprecation(af_signatures, U, 2, 0, 50, CAP)
    np.testing.assert_array_equal(
        positional, af_signatures(U, k=2, seed=0, iters=50, max_events=CAP))


def test_passing_both_max_events_and_n_max_is_an_error():
    U = _unstained()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(TypeError) as err:
            af_signatures(U, k=2, max_events=CAP, n_max=CAP)
    assert 'max_events' in str(err.value) and 'n_max' in str(err.value)


def test_the_default_cap_is_af_max_events_and_unchanged():
    assert AF_MAX_EVENTS == 20000
    # Above the bright-quarter count the cap never bites, so a run on these
    # events could not tell the default from any other large cap. Repeat the
    # events until the bright quarter exceeds AF_MAX_EVENTS (with jitter so
    # the copies differ), then the default must match the explicit constant
    # and differ from a neighbouring cap.
    rng = np.random.default_rng(4)
    U = np.tile(_unstained(), (25, 1))
    U = U + rng.normal(0, 1, U.shape)
    assert len(U) // 4 > AF_MAX_EVENTS
    default = _no_deprecation(af_signatures, U, k=2)
    np.testing.assert_array_equal(
        default, af_signatures(U, k=2, max_events=AF_MAX_EVENTS))
    other = af_signatures(U, k=2, max_events=AF_MAX_EVENTS + 1)
    assert default.shape != other.shape or not np.array_equal(default, other)


# ── VoltageDialog export button: tooltip names the figure formats ──────────

def test_voltage_export_tooltip_names_figure_formats():
    """The Export… button saves a figure, so its hover tip must say so (it
    used to promise a CSV). Read through the real path: auto_tip resolves
    the stub editor from the widget tree and calls its ``_tip``."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                          # noqa: BLE001 - any Tk start failure means no GUI
        gui_unavailable(f"Tk cannot initialise without a display: {e}")
    root.withdraw()
    texts = {}
    root._tip = lambda widget, text: texts.__setitem__(str(widget), text)
    try:
        from openflo.ui_voltage import VoltageDialog
        dlg = VoltageDialog(root)
        text = texts.get(str(dlg._export_btn))
        assert text, 'the Export… button has no tooltip'
        assert 'CSV' not in text.upper()
        assert 'PNG' in text
    finally:
        root.destroy()


# ── editor mixins: old names are deprecated aliases ─────────────────────────

@pytest.mark.parametrize('modname, old, new', [
    ('openflo.editor_combos', 'CombinationsMixin', 'CombosMixin'),
    ('openflo.editor_console', 'ConsoleLogMixin', 'ConsoleMixin'),
])
def test_old_mixin_names_warn_and_are_the_new_classes(modname, old, new):
    mod = importlib.import_module(modname)
    cls = _no_deprecation(getattr, mod, new)
    assert inspect.isclass(cls) and cls.__name__ == new
    with pytest.warns(DeprecationWarning, match=new):
        alias = getattr(mod, old)
    assert alias is cls
    with pytest.raises(AttributeError):
        mod.NoSuchMixin  # noqa: B018


# ── window exports: _export_figure / _export_csv ────────────────────────────

FIGURE_WINDOWS = [
    ('ui_figure_window', 'FigureWindow'),
    ('ui_plate', 'PlateWindow'),
    ('ui_bead_tracking', 'BeadTrackingWindow'),
    ('ui_spectral_qc', 'SpectralQCWindow'),
    ('ui_preview', 'QuickPreviewDialog'),
    ('ui_voltage', 'VoltageDialog'),
    ('ui_unmixed_pairs', 'UnmixedPairsWindow'),
    ('ui_flowsom_tree', 'FlowSOMTreeWindow'),
    ('ui_comp_spread', 'SpreadMatrixWindow'),
    ('ui_proliferation', 'ProliferationWindow'),
    ('ui_sample_qc', 'SampleQCWindow'),
]
TABLE_WINDOWS = [
    ('ui_compare', 'CompareWspDialog'),
    ('ui_diff', 'DiffAbundanceWindow'),
    ('ui_annotation', 'PopulationAnnotationWindow'),
]
DISPATCHERS = [('ui_trajectory', 'TrajectoryWindow'), ('ui_audit', 'AuditWindow')]


def _cls(modname, clsname):
    return getattr(importlib.import_module(f'openflo.{modname}'), clsname)


def _no_vague_export(cls):
    assert not hasattr(cls, '_export_png'), f'{cls.__name__}._export_png'
    assert not hasattr(cls, '_export_fig'), f'{cls.__name__}._export_fig'
    export = getattr(cls, '_export', None)
    if export is not None:
        params = list(inspect.signature(export).parameters)
        assert len(params) > 1, f'{cls.__name__}._export takes no format'


@pytest.mark.parametrize('modname, clsname', FIGURE_WINDOWS)
def test_figure_windows_export_with_export_figure(modname, clsname):
    cls = _cls(modname, clsname)
    assert callable(getattr(cls, '_export_figure', None))
    _no_vague_export(cls)


@pytest.mark.parametrize('modname, clsname', TABLE_WINDOWS)
def test_table_windows_export_with_export_csv(modname, clsname):
    cls = _cls(modname, clsname)
    assert callable(getattr(cls, '_export_csv', None))
    _no_vague_export(cls)


@pytest.mark.parametrize('modname, clsname', DISPATCHERS)
def test_export_dispatchers_keep_their_format_argument(modname, clsname):
    params = list(inspect.signature(_cls(modname, clsname)._export).parameters)
    assert len(params) == 2


# ── parsers named for what they return ──────────────────────────────────────

@pytest.mark.parametrize('modname', ['saturation', 'gatingml', 'ui_panel_check',
                                     'ui_kinetics'])
def test_finite_float_parsers(modname):
    parse = importlib.import_module(f'openflo.{modname}')._finite_float
    assert parse('  7 ') == 7.0
    assert parse(2.5) == 2.5
    assert parse('-1e3') == -1000.0
    for bad in ('inf', '-inf', 'nan', 'x', '', None):
        assert parse(bad) is None, bad


def test_rounded_number():
    assert acquisition._rounded_number(1.23456) == 1.235
    assert acquisition._rounded_number('450') == 450.0
    assert acquisition._rounded_number('x') is None
    assert acquisition._rounded_number(None) is None


def test_require_number():
    session_models._require_number({'n': 3}, 'n')
    session_models._require_number({'n': 0.5}, 'n')
    for bad in (True, 'x', None):
        with pytest.raises(ValueError, match="'n' is not a number"):
            session_models._require_number({'n': bad}, 'n')


def test_the_old_parser_names_are_gone():
    for modname in ('saturation', 'gatingml', 'ui_panel_check', 'ui_kinetics'):
        mod = importlib.import_module(f'openflo.{modname}')
        assert not hasattr(mod, '_num') and not hasattr(mod, '_number'), modname
    assert not hasattr(acquisition, '_number')
    assert not hasattr(session_models, '_number')


# ── read_events_csv uses compensation_io.is_number ──────────────────────────

def _events(seed=2):
    return np.random.default_rng(seed).uniform(1, 1000, (20, 3)).round(3)


def test_read_events_csv_takes_names_from_a_savetxt_header(tmp_path):
    data = _events()
    path = tmp_path / 'sample_1.csv'
    np.savetxt(path, data, delimiter=',', header='FL1-A,FL2-A,FL3-A', fmt='%.3f')
    df, notes = events_csv.read_events_csv(str(path))
    assert list(df.columns) == ['FL1-A', 'FL2-A', 'FL3-A']
    np.testing.assert_allclose(df.to_numpy(), data)
    assert any("'#' line" in n for n in notes)


def test_read_events_csv_keeps_the_first_row_of_a_headerless_file(tmp_path):
    data = _events(3)
    path = tmp_path / 'sample_2.csv'
    np.savetxt(path, data, delimiter=',', fmt='%.3f')
    df, notes = events_csv.read_events_csv(str(path))
    assert list(df.columns) == ['Ch1', 'Ch2', 'Ch3']
    assert len(df) == len(data)
    np.testing.assert_allclose(df.to_numpy(), data)
    assert any('no header row' in n for n in notes)
