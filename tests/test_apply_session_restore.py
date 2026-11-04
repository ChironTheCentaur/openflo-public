"""What ``SessionMixin._apply_session`` restores from a parsed session dict.

Runs the method unbound on a stand-in editor (no Tk): plain attributes for
the editor's state, recording stand-ins for its Tk variables, the audit
window and the hooks it calls (tab restore, sample loading, ``after``).
Covers the channel display, the provenance trail, cluster labels, the
tabs and layout pages, the display settings, the staged combo restore,
the sample load and the two deferred callbacks. Synthetic names only."""
from __future__ import annotations

import pytest

from openflo import editor_session
from openflo.editor_session import SessionMixin

X, Y, C = 'FL1-A', 'FL2-A', 'FL3-A'

# The pieces _apply_session is made of, borrowed from the real mixin when it
# has them (older code did all of it inline in _apply_session).
_HELPERS = ('_restore_channel_display', '_restore_provenance',
            '_restore_cluster_labels', '_restore_view_pages',
            '_restore_display_settings', '_stage_session_restore')


class Var:
    """A Tk variable; ``boom`` makes ``set`` raise."""

    def __init__(self, value=None, boom=False):
        self.value = value
        self.boom = boom

    def set(self, value):
        if self.boom:
            raise RuntimeError('variable is gone')
        self.value = value

    def get(self):
        return self.value


class AuditWin:
    """The History window: ``exists`` is what ``winfo_exists`` answers."""

    def __init__(self, exists=True):
        self.exists = exists
        self.refreshed = 0

    def winfo_exists(self):
        return self.exists

    def refresh(self):
        self.refreshed += 1


class Ed:
    """The editor state ``_apply_session`` touches, and a log of the hooks
    it calls, in order."""

    PLOT_MODES = ('dot', 'pseudocolor', 'contour', 'histogram')

    def __init__(self, missing=(), tabs_error=None):
        self._fmo_controls = {'FL1-A': 'old control'}
        self._channel_scale = {}
        self._channel_range = {}
        self._channel_labels = {}
        self._session_channel_labels = {}
        self._audit_log = None
        self._applied_fixes = []
        self._declined_fixes = []
        self._audit_window = None
        self._cluster_labels = {}
        self._layout_doc = 'previous layout'
        self._pending_sample_meta = {'old.fcs': 'meta'}
        self._pending_sample_prep = {'old.fcs': 'prep'}
        self._session_restore = None
        self.ds_display_var = Var(False)
        self.ds_propagate_var = Var(True)
        self.max_points_var = Var('50000')
        self.show_removed_var = Var(True)
        self.contour_scatter_var = Var(False)
        self.contour_outliers_var = Var(False)
        self.hist_y_mode = Var('Count')
        self.mode_var = Var('dot')
        self.missing = list(missing)
        self.tabs_error = tabs_error
        self.calls = []

    def _audit(self, action, **details):
        self.calls.append(('audit', action, details))
        self._audit_log.record(action, details=details)

    def _restore_tabs_state(self, state):
        self.calls.append(('tabs', state))
        if self.tabs_error is not None:
            raise self.tabs_error

    def _sync_ds_mode_var(self):
        self.calls.append(('sync_ds',))

    def _update_ds_visibility(self):
        self.calls.append(('ds_visibility',))

    def _update_mode_options(self):
        self.calls.append(('mode_options',))

    def _restore_samples(self, samples, sample_gates, active):
        self.calls.append(('restore_samples', samples, sample_gates, active))
        return self.missing

    def after(self, ms, fn):
        self.calls.append(('after', ms, fn))

    def _offer_locate_missing(self):
        pass

    def _apply_session_restore(self):
        pass

    def named(self, kind):
        return [c for c in self.calls if c[0] == kind]


for _name in _HELPERS:
    if hasattr(SessionMixin, _name):
        setattr(Ed, _name, getattr(SessionMixin, _name))


@pytest.fixture
def notes(monkeypatch):
    """What ``_apply_session`` reports through ``note``."""
    seen = []
    monkeypatch.setattr(editor_session, 'note',
                        lambda what, exc=None: seen.append((what, exc)))
    return seen


@pytest.fixture
def layouts(monkeypatch):
    """``LayoutDoc.from_dict`` replaced by a recorder; set ``error`` to make
    it raise."""
    from openflo import layout_page
    rec = {'seen': [], 'error': None}

    def from_dict(d):
        rec['seen'].append(d)
        if rec['error'] is not None:
            raise rec['error']
        return ('layout', d)

    monkeypatch.setattr(layout_page.LayoutDoc, 'from_dict', from_dict)
    return rec


def apply(data, **kw):
    ed = Ed(**kw)
    SessionMixin._apply_session(ed, data)
    return ed


def sample(name='sample_1', **kw):
    return {'name': name, 'path': f'{name}.fcs', **kw}


# ── channel display ─────────────────────────────────────────────────────────

def test_channel_scales_are_stringified():
    """Saved scales come back as strings, merged into the editor's."""
    ed = Ed()
    ed._channel_scale = {C: 'linear'}
    SessionMixin._apply_session(ed, {'channel_scale': {X: 'logicle', Y: 5}})
    assert ed._channel_scale == {C: 'linear', X: 'logicle', Y: '5'}


def test_valid_channel_range_becomes_float_tuple():
    """A two-item range is kept as a tuple of floats."""
    ed = apply({'channel_range': {X: ['1', 2], Y: (0, 1e5)}})
    assert ed._channel_range == {X: (1.0, 2.0), Y: (0.0, 100000.0)}
    assert all(isinstance(v, float) for v in ed._channel_range[X])


def test_channel_range_of_wrong_length_is_skipped():
    """A range without exactly two items, or not a list, is skipped and the
    good ones still load."""
    ed = apply({'channel_range': {X: [1, 2, 3], Y: [1], C: 'wide',
                                  'FL4-A': [0, 10]}})
    assert ed._channel_range == {'FL4-A': (0.0, 10.0)}


def test_non_numeric_channel_range_is_skipped():
    """A range that is not numbers is skipped without aborting the load."""
    ed = apply({'channel_range': {X: ['low', 'high'], Y: [None, 1],
                                  C: [0, 10]},
                'samples': [sample()]})
    assert ed._channel_range == {C: (0.0, 10.0)}
    assert ed.named('restore_samples')


def test_session_channel_labels_win_over_editor_labels():
    """The session's labels overwrite the editor's for the same channel."""
    ed = Ed()
    ed._channel_labels = {X: 'editor', Y: 'kept'}
    SessionMixin._apply_session(ed, {'channel_labels': {X: 'CD3', C: 7}})
    assert ed._channel_labels == {X: 'CD3', Y: 'kept', C: '7'}


def test_session_channel_labels_are_kept_apart():
    """The session's own labels are kept in a separate dict, stringified."""
    ed = Ed()
    ed._channel_labels = {Y: 'editor'}
    SessionMixin._apply_session(ed, {'channel_labels': {X: 'CD3', C: 7}})
    assert ed._session_channel_labels == {X: 'CD3', C: '7'}
    assert ed._session_channel_labels is not ed._channel_labels


def test_fmo_controls_are_reset():
    """The previous session's FMO controls are dropped."""
    assert apply({})._fmo_controls == {}


# ── provenance ──────────────────────────────────────────────────────────────

def test_audit_log_is_restored_then_load_is_audited():
    """The saved trail comes back and the load itself is appended to it."""
    trail = [{'seq': 1, 'time': 't0', 'action': 'gate.add',
              'details': {'sample': 'sample_1'}}]
    ed = apply({'audit': trail, 'created': '2026-01-02',
                'samples': [sample(), sample('sample_2')]})
    actions = [(e['action'], e['details']) for e in ed._audit_log.entries()]
    assert actions == [
        ('gate.add', {'sample': 'sample_1'}),
        ('session.load', {'created': '2026-01-02', 'n_samples': 2})]


def test_load_audit_defaults_without_created_or_samples():
    """With no creation time or samples the load is audited as ('', 0)."""
    ed = apply({})
    assert ed.named('audit') == [
        ('audit', 'session.load', {'created': '', 'n_samples': 0})]


def test_fixes_lists_are_restored_as_strings():
    """The applied and declined fixes come back as lists of strings."""
    ed = apply({'fixes_applied': ['fix.a', 3], 'fixes_declined': ['fix.b']})
    assert ed._applied_fixes == ['fix.a', '3']
    assert ed._declined_fixes == ['fix.b']


def test_missing_fixes_lists_become_empty():
    """No fixes in the session clears the editor's lists."""
    ed = Ed()
    ed._applied_fixes = ['stale']
    ed._declined_fixes = ['stale']
    SessionMixin._apply_session(ed, {'fixes_applied': None})
    assert (ed._applied_fixes, ed._declined_fixes) == ([], [])


def test_open_audit_window_is_refreshed():
    """An open History window is refreshed once."""
    ed = Ed()
    ed._audit_window = AuditWin(exists=True)
    SessionMixin._apply_session(ed, {})
    assert ed._audit_window.refreshed == 1


def test_closed_audit_window_is_not_refreshed():
    """A History window that no longer exists is left alone."""
    ed = Ed()
    ed._audit_window = AuditWin(exists=False)
    SessionMixin._apply_session(ed, {})
    assert ed._audit_window.refreshed == 0


# ── cluster labels ──────────────────────────────────────────────────────────

def test_cluster_ids_are_coerced_to_int():
    """Stringified cluster ids come back as ints."""
    ed = apply({'cluster_labels': {'sample_1': {'0': 'T cells', '12': 'B'}}})
    assert ed._cluster_labels == {'sample_1': {0: 'T cells', 12: 'B'}}


def test_non_numeric_cluster_id_is_kept_as_is():
    """A key that is not a number is kept unchanged."""
    ed = apply({'cluster_labels': {'sample_1': {'0': 'T', 'noise': 'N'}}})
    assert ed._cluster_labels['sample_1'] == {0: 'T', 'noise': 'N'}


def test_non_dict_cluster_labels_are_skipped():
    """A sample whose labels are not a dict is skipped; the rest load."""
    ed = Ed()
    ed._cluster_labels = {'sample_3': {1: 'kept'}}
    SessionMixin._apply_session(ed, {'cluster_labels': {
        'sample_1': ['bad'], 'sample_2': {'1': 'NK'}}})
    assert ed._cluster_labels == {'sample_3': {1: 'kept'},
                                  'sample_2': {1: 'NK'}}


# ── tabs and layout ─────────────────────────────────────────────────────────

def test_view_tabs_are_handed_to_tab_restore():
    """The saved tabs go to ``_restore_tabs_state``."""
    ed = apply({'view_tabs': {'tabs': ['Tab 1']}})
    assert ed.named('tabs') == [('tabs', {'tabs': ['Tab 1']})]


def test_tabs_failure_is_printed_and_load_continues(capsys):
    """A tab restore that raises is printed; the rest still loads."""
    ed = apply({'samples': [sample()], 'x_channel': X},
               tabs_error=RuntimeError('bad tabs'))
    assert '[session] tabs not restored: bad tabs' in capsys.readouterr().out
    assert ed._session_restore['x'] == X
    assert ed.named('restore_samples')


def test_layout_is_rebuilt_from_dict(layouts):
    """A saved layout is rebuilt with ``LayoutDoc.from_dict``."""
    ed = apply({'layout': {'pages': [1]}})
    assert ed._layout_doc == ('layout', {'pages': [1]})


def test_empty_layout_is_not_restored(layouts):
    """An empty layout leaves the editor's layout as it was."""
    ed = apply({'layout': {}})
    assert layouts['seen'] == []
    assert ed._layout_doc == 'previous layout'


def test_layout_failure_is_printed_and_load_continues(layouts, capsys):
    """A layout that will not rebuild is printed; the rest still loads."""
    layouts['error'] = ValueError('not a layout')
    ed = apply({'layout': {'pages': [1]}, 'samples': [sample()]})
    out = capsys.readouterr().out
    assert '[session] layout not restored: not a layout' in out
    assert ed._layout_doc == 'previous layout'
    assert ed.named('restore_samples')


# ── display settings ────────────────────────────────────────────────────────

def test_display_settings_are_restored():
    """Each saved display setting reaches its variable."""
    ed = apply({'downsample_display': False, 'downsample_propagate': True,
                'max_points': 20000, 'show_removed': True,
                'contour_scatter': False, 'contour_outliers': False,
                'hist_y_mode': '% of Max', 'plot_mode': 'contour'})
    assert ed.ds_display_var.value is False
    assert ed.ds_propagate_var.value is True
    assert ed.max_points_var.value == '20000'
    assert ed.show_removed_var.value is True
    assert ed.contour_scatter_var.value is False
    assert ed.contour_outliers_var.value is False
    assert ed.hist_y_mode.value == '% of Max'
    assert ed.mode_var.value == 'contour'


def test_display_settings_default_when_absent():
    """Absent toggles take their defaults."""
    ed = apply({})
    assert ed.ds_display_var.value is True
    assert ed.ds_propagate_var.value is False
    assert ed.show_removed_var.value is False
    assert ed.contour_scatter_var.value is True
    assert ed.contour_outliers_var.value is True


def test_display_hooks_run_in_order():
    """Downsampling is synced and shown, then mode options refreshed."""
    ed = apply({})
    hooks = [c[0] for c in ed.calls
             if c[0] in ('sync_ds', 'ds_visibility', 'mode_options')]
    assert hooks == ['sync_ds', 'ds_visibility', 'mode_options']


@pytest.mark.parametrize('value', [None, 0, ''])
def test_max_points_only_set_when_given(value):
    """An empty max_points leaves the editor's value."""
    assert apply({'max_points': value}).max_points_var.value == '50000'


@pytest.mark.parametrize('mode', ['Fraction', 'Count', '% of Max'])
def test_valid_hist_y_mode_is_restored(mode):
    """Each known histogram y mode is restored."""
    ed = Ed()
    ed.hist_y_mode.value = None
    SessionMixin._apply_session(ed, {'hist_y_mode': mode})
    assert ed.hist_y_mode.value == mode


def test_unknown_hist_y_mode_is_ignored():
    """An unknown histogram y mode leaves the editor's."""
    assert apply({'hist_y_mode': 'Density'}).hist_y_mode.value == 'Count'


def test_unknown_plot_mode_is_ignored():
    """A plot mode not in PLOT_MODES leaves the editor's."""
    assert apply({'plot_mode': 'violin'}).mode_var.value == 'dot'


def test_failing_display_setting_is_noted_and_load_continues(notes):
    """A variable that raises is reported via note; the rest still loads."""
    ed = Ed()
    ed.show_removed_var = Var(boom=True)
    SessionMixin._apply_session(ed, {'samples': [sample()], 'y_channel': Y})
    assert [w for w, _ in notes] == [
        'restore display settings from the session file']
    assert isinstance(notes[0][1], RuntimeError)
    assert ed._session_restore['y'] == Y
    assert ed._pending_sample_meta == {}
    assert ed.named('restore_samples')


# ── staged restore and sample load ──────────────────────────────────────────

def test_pending_sample_bundles_are_cleared():
    """A previous session's staged meta and prep are cleared in place."""
    ed = Ed()
    meta, prep = ed._pending_sample_meta, ed._pending_sample_prep
    SessionMixin._apply_session(ed, {})
    assert ed._pending_sample_meta is meta and meta == {}
    assert ed._pending_sample_prep is prep and prep == {}


def test_session_restore_is_staged():
    """Combos, active sample and plot toggles are staged for later."""
    ed = apply({'x_channel': X, 'y_channel': Y, 'color_channel': C,
                'active_sample': 'sample_2',
                'samples': [sample(plot_enabled=True), sample('sample_2')]})
    assert ed._session_restore == {
        'x': X, 'y': Y, 'color': C, 'active': 'sample_2',
        'plot_enabled': {'sample_1': True, 'sample_2': False}}


def test_session_restore_without_samples():
    """An empty session stages Nones and no plot toggles."""
    assert apply({})._session_restore == {
        'x': None, 'y': None, 'color': None, 'active': None,
        'plot_enabled': {}}


def test_samples_are_loaded_with_gates_and_active():
    """``_restore_samples`` gets the samples, their gates and the active
    sample."""
    samples = [sample(), sample('sample_2')]
    gates = {'sample_1': [{'name': 'P1'}]}
    ed = apply({'samples': samples, 'sample_gates': gates,
                'active_sample': 'sample_1'})
    assert ed.named('restore_samples') == [
        ('restore_samples', samples, gates, 'sample_1')]


def test_samples_load_defaults_when_absent():
    """No samples or gates loads ``([], {}, None)``."""
    ed = apply({'sample_gates': None})
    assert ed.named('restore_samples') == [('restore_samples', [], {}, None)]


def test_missing_samples_offer_locate_then_restore():
    """Missing files schedule the locate offer, then the combo restore."""
    ed = apply({'samples': [sample()]}, missing=['sample_1'])
    assert ed.named('after') == [
        ('after', 800, ed._offer_locate_missing),
        ('after', 600, ed._apply_session_restore)]


def test_no_missing_samples_only_schedules_restore():
    """With every file found only the combo restore is scheduled."""
    ed = apply({'samples': [sample()]})
    assert ed.named('after') == [('after', 600, ed._apply_session_restore)]


def test_display_state_is_restored_before_samples_load():
    """Everything is restored and staged before samples start loading."""
    ed = apply({'samples': [sample()]})
    kinds = [c[0] for c in ed.calls]
    assert kinds.index('mode_options') < kinds.index('restore_samples')
    assert kinds.index('audit') < kinds.index('restore_samples')
    assert kinds.index('restore_samples') < kinds.index('after')


# ── restore order ───────────────────────────────────────────────────────────

class LoggedVar(Var):
    """A Tk variable that notes each ``set`` in the editor's event log."""

    def __init__(self, log, name, value=None):
        super().__init__(value)
        self.log, self.name = log, name

    def set(self, value):
        self.log.append(('set', self.name))
        super().set(value)


class LoggedDict(dict):
    """A dict that notes each change in the editor's event log."""

    def __init__(self, log, name, *a):
        super().__init__(*a)
        self.log, self.name = log, name

    def __setitem__(self, key, value):
        self.log.append(('set', self.name))
        super().__setitem__(key, value)

    def update(self, *a, **kw):
        self.log.append(('set', self.name))
        super().update(*a, **kw)

    def setdefault(self, key, default=None):
        self.log.append(('set', self.name))
        return super().setdefault(key, default)


class OrderedEd(Ed):
    """``Ed`` whose state changes land in the same ordered log as its hook
    calls, so the order the parts are restored in can be read off."""

    def __init__(self, **kw):
        super().__init__(**kw)
        log = self.calls
        self._channel_labels = LoggedDict(log, 'channel_labels')
        self._cluster_labels = LoggedDict(log, 'cluster_labels')
        self.ds_display_var = LoggedVar(log, 'ds_display_var', False)
        self._logging = True

    def __setattr__(self, name, value):
        if name == '_session_restore' and self.__dict__.get('_logging'):
            self.calls.append(('set', name))
        super().__setattr__(name, value)


def events(data):
    """The ordered event names one restore of ``data`` produces."""
    ed = OrderedEd()
    SessionMixin._apply_session(ed, data)
    return [c[1] if c[0] == 'set' else c[0] for c in ed.calls]


FULL = {'channel_labels': {X: 'CD3'},
        'cluster_labels': {'sample_1': {'0': 'T cells'}},
        'view_tabs': {'tabs': ['Tab 1']}, 'samples': [sample()]}


def test_channel_display_restored_before_load_is_audited():
    """The channel labels are in place before the load is audited."""
    ev = events(FULL)
    assert ev.index('channel_labels') < ev.index('audit')


def test_load_is_audited_before_cluster_labels_restored():
    """The load is audited before the cluster labels come back."""
    ev = events(FULL)
    assert ev.index('audit') < ev.index('cluster_labels')


def test_tabs_restored_before_display_settings():
    """The tabs are restored before the display settings are applied."""
    ev = events(FULL)
    assert ev.index('tabs') < ev.index('ds_display_var')


def test_display_settings_applied_before_restore_is_staged():
    """The display settings are finished before the combo restore is
    staged."""
    ev = events(FULL)
    assert ev.index('mode_options') < ev.index('_session_restore')


# ── provenance, more ────────────────────────────────────────────────────────

def test_null_audit_trail_restores_only_the_load_entry():
    """A null saved trail gives a trail holding just the load entry."""
    ed = apply({'audit': None})
    assert [e['action'] for e in ed._audit_log.entries()] == ['session.load']


class CountingAuditWin(AuditWin):
    """A History window that remembers how many entries it showed."""

    def __init__(self, ed):
        super().__init__(exists=True)
        self.ed = ed
        self.shown = None

    def refresh(self):
        super().refresh()
        self.shown = [e['action'] for e in self.ed._audit_log.entries()]


def test_refreshed_audit_window_shows_the_load_entry():
    """The History window is refreshed after the load is audited, so it
    shows the load entry."""
    ed = Ed()
    ed._audit_window = CountingAuditWin(ed)
    SessionMixin._apply_session(ed, {})
    assert ed._audit_window.shown == ['session.load']


# ── cluster labels, more ────────────────────────────────────────────────────

def test_fractional_cluster_id_is_kept_as_is():
    """A key like '1.5' is not an int id and is kept unchanged."""
    ed = apply({'cluster_labels': {'sample_1': {'1.5': 'odd', '2': 'B'}}})
    assert ed._cluster_labels['sample_1'] == {'1.5': 'odd', 2: 'B'}


def test_none_cluster_id_is_kept_as_is():
    """A None key is kept unchanged."""
    ed = apply({'cluster_labels': {'sample_1': {None: 'unassigned'}}})
    assert ed._cluster_labels['sample_1'] == {None: 'unassigned'}


def test_existing_sample_cluster_labels_are_replaced():
    """A sample's saved labels replace the ones the editor had for it."""
    ed = Ed()
    ed._cluster_labels = {'sample_1': {7: 'stale', 1: 'old'}}
    SessionMixin._apply_session(
        ed, {'cluster_labels': {'sample_1': {'1': 'NK'}}})
    assert ed._cluster_labels == {'sample_1': {1: 'NK'}}


# ── tabs and layout, more ───────────────────────────────────────────────────

def test_absent_view_tabs_pass_none_to_tab_restore():
    """With no saved tabs, ``_restore_tabs_state`` gets None."""
    assert apply({}).named('tabs') == [('tabs', None)]


@pytest.fixture
def prints(monkeypatch):
    """The module's ``print`` calls, as (text, kwargs)."""
    seen = []
    monkeypatch.setattr(editor_session, 'print',
                        lambda *a, **kw: seen.append((' '.join(map(str, a)),
                                                      kw)),
                        raising=False)
    return seen


def test_tabs_failure_is_printed_with_flush(prints):
    """The tabs failure message is flushed straight away."""
    apply({}, tabs_error=RuntimeError('bad tabs'))
    assert prints == [('[session] tabs not restored: bad tabs',
                       {'flush': True})]


def test_layout_failure_of_any_kind_is_caught(layouts, capsys):
    """A layout that fails with something other than ValueError is still
    printed and the rest loads."""
    layouts['error'] = KeyError('pages')
    ed = apply({'layout': {'pages': [1]}, 'samples': [sample()]})
    assert "[session] layout not restored: 'pages'" in capsys.readouterr().out
    assert ed._layout_doc == 'previous layout'
    assert ed.named('restore_samples')


# ── display settings, more ──────────────────────────────────────────────────

def test_downsample_display_defaults_on_when_absent():
    """No saved downsample_display turns display downsampling on."""
    ed = Ed()
    ed.ds_display_var.value = None
    SessionMixin._apply_session(ed, {})
    assert ed.ds_display_var.value is True
