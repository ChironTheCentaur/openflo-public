"""First run of the desktop app: the welcome window's flag, the bundled
sample and the getting-started tour's steps.

No Tk here; the windows are in ui_first_run and the editor's Help mixin.

The welcome shows once: on the first launch of ``openflo-gui`` (gui.main),
never when the editor is built any other way (tests, scripts, the CLI), and
never with ``OPENFLO_NO_WELCOME`` set. Help, Getting started tour and Help,
Open the sample data reopen what it offers.

The sample is the phone tutorial's (openflo.tutorial_sample) at fewer
events, written by scripts/make_desktop_sample.py into the package. It is
read in place and unpacked into a new folder under ``~/.openflo/sample``, so
opening it never writes into the install folder.
"""
from __future__ import annotations

import os
from importlib import resources

# Set once the welcome has been shown (or turned off): it does not show again.
PREF_KEY = 'welcome_done'
# Set to any value to keep the welcome from showing (kiosks, demos, scripts).
NO_WELCOME_ENV = 'OPENFLO_NO_WELCOME'

SAMPLE_FILE = 'openflo-sample.flowproject'
SAMPLE_EVENTS = 5_000
SAMPLE_NAMES = ('Control', 'Treated')
# The sample the plot shows when it opens (the gated plot the welcome promises).
SAMPLE_PLOTTED = ('Control',)
# The folder each opened copy of the sample is unpacked under, and the
# copy's name (a second copy gets " (2)", as unpacking an archive does).
SAMPLE_FOLDER = 'sample'
SAMPLE_COPY_NAME = 'OpenFlo sample'

# (target, title, text): target names the editor widget the step points at
# (ui_first_run.tour_target).
TOUR_STEPS = (
    ('files', "Open your files",
     "Use File > Add FCS to load your own .fcs files, or drag them onto the "
     "window. File > Open session reopens saved work."),
    ('tree', "Pick a population",
     "This tree lists each sample and its gates. Click a population to plot "
     "only its events."),
    ('axes', "Change the axes",
     "Choose the channel for each axis here. The plot redraws at once."),
    ('gates', "Draw a gate",
     "Pick a shape, then drag on the plot. The new gate appears in the tree "
     "under the population you picked."),
    ('files', "Export your results",
     "File > Export saves statistics, plot images, FCS files and FlowJo "
     "workspaces. File > Save session keeps your gates for next time."),
)


def should_welcome(prefs, environ=None):
    """Whether the welcome window shows now, given the stored `prefs`
    (prefs.read_prefs()): not yet shown or turned off, and not kept off by
    ``OPENFLO_NO_WELCOME`` or a running test."""
    environ = os.environ if environ is None else environ
    if environ.get(NO_WELCOME_ENV) or environ.get('PYTEST_CURRENT_TEST'):
        return False
    return not (isinstance(prefs, dict) and prefs.get(PREF_KEY))


def bundled_sample():
    """The sample archive shipped in the package (a Traversable)."""
    return resources.files('openflo') / 'sample_data' / SAMPLE_FILE


def sample_root():
    """``~/.openflo/sample``: where opened copies of the sample go."""
    return os.path.join(os.path.expanduser('~'), '.openflo', SAMPLE_FOLDER)


def unpack_sample(root=None):
    """Unpack the bundled sample into a new folder under `root`
    (sample_root() by default) and return its session file's path. The
    archive is only read; nothing is written beside it."""
    from .project_archive import open_project_archive, unpack_folder_for
    root = root or sample_root()
    os.makedirs(root, exist_ok=True)
    dest = unpack_folder_for(os.path.join(root, SAMPLE_COPY_NAME + '.flowproject'))
    with resources.as_file(bundled_sample()) as path:
        session, _manifest = open_project_archive(os.fspath(path), dest)
    return session


def tour_step_text(index):
    """The tour's header for step `index` (from 0), e.g. "Step 2 of 5"."""
    return f"Step {index + 1} of {len(TOUR_STEPS)}"
