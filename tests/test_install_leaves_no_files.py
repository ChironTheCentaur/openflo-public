"""A fresh install keeps nothing it does not need to.

Two things used to write inside the installed package folder (src/openflo, or
site-packages after a pip install), which is read-only on a system Python and
wiped by every upgrade:

* PhenoGraph's Louvain scratch files went to a ``.phenograph_cache`` folder
  there, and a run that failed left its files behind for good. Each call now
  runs in its own temporary folder, removed whatever happens.
* Saved gate templates went to a ``templates`` folder there. They now go to
  ``~/.openflo/templates``; the old folder is still read, never created.
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd
import pytest


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        np.vstack([rng.normal(m, 0.3, (150, 3)) for m in (0, 6)]
                  + [rng.normal(3, 2.0, (200, 3))]),
        columns=['A', 'B', 'C'])


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """A temp root and a working folder of the test's own, both empty."""
    import tempfile
    tmp_root = tmp_path / 'tmp'
    cwd = tmp_path / 'cwd'
    tmp_root.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_root))
    monkeypatch.chdir(cwd)
    return tmp_root, cwd


def _spy_phenograph(monkeypatch, fail=False):
    """Record where phenograph.cluster runs; write a scratch file there the
    way the Louvain backend does, and optionally fail after writing it."""
    ph = pytest.importorskip('phenograph')
    real = ph.cluster
    seen = []

    def spy(*a, **k):
        here = os.getcwd()
        seen.append(here)
        with open(os.path.join(here, 'leftover.bin'), 'wb') as f:
            f.write(b'x')
        if fail:
            raise RuntimeError('louvain died')
        return real(*a, **k)
    monkeypatch.setattr(ph, 'cluster', spy)
    return seen


def test_phenograph_runs_in_a_temp_folder_removed_after(isolated_dirs,
                                                       monkeypatch):
    from openflo.pipeline import FlowSample
    tmp_root, cwd = isolated_dirs
    seen = _spy_phenograph(monkeypatch)
    s = FlowSample.from_dataframe(_df())
    with contextlib.redirect_stdout(io.StringIO()):      # phenograph is chatty
        s.cluster(channels=['A', 'B', 'C'], k=15, n_jobs=1,
                  reproducible=False, use_gpu=False)
    assert (s.data['cluster'] >= 0).sum() > 0             # it still clusters
    assert len(seen) == 1
    assert os.path.dirname(seen[0]) == str(tmp_root)      # its own folder
    assert os.listdir(tmp_root) == []                     # ...and it is gone
    assert os.listdir(cwd) == []
    assert os.getcwd() == str(cwd)                        # cwd restored


def test_failed_phenograph_run_leaves_nothing(isolated_dirs, monkeypatch):
    from openflo.pipeline import ClusteringError, FlowSample
    tmp_root, cwd = isolated_dirs
    seen = _spy_phenograph(monkeypatch, fail=True)
    s = FlowSample.from_dataframe(_df())
    with pytest.raises(ClusteringError):
        s.cluster(channels=['A', 'B', 'C'], k=15, n_jobs=1,
                  reproducible=False, use_gpu=False)
    assert seen and not os.path.exists(seen[0])
    assert os.listdir(tmp_root) == []
    assert os.listdir(cwd) == []


@pytest.fixture
def export_mixin(tmp_path, monkeypatch):
    pytest.importorskip('tkinter')
    from openflo import editor_export
    home = tmp_path / 'home'
    pkg = tmp_path / 'pkg'
    home.mkdir()
    pkg.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))         # Windows
    monkeypatch.setattr(editor_export, 'BASE', str(pkg))
    return editor_export.ExportMixin, home, pkg


def test_saved_templates_live_in_the_home_folder(export_mixin):
    M, home, pkg = export_mixin
    d = M._templates_dir()
    assert os.path.normcase(d) == os.path.normcase(
        str(home / '.openflo' / 'templates'))
    assert os.path.isdir(d)
    M._bundled_templates()
    assert not (pkg / 'templates').exists()              # package untouched


def test_templates_saved_by_an_older_openflo_still_listed(export_mixin):
    import json
    M, home, pkg = export_mixin
    (pkg / 'templates').mkdir()
    with open(pkg / 'templates' / 'old_panel.json', 'w') as f:
        json.dump({'name': 'Old panel', 'gates': []}, f)
    with open(os.path.join(M._templates_dir(), 'mine.json'), 'w') as f:
        json.dump({'name': 'Mine', 'gates': []}, f)
    names = [n for n, _d, _p in M._bundled_templates()]
    assert 'Old panel' in names
    assert 'Mine' in names
