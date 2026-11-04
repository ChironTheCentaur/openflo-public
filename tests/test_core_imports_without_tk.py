"""The core library runs without Tk and without the CLI (best practices BP2).

Colouring groups (group_palettes), matching channels across samples and
clustering them together (shared_clustering) and the Pipeline Workspace's
run logic (workspace_run, which its job child runs) are Tk-free core: the
library uses them without importing theme, which imports tkinter and reads
preferences, or openflo.cli, which imports matplotlib.pyplot and caps thread
pools. theme keeps its preference-following wrappers; the GUI passes the
user's group palette to the core. Each check runs in a fresh interpreter
with tkinter blocked, so a stray import fails loudly.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from openflo import group_palettes, theme

_BLOCK_TK = textwrap.dedent("""
    import sys

    class _NoTk:
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in ('tkinter', '_tkinter'):
                raise ImportError('tkinter is blocked in this test')
            return None

    sys.modules['tkinter'] = None
    sys.meta_path.insert(0, _NoTk())
""")

_ASSERT_NO_GUI = textwrap.dedent("""
    for mod in ('tkinter', '_tkinter', 'openflo.theme', 'openflo.cli'):
        assert mod not in sys.modules or sys.modules[mod] is None, mod
    print('OK')
""")


def _run_without_tk(body: str) -> None:
    code = _BLOCK_TK + textwrap.dedent(body) + _ASSERT_NO_GUI
    res = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert res.returncode == 0, res.stderr[-4000:]
    assert res.stdout.strip().endswith('OK'), res.stdout


def test_categorical_cmap_needs_no_tk():
    _run_without_tk("""
        import matplotlib
        matplotlib.use('Agg')
        import openflo.pipeline as pipeline
        for name in ('auto', 'classic', 'okabe_ito'):
            for n in (1, 3, 12, 30):
                cm = pipeline.categorical_cmap(name, n)
                assert cm.N == n, (name, n, cm.N)
    """)


def test_cluster_jointly_and_label_alignment_need_no_tk_or_cli():
    _run_without_tk("""
        import logging
        logging.disable(logging.INFO)
        import numpy as np
        import pandas as pd
        from openflo.label_alignment import align_fluor_labels
        from openflo.pipeline import FlowExperiment, FlowSample

        rng = np.random.default_rng(3)

        def sample(name, f_hi, n=400):
            n_hi = int(n * f_hi)
            df = pd.DataFrame({
                'FL1-A': np.r_[rng.normal(0.8, 0.03, n_hi),
                               rng.normal(0.3, 0.03, n - n_hi)],
                'FL2-A': np.r_[rng.normal(0.3, 0.03, n_hi),
                               rng.normal(0.8, 0.03, n - n_hi)]})
            s = FlowSample.from_dataframe(df, name=name)
            s.fluor_channels = ['FL1-A', 'FL2-A']
            s.channel_labels = {'FL1-A': 'CD901b', 'FL2-A': 'CD902b'}
            return s

        exp = FlowExperiment.__new__(FlowExperiment)
        exp.samples = {'S1': sample('S1', 0.7), 'S2': sample('S2', 0.3)}
        labels = exp.cluster_jointly(method='flowsom', n_metaclusters=2,
                                     n_jobs=1)
        assert set(labels) == {'S1', 'S2'}
        assert all(len(v) == 400 for v in labels.values())
        table = align_fluor_labels(list(exp.samples.values()))
        assert table['common'] == ['CD901b', 'CD902b'], table['common']
    """)


@pytest.mark.parametrize('name', ['okabe_ito', 'classic'])
def test_theme_wrappers_match_the_core_functions(name):
    assert theme.group_colors(name) == group_palettes.palette_colors(name)
    for n in range(26):
        assert (theme.categorical_colors(n, name)
                == group_palettes.n_group_colors(n, name))


def test_theme_reexports_the_palette_data():
    assert theme.OKABE_ITO is group_palettes.OKABE_ITO
    assert theme.TAB10 is group_palettes.TAB10
    assert theme.GROUP_PALETTES is group_palettes.GROUP_PALETTES
    assert theme.DEFAULT_GROUP_PALETTE == group_palettes.DEFAULT_GROUP_PALETTE


@pytest.mark.parametrize('pref', ['okabe_ito', 'classic'])
def test_theme_without_a_name_still_follows_the_preference(monkeypatch, pref):
    monkeypatch.setattr(theme, 'read_prefs', lambda: {'group_palette': pref})
    assert theme.group_colors() == group_palettes.palette_colors(pref)
    for n in (0, 3, 9, 25):
        assert (theme.categorical_colors(n)
                == group_palettes.n_group_colors(n, pref))


def test_workspace_run_draws_a_by_source_embedding_without_tk_or_prefs():
    _run_without_tk("""
        import os
        import tempfile
        import types
        import matplotlib
        matplotlib.use('Agg')
        import pandas as pd
        from openflo import workspace_run

        df = pd.DataFrame({'UMAP1': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                           'UMAP2': [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                           '__sample__': ['S1', 'S2', 'S3'] * 2})
        out = tempfile.mkdtemp()
        p = workspace_run._plot_embedding(
            types.SimpleNamespace(data=df), 'UMAP', out, 'u',
            color_by='__sample__', suffix='_by_source')
        assert p and os.path.getsize(p) > 0, p
        for mod in ('openflo.prefs', 'openflo.workspace'):
            assert mod not in sys.modules, mod
    """)


def _by_source_colors(monkeypatch, cfg, pref):
    """Run compute_run with ``cfg`` on three synthetic samples while the
    'group_palette' preference says ``pref``; return the colours of the
    by-source plot's groups, in order."""
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    from matplotlib.axes import Axes

    from openflo import workspace_run
    from openflo.pipeline import FlowSample

    monkeypatch.setattr(theme, 'read_prefs', lambda: {'group_palette': pref})
    monkeypatch.setattr(workspace_run, 'embedding_unavailable',
                        lambda *a, **k: None)

    def fake_umap(self, channels=None, **_k):
        self.data['UMAP1'] = self.data[channels[0]].to_numpy()
        self.data['UMAP2'] = self.data[channels[1]].to_numpy()

    monkeypatch.setattr(FlowSample, 'run_umap', fake_umap)
    drawn = []
    real_scatter = Axes.scatter

    def scatter(self, *a, **k):
        if 'label' in k:
            drawn.append((k['label'], k.get('color')))
        return real_scatter(self, *a, **k)

    monkeypatch.setattr(Axes, 'scatter', scatter)
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.normal(0, 1, (300, 2)), columns=['FL1-A', 'FL2-A'])
    df['__sample__'] = pd.Categorical(['S1', 'S2', 'S3'] * 100)
    prep = {'data': df, 'channels': ['FL1-A', 'FL2-A'], 'label': 'u',
            'n_events': 300, 'color_by': '__sample__'}
    run_cfg = dict({'method': 'flowsom', 'n_metaclusters': 2, 'seed': 42,
                    'max_events': 0, 'umap': True, 'tsne': False,
                    'phate': False, 'trimap': False, 'pacmap': False}, **cfg)
    import tempfile
    res = workspace_run.compute_run(prep, run_cfg, tempfile.mkdtemp())
    assert res['ok'], res
    assert [lab for lab, _ in drawn] == ['S1', 'S2', 'S3'], drawn
    return [c for _, c in drawn]


@pytest.mark.parametrize('palette, other, first', [
    ('classic', 'okabe_ito', ['#1f77b4', '#ff7f0e', '#2ca02c']),
    ('okabe_ito', 'classic', ['#0072B2', '#E69F00', '#009E73']),
])
def test_compute_run_colours_groups_from_the_palette_in_its_cfg(
        monkeypatch, palette, other, first):
    """The palette comes from cfg, not from the preference (which says
    otherwise here)."""
    assert _by_source_colors(
        monkeypatch, {'group_palette': palette}, other) == first


def test_compute_run_without_a_palette_uses_the_default_not_the_preference(
        monkeypatch):
    assert group_palettes.DEFAULT_GROUP_PALETTE == 'okabe_ito'
    assert _by_source_colors(monkeypatch, {}, 'classic') == \
        ['#0072B2', '#E69F00', '#009E73']
