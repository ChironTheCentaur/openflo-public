"""Public names that nothing in OpenFlo uses any more are deprecated before
they are removed: each still works and returns what it did, with a
DeprecationWarning."""

import pytest

# ── functions ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('own, target, expected', [
    (('sample_a', 'g1', 'g2'), ('sample_a', 'g1', 'g3'), '../g3'),
    (('sample_a', 'g1'), ('sample_a', 'g1', 'g2'), '../g1/g2'),
    (('sample_a', 'g1', 'g2'), ('sample_a',), '../..'),
    (('g1',), ('g2', 'g3'), '../g2/g3'),
])
def test_wsp_dependent_name(own, target, expected):
    from openflo import gate_text
    with pytest.warns(DeprecationWarning, match='wsp_dependent_name'):
        assert gate_text.wsp_dependent_name(own, target) == expected


@pytest.mark.parametrize('body, expected', [
    ('<gating:Gating-ML xmlns:gating='
     '"http://www.isac-net.org/std/Gating-ML/v2.0/gating"/>', True),
    ('<Gating-ML/>', True),
    ('<root/>', False),
    ('<<not xml', False),
    ('', False),
])
def test_is_gatingml(tmp_path, body, expected):
    from openflo import gatingml
    path = tmp_path / 'gates.xml'
    path.write_text(body, encoding='utf-8')
    with pytest.warns(DeprecationWarning, match='is_gatingml'):
        assert gatingml.is_gatingml(str(path)) is expected


def test_is_gatingml_missing_file(tmp_path):
    from openflo import gatingml
    with pytest.warns(DeprecationWarning, match='is_gatingml'):
        assert gatingml.is_gatingml(str(tmp_path / 'missing.xml')) is False


@pytest.mark.parametrize('scale, expected', [
    ('logicle', 'Logicle (W 0.5)'),
    ('logicle:w=1.2', 'Logicle (W 1.2)'),
    ('asinh', 'Arcsinh (cofactor 150)'),
    ('asinh:c=5', 'Arcsinh (cofactor 5)'),
    ('log', 'Log'),
    ('linear', 'Linear'),
])
def test_scale_label(scale, expected):
    from openflo import scales
    with pytest.warns(DeprecationWarning, match='scale_label'):
        assert scales.scale_label(scale) == expected


# ── constants (served by a module __getattr__) ──────────────────────────────

def test_scatter_keywords():
    pipeline = pytest.importorskip('openflo.pipeline')
    with pytest.warns(DeprecationWarning, match='SCATTER_KEYWORDS'):
        assert pipeline.SCATTER_KEYWORDS == ['FSC', 'SSC', 'Time', 'time',
                                             'Width', 'width']


def test_fixes_kinds_and_modes():
    from openflo import fixes
    with pytest.warns(DeprecationWarning, match='KINDS'):
        assert fixes.KINDS == ('error', 'default-change')
    with pytest.warns(DeprecationWarning, match='MODES'):
        assert fixes.MODES == ('recompute', 'shift')
    with pytest.raises(AttributeError):
        fixes.NOT_A_NAME  # noqa: B018


def test_plot_dark_themes():
    pytest.importorskip('tkinter')
    from openflo import theme
    with pytest.warns(DeprecationWarning, match='PLOT_DARK_THEMES'):
        assert theme.PLOT_DARK_THEMES == {'midnight', 'light-dark'}
    with pytest.raises(AttributeError):
        theme.NOT_A_NAME  # noqa: B018


def test_beads_label():
    pytest.importorskip('tkinter')
    from openflo import ui_mass_cytometry
    with pytest.warns(DeprecationWarning, match='BEADS_LABEL'):
        assert ui_mass_cytometry.BEADS_LABEL == 'Bead events'
    with pytest.raises(AttributeError):
        ui_mass_cytometry.NOT_A_NAME  # noqa: B018


def test_from_import_still_works():
    from openflo import fixes
    with pytest.warns(DeprecationWarning, match='KINDS'):
        from openflo.fixes import KINDS
    assert KINDS == ('error', 'default-change')
    # Fix is a live name, not deprecated: reading it warns nothing, and it
    # is the class of every registered fix.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fix_cls = fixes.Fix
    assert fixes.REGISTRY
    assert all(type(f) is fix_cls for f in fixes.REGISTRY)
