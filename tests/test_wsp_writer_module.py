"""The FlowJo .wsp writer lives in wsp_writer, the gate-coordinate helpers in
wsp_gate_coords, not pipeline.

The writer block of pipeline.py (WspWriter and its export checks) and the gate
coordinates across the .wsp boundary (map_gate_coords and the scale functions
it takes) moved unchanged into the two modules, which import wsp_reader but
not pipeline. These tests pin the move:

* neither new module imports pipeline (or Tk);
* the names pipeline still uses are the SAME objects in pipeline (imported,
  not copied), under their old names;
* every other old name, including the imports pipeline kept only for the
  moved code, still resolves as ``pipeline.X`` to the same object, with a
  DeprecationWarning naming its home;
* no module in the package imports a moved name from pipeline any more.
"""
import ast
import pathlib
import subprocess
import sys
import warnings

import pytest

from openflo import pipeline, wsp_writer

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / 'src' / 'openflo'

# Moved names pipeline still uses: imported there (old name -> (module, new)).
# _polygon_edge_fns went to _MOVED when gate_to_mask moved to gate_eval.
IMPORTED = {'WspWriter': (wsp_writer, 'WspWriter')}
# Old names served by pipeline.__getattr__ (old name -> (module, new name)).
DEPRECATED = {old: home for old, home in pipeline._MOVED.items()
              if home[0] in ('wsp_writer', 'wsp_gate_coords')}
# Imports pipeline kept only for the moved code: served the same way.
NO_LONGER_USED = ('transform_params', '_fcs_text_pairs', '_fcs_timestep',
                  '_positive_float', 'scale_time_coords', '_scale_time_coords',
                  '_xml_safe', '_WSP_BOOLEAN_TAGS', 'unique_population_name',
                  'wsp_population_name', '_BIEX_KEPT_FROM', '_BIEX_OPENED_UPTO',
                  'WSP_FLOWJO_VERSION', 'WSP_OPENFLO_ATTR', '_ellipse_polygon',
                  '_flowjo_biex_lut', '_flowjo_edge_axis', '_strip_comp_marker')


def test_the_new_modules_do_not_import_pipeline_or_tk():
    code = ('import sys, openflo.wsp_writer, openflo.wsp_gate_coords; '
            "print('openflo.pipeline' in sys.modules, 'tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['False', 'False'], out.stdout + out.stderr


def test_pipeline_uses_the_same_objects_not_copies():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for old, (module, new) in IMPORTED.items():
            assert old in vars(pipeline), old
            assert getattr(pipeline, old) is getattr(module, new), old
    assert pipeline.WspWriter.__module__ == 'openflo.wsp_writer'


def test_the_moved_set_is_the_whole_section():
    moved = {new for _home, new in DEPRECATED.values()} | {
        new for _module, new in IMPORTED.values()}
    assert {'WspWriter', '_q', '_wsp_dataset_uri', '_edges_flowjo_opens',
            '_time_upper_edge', '_now_for_wsp', 'map_gate_coords',
            'gate_scale_fns', 'gate_rescale_fns', 'rank_matched_fn',
            'mark_unknown_scale', 'stat_axes_from_json', 'edge_axis_fns',
            'polygon_edge_fns', 'POLYGON_EDGE_STEPS',
            'wsp_uncompensated_channels'} <= moved
    for old in (*DEPRECATED, *NO_LONGER_USED):
        assert old not in vars(pipeline), old


@pytest.mark.parametrize('old', sorted(DEPRECATED) + sorted(NO_LONGER_USED))
def test_the_old_pipeline_name_still_works_with_a_warning(old):
    home, new = pipeline._MOVED[old]
    module = sys.modules.get(f'openflo.{home}') or __import__(
        f'openflo.{home}', fromlist=[new])
    with pytest.warns(DeprecationWarning, match=f'use openflo.{home}.{new}'):
        obj = getattr(pipeline, old)
    assert obj is getattr(module, new)


def test_no_module_imports_a_moved_name_from_pipeline():
    moved = set(DEPRECATED) | set(NO_LONGER_USED) | {'WspWriter', 'wsp_stat_axes'}
    hits = []
    for path in sorted(PACKAGE.rglob('*.py')):
        if path.name == '__init__.py':      # the lazy public API stays on pipeline
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                    (node.level and node.module == 'pipeline')
                    or node.module == 'openflo.pipeline'):
                hits.extend(f'{path.name}:{node.lineno} {a.name}'
                            for a in node.names if a.name in moved)
    assert hits == []
