"""Importing an openflo module does nothing else (docs/best-practices.md BP3).

`import openflo.pipeline` used to add a process-wide "ignore FutureWarning"
filter, hiding every FutureWarning from any program that imported the
library, and `import openflo.inspect_fcs` used to exit Python. The
application entry points keep hiding FutureWarnings, from their main().
"""
import ast
import json
import os
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'openflo'


def _run(code):
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_pipeline_leaves_the_warning_filters_alone():
    # A fresh interpreter: pytest's own filterwarnings setting is not there.
    # numpy adds narrow filters of its own on import; openflo adds none that
    # ignores a whole warning category.
    got = _run(
        'import warnings\n'
        'before = list(warnings.filters)\n'
        'import openflo.pipeline\n'
        'print([(f[0], f[2].__name__) for f in warnings.filters\n'
        '       if f not in before and f[0] == "ignore" and f[1] is None])\n')
    assert got == '[]'


def test_a_future_warning_after_importing_pipeline_still_reaches_the_caller():
    got = _run(
        'import warnings\n'
        'import openflo.pipeline\n'
        'with warnings.catch_warnings(record=True) as w:\n'
        '    warnings.warn("next release changes this", FutureWarning)\n'
        'print(len(w))\n')
    assert got == '1'


def test_importing_inspect_fcs_without_arguments_does_not_exit():
    got = _run(
        'import sys\n'
        'sys.argv = ["python"]\n'
        'import openflo.inspect_fcs as m\n'
        'print(callable(m.main))\n')
    assert got == 'True'


def test_hide_future_warnings_ignores_future_warnings_only():
    from openflo._console import hide_future_warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        hide_future_warnings()
        warnings.warn('later', FutureWarning, stacklevel=2)
        warnings.warn('now', UserWarning, stacklevel=2)
    assert [type(x.message) for x in w] == [UserWarning]


def _calls_hide(func):
    return any(isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'hide_future_warnings'
               for n in ast.walk(func))


def test_every_installed_command_hides_future_warnings_in_main():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    targets = {**project['scripts'], **project['gui-scripts']}.values()
    assert targets
    for target in targets:
        module, func = target.split(':')
        tree = ast.parse((SRC / (module.split('.')[-1] + '.py')).read_text())
        mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func]
        assert mains and _calls_hide(mains[0]), target


def test_the_pipeline_job_child_hides_future_warnings():
    tree = ast.parse((SRC / 'workspace_run.py').read_text())
    child = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name == 'subprocess_main']
    assert child and _calls_hide(child[0])


def test_every_process_pool_sets_the_filter_in_its_workers():
    # spawn and forkserver workers start with default warning filters, so the
    # parent's hide_future_warnings() does not reach them.
    pools = []
    for path in sorted(SRC.glob('*.py')):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'ProcessPoolExecutor':
                init = [k.value for k in n.keywords if k.arg == 'initializer']
                args = [k.value for k in n.keywords if k.arg == 'initargs']
                pools.append((path.name, n.lineno,
                              bool(init) and getattr(init[0], 'id', None) == 'start_worker'
                              and bool(args) and 'worker_settings()' in ast.unparse(args[0])))
    assert pools
    assert [p for p in pools if not p[2]] == []
    tree = ast.parse((SRC / '_console.py').read_text())
    start = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'start_worker']
    assert start and _calls_hide(start[0])


def test_every_process_pool_spawns_its_workers():
    # Linux forks by default: a copy of a process whose threads hold a lock
    # can hang (Python 3.12 warns, 3.14 changes the default). Spawn on every
    # system, as Windows and macOS already do, so the workers match there.
    pools = []
    for path in sorted(SRC.glob('*.py')):
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Call) and getattr(n.func, 'id', None) == 'ProcessPoolExecutor':
                ctx = [k.value for k in n.keywords if k.arg == 'mp_context']
                spawn = (bool(ctx) and isinstance(ctx[0], ast.Call)
                         and getattr(ctx[0].func, 'attr', None) == 'get_context'
                         and [ast.literal_eval(a) for a in ctx[0].args] == ['spawn'])
                pools.append((path.name, n.lineno, spawn))
    assert len(pools) >= 3
    assert [p for p in pools if not p[2]] == []


def test_a_spawned_worker_starts_with_the_parents_console_settings(tmp_path):
    # A spawned worker is a fresh interpreter: without start_worker it logs
    # nothing below WARNING, writes the locale's encoding, plots on the
    # default backend and forgets the palette the user picked.
    code = (
        'import json, logging, multiprocessing, sys\n'
        'from concurrent.futures import ProcessPoolExecutor\n'
        'from openflo._console import start_worker, worker_settings\n'
        'def probe():\n'
        '    import logging, sys, matplotlib\n'
        '    from openflo import flow_sample\n'
        '    root = logging.getLogger()\n'
        '    fmt = [h.formatter._fmt for h in root.handlers if h.formatter]\n'
        '    return [logging.getLevelName(root.level), fmt,\n'
        '            flow_sample._DEFAULT_CATEGORICAL_PALETTE,\n'
        '            matplotlib.get_backend().lower(), sys.stdout.encoding.lower()]\n'
        'def run():\n'
        '    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),\n'
        '                             initializer=start_worker,\n'
        '                             initargs=(worker_settings(),)) as ex:\n'
        '        return ex.submit(probe).result()\n'
        'if __name__ == "__main__":\n'
        '    out = [run()]\n'
        '    logging.basicConfig(level=logging.DEBUG, format="SYNTH %(name)s %(message)s")\n'
        '    from openflo.flow_sample import set_default_palette\n'
        '    set_default_palette("Set1")\n'
        '    out.append(run())\n'
        '    sys.stderr.write(json.dumps(out) + "\\n")\n')
    script = tmp_path / 'worker_settings_probe.py'
    script.write_text(code)
    env = {**os.environ, 'PYTHONIOENCODING': 'ascii', 'MPLBACKEND': 'svg'}
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=300, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    plain, configured = json.loads(out.stderr.strip().splitlines()[-1])
    # A parent that set nothing up gives a worker that sets nothing up.
    assert plain == ['WARNING', [], 'auto', 'agg', 'utf-8']
    assert configured == ['DEBUG', ['SYNTH %(name)s %(message)s'], 'Set1',
                          'agg', 'utf-8']


def test_a_format_the_worker_cannot_replay_still_starts_the_worker():
    # An initializer that raises breaks the pool, so every sample fails:
    # a format logging rejects falls back to its default format, same level.
    code = ('import logging\n'
            'from openflo._console import start_worker\n'
            'start_worker({"handled": True, "level": logging.INFO,\n'
            '              "format": "{levelname} {message}", "datefmt": None,\n'
            '              "style": "%"})\n'
            'root = logging.getLogger()\n'
            'print(root.level, len(root.handlers))\n')
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ['20', '1']


def test_a_spawned_worker_with_the_initializer_prints_no_future_warning(tmp_path):
    code = (
        'import multiprocessing, warnings\n'
        'from concurrent.futures import ProcessPoolExecutor\n'
        'from openflo._console import hide_future_warnings\n'
        'def warn():\n'
        '    warnings.warn("next release changes this", FutureWarning, stacklevel=2)\n'
        '    return 1\n'
        'if __name__ == "__main__":\n'
        '    for init in (None, hide_future_warnings):\n'
        '        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn"),\n'
        '                                 initializer=init) as ex:\n'
        '            ex.submit(warn).result()\n'
        '        import sys; sys.stderr.write("===\\n"); sys.stderr.flush()\n')
    script = tmp_path / 'spawn_warning_probe.py'
    script.write_text(code)
    out = subprocess.run([sys.executable, str(script)], capture_output=True,
                         text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    without, with_init = out.stderr.split('===\n')[:2]
    assert 'FutureWarning' in without      # the probe can see a worker warning
    assert 'FutureWarning' not in with_init
