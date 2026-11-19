"""scripts/function_size.py: complexity is counted the way radon counts it,
length is def line to last line, and the list of functions over a limit may
only shrink.

Every snippet here is synthetic.
"""
import importlib.util
import json
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    'function_size', ROOT / 'scripts' / 'function_size.py')
assert _spec and _spec.loader
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)


def sizes(source):
    """{name: (cc, lines)} for every function in `source`."""
    return {s.name: (s.cc, s.lines)
            for s in fs.measure_source('src/openflo/x.py',
                                       textwrap.dedent(source))}


def cc(body):
    """Complexity of a function whose body is `body` (dedented)."""
    src = 'def f(a, b, c):\n' + textwrap.indent(textwrap.dedent(body), '    ')
    return sizes(src)['f'][0]


# Each construct with the value radon 6 gives it (checked by hand against
# `radon cc -s` when this was written).
@pytest.mark.parametrize('body, want', [
    ('return a\n', 1),
    ('if a:\n    pass\n', 2),
    ('if a:\n    pass\nelif b:\n    pass\nelse:\n    pass\n', 3),
    ('x = a if b else c\n', 2),
    ('for x in a:\n    pass\n', 2),
    ('for x in a:\n    pass\nelse:\n    pass\n', 3),
    ('while a:\n    pass\n', 2),
    ('while a:\n    pass\nelse:\n    pass\n', 3),
    ('try:\n    pass\nexcept ValueError:\n    pass\nexcept KeyError:\n    pass\n', 3),
    ('try:\n    pass\nexcept ValueError:\n    pass\nelse:\n    pass\n', 3),
    ('try:\n    pass\nfinally:\n    pass\n', 1),
    ('return a and b and c\n', 3),
    ('return a or b\n', 2),
    ('return [x for x in a if x if b]\n', 4),
    ('return {x: y for x in a for y in b}\n', 3),
    ('assert a\n', 2),
    ('assert a and b, c\n', 2),           # nothing inside an assert counts
    ('with a:\n    pass\n', 1),
    ('g = lambda: a if b else c\n', 2),   # a lambda counts toward its function
    ('match a:\n    case 1:\n        pass\n    case 2:\n        pass\n', 3),
    ('match a:\n    case 1:\n        pass\n    case _:\n        pass\n', 2),
    ('def inner():\n    if a:\n        pass\nreturn inner\n', 1),
    ('class K:\n    def m(self):\n        if a:\n            pass\n', 1),
])
def test_complexity_counts_like_radon(body, want):
    assert cc(body) == want


def test_nested_functions_and_methods_are_scored_on_their_own():
    found = sizes('''
        class K:
            def m(self):
                if self:
                    pass
                def inner():
                    for x in self:
                        if x:
                            pass
                return inner

        def top():
            if True:
                def under_if():
                    return 1
    ''')
    assert found == {'K.m': (2, 8), 'K.m.inner': (3, 4), 'top': (2, 4),
                     'top.under_if': (1, 2)}


def test_length_is_def_line_to_last_line_without_decorators():
    found = sizes('''
        @property
        @other
        def f():
            x = 1

            return x
    ''')
    assert found['f'][1] == 4


def _tree(tmp_path, source, baseline=None):
    pkg = tmp_path / 'src' / 'openflo'
    pkg.mkdir(parents=True)
    (pkg / 'mod.py').write_text(textwrap.dedent(source), encoding='utf-8')
    (tmp_path / 'scripts').mkdir()
    if baseline is not None:
        (tmp_path / 'scripts' / fs.BASELINE_NAME).write_text(
            json.dumps(baseline), encoding='utf-8')
    return tmp_path


def _branchy(name, n):
    """A function with CC exactly n (n - 1 ifs)."""
    ifs = ''.join(f'    if a == {i}:\n        a += 1\n' for i in range(n - 1))
    return f'def {name}(a):\n{ifs}    return a\n'


def _long(name, n):
    """A function exactly n lines long."""
    return f'def {name}():\n' + '    x = 1\n' * (n - 2) + '    return x\n'


def test_limits_are_cc_30_and_more_than_200_lines():
    found = {s.name: s.over for s in fs.measure_source('src/openflo/x.py',
             _branchy('cc29', 29) + _branchy('cc30', 30)
             + _long('l200', 200) + _long('l201', 201))}
    assert found == {'cc29': False, 'cc30': True, 'l200': False, 'l201': True}


def test_a_new_function_over_a_limit_fails(tmp_path, capsys):
    root = _tree(tmp_path, _branchy('old', 30) + _branchy('new', 31),
                 baseline={'src/openflo/mod.py::old': {'cc': 30, 'lines': 61}})
    assert fs.main(['--root', str(root)]) == 1
    out = capsys.readouterr().out
    assert 'new: CC 31' in out and 'old:' not in out


def test_listed_functions_pass_even_when_they_grow(tmp_path, capsys):
    root = _tree(tmp_path, _branchy('old', 40),
                 baseline={'src/openflo/mod.py::old': {'cc': 30, 'lines': 61}})
    assert fs.main(['--root', str(root)]) == 0
    assert 'none new' in capsys.readouterr().out


def test_a_fixed_function_passes_with_a_note_and_write_baseline_drops_it(
        tmp_path, capsys):
    root = _tree(tmp_path, _branchy('old', 29) + _branchy('still', 30),
                 baseline={'src/openflo/mod.py::old': {'cc': 30, 'lines': 61},
                           'src/openflo/mod.py::still': {'cc': 30, 'lines': 61}})
    assert fs.main(['--root', str(root)]) == 0
    assert 'mod.py::old: now under both limits' in capsys.readouterr().out
    assert fs.main(['--root', str(root), '--write-baseline']) == 0
    written = json.loads((root / 'scripts' / fs.BASELINE_NAME).read_text())
    assert list(written) == ['src/openflo/mod.py::still']


def test_write_baseline_will_not_add_a_function_without_allow_new(tmp_path):
    root = _tree(tmp_path, _branchy('new', 30), baseline={})
    path = root / 'scripts' / fs.BASELINE_NAME
    assert fs.main(['--root', str(root), '--write-baseline']) == 1
    assert json.loads(path.read_text()) == {}
    assert fs.main(['--root', str(root), '--write-baseline', '--allow-new']) == 0
    assert list(json.loads(path.read_text())) == ['src/openflo/mod.py::new']


def test_the_repository_passes_and_its_list_is_current():
    """The real tree: nothing new over a limit, and no listed entry that is
    already fixed (that one would only need --write-baseline)."""
    new, notes = fs.compare(fs.measure_tree(ROOT), json.loads(
        (ROOT / 'scripts' / fs.BASELINE_NAME).read_text(encoding='utf-8')))
    assert [str(s) for s in new] == []
    assert notes == []
