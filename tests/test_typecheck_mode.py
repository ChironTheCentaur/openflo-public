"""pyright runs in "standard" mode or stricter (best-practices survey R10).

"basic" misses names that are only possibly bound, subclass overrides that
break the base class's signature or attribute type, and new attributes set on
function objects. src/ is clean under "standard"; this stops the setting from
quietly going back.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'


def test_pyright_mode_is_standard_or_strict():
    cfg = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pyright']
    assert cfg.get('typeCheckingMode') in ('standard', 'strict'), cfg.get('typeCheckingMode')


def test_rules_standard_turns_on_are_not_switched_off():
    # Turning one rule off by name would undo the mode for that rule.
    cfg = tomllib.loads(PYPROJECT.read_text(encoding='utf-8'))['tool']['pyright']
    rules = ('reportPossiblyUnboundVariable', 'reportIncompatibleMethodOverride',
             'reportIncompatibleVariableOverride', 'reportFunctionMemberAccess')
    off = {r: cfg[r] for r in rules if cfg.get(r) in ('none', 'information', False)}
    assert off == {}, off
