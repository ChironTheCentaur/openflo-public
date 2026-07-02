"""Install health check ("doctor") — verify an OpenFlo install is intact.

When something behaves oddly and you suspect a corrupted / drifted install,
this reports whether anything is operating outside of norms, in four layers:

  1. Python + OpenFlo version and the interpreter being used.
  2. **Dependency integrity** — every distribution OpenFlo pins is importable
     and at the pinned version. A partial upgrade, a clobbered dependency, or a
     wrong-environment launch shows up here.
  3. **Optional engines** — which analysis backends (UMAP, PhenoGraph, Leiden,
     PHATE, …) are present, via the capabilities probe.
  4. **Behavioural self-test** — seeded synthetic data run through the core
     feature paths and compared to the committed golden baseline
     (:mod:`openflo.selftest`). A green run means the code still reproduces
     reference numbers on this machine.

Run it standalone — works even when the GUI won't start:

    python -m openflo.diagnostics           # human-readable report, exit 0/1
    python -m openflo.diagnostics --json     # machine-readable
    python -m openflo.diagnostics --quick    # skip the (slower) self-test
    openflo-doctor                           # same, via the console script

or double-click ``scripts/diagnose.bat`` (Windows) / ``scripts/diagnose.sh``.

The GUI's **Help ▸ Run diagnostics…** launches this in a SEPARATE process, so a
genuinely broken install (or a native-library crash) is reported rather than
taking the editor down with it.
"""
from __future__ import annotations

import argparse
import importlib.metadata as _md
import json
import platform
import re
import sys


def _parse_requirement(req: str):
    """``'numpy==2.4.6'`` -> ``('numpy', '2.4.6', is_extra)``. ``pinned`` is the
    ``==`` version or None for any other specifier; ``is_extra`` flags a
    requirement gated behind a pip extra (``; extra == 'embed'``)."""
    main, _, marker = req.partition(';')
    is_extra = 'extra' in marker
    m = re.match(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==)?\s*([^\s,;]+)?', main)
    if not m:
        return main.strip(), None, is_extra
    pinned = m.group(3) if m.group(2) == '==' else None
    return m.group(1), pinned, is_extra


def check_dependencies() -> list[dict]:
    """For each CORE distribution OpenFlo pins, compare installed vs pinned.
    Returns ``[{name, required, installed, status, detail}]`` where ``status``
    is ``'ok'`` | ``'drift'`` | ``'missing'``. Extras are left to the engine
    probe (they're optional by design)."""
    try:
        reqs = _md.requires('openflo') or []
    except _md.PackageNotFoundError:
        reqs = []
    rows = []
    for req in reqs:
        name, pinned, is_extra = _parse_requirement(req)
        if is_extra:
            continue
        try:
            installed = _md.version(name)
        except _md.PackageNotFoundError:
            installed = None
        if installed is None:
            status, detail = 'missing', 'NOT INSTALLED'
        elif pinned and installed != pinned:
            status, detail = 'drift', f'pinned {pinned}'
        else:
            status, detail = 'ok', 'ok'
        rows.append({'name': name, 'required': pinned or '(any)',
                     'installed': installed or '—', 'status': status,
                     'detail': detail})
    return rows


def check_selftest() -> dict:
    """Run the seeded behavioural self-test. Wrapped so a hard failure (e.g. a
    broken native backend) is reported, not raised."""
    try:
        from .selftest import run_selftest
        results, ok = run_selftest()
        return {'available': True, 'ok': ok, 'results': results}
    except Exception as exc:                       # noqa: BLE001
        return {'available': False, 'ok': False, 'error': str(exc),
                'results': []}


def run_diagnostics(include_selftest: bool = True) -> dict:
    """Assemble the full health report. ``ok`` is True iff no core dependency is
    missing and (when run) the self-test passes; version drift is surfaced as a
    warning but doesn't on its own flip the install to unhealthy."""
    from .capabilities import openflo_version, probe_capabilities
    deps = check_dependencies()
    engines = probe_capabilities()
    selftest = check_selftest() if include_selftest else None

    missing = [d for d in deps if d['status'] == 'missing']
    drift = [d for d in deps if d['status'] == 'drift']
    ok = not missing
    if selftest is not None:
        ok = ok and selftest['ok']
    return {
        'python': platform.python_version(),
        'executable': sys.executable,
        'platform': platform.platform(),
        'openflo': openflo_version(),
        'dependencies': deps,
        'engines': engines,
        'selftest': selftest,
        'warnings': len(drift),
        'ok': bool(ok),
    }


def format_report(report: dict) -> str:
    """Human-readable diagnostic report. Plain ASCII marks so it renders in any
    terminal and in the GUI's results pane."""
    L = []
    L.append("OpenFlo diagnostics - install health check")
    L.append("=" * 52)
    L.append(f"  OpenFlo      {report['openflo']}")
    L.append(f"  Python       {report['python']}")
    L.append(f"  Platform     {report['platform']}")
    L.append(f"  Interpreter  {report['executable']}")
    L.append("")

    L.append("Core dependencies (pinned versions):")
    for d in report['dependencies']:
        mark = {'ok': '[ OK ]', 'drift': '[WARN]', 'missing': '[FAIL]'}[d['status']]
        note = '' if d['status'] == 'ok' else f"   <-- {d['detail']}"
        L.append(f"  {mark} {d['name']:<16} {d['installed']:<12}{note}")
    L.append("")

    L.append("Optional analysis engines:")
    for e in report['engines']:
        mark = '[ OK ]' if e['available'] else '[ -- ]'
        ver = e['version'] if e['available'] else 'not installed'
        L.append(f"  {mark} {e['label']:<14} {ver:<12} {e['powers']}")
    L.append("")

    st = report['selftest']
    if st is None:
        L.append("Behavioural self-test: skipped (--quick)")
    elif not st['available']:
        L.append(f"Behavioural self-test: [FAIL] could not run — {st.get('error', '')}")
    else:
        n_ok = sum(r['passed'] for r in st['results'])
        L.append(f"Behavioural self-test ({n_ok}/{len(st['results'])} checks vs golden baseline):")
        for r in st['results']:
            mark = '[ OK ]' if r['passed'] else '[FAIL]'
            unit = r['unit'] if r['unit'] == '%' else ''
            got = '—' if r['got'] is None else f"{r['got']:g}{unit}"
            L.append(f"  {mark} {r['label']:<44} {got:>9}  "
                     f"(exp {r['expected']:g}{unit} +/-{r['tol']:g})")
    L.append("")

    L.append("-" * 52)
    if report['ok']:
        msg = "HEALTHY - install reproduces reference behaviour."
        if report['warnings']:
            msg += f" ({report['warnings']} version warning(s) above.)"
        L.append("  " + msg)
    else:
        L.append("  ISSUES FOUND — see [FAIL] rows above. Try reinstalling:")
        L.append("      pip install --upgrade --force-reinstall openflo")
    return '\n'.join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog='openflo-doctor', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', action='store_true',
                    help='Emit the full report as JSON and exit.')
    ap.add_argument('--quick', action='store_true',
                    help='Skip the behavioural self-test (faster; checks '
                         'versions + engines only).')
    args = ap.parse_args(argv)

    # The report carries a few non-ASCII glyphs (engine/metric labels). A
    # legacy console (cp1252) would otherwise crash on encode — degrade to
    # replacement chars instead of failing the very tool meant to diagnose.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[union-attr]
        except Exception:
            pass

    report = run_diagnostics(include_selftest=not args.quick)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
