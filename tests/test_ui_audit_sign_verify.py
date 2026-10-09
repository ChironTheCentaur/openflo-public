"""History window (ui_audit.AuditWindow, 8 % covered): export, sign, verify.

Driven through the real editor (`_show_audit_window`, as the menu does) with
file dialogs / message boxes / simpledialog replaced. Known answer: the
signed record must carry the SHA-256 of each loaded sample's file (computed
here with hashlib), verify as VALID, and turn INVALID naming exactly the file
that is changed after signing.
"""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _editor_or_skip():
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"Tk cannot initialise: {e}")
    root.withdraw()
    import importlib
    gui = importlib.import_module('openflo.gui')
    gui.messagebox.askyesno = lambda *a, **k: True
    ed = gui.ViewGateEditorWindow(root, fcs_dir=None, labels_str='',
                                  on_save=None, primary=False)
    ed.withdraw()
    return root, ed


@pytest.fixture
def env(tmp_path, monkeypatch):
    from openflo import ui_audit
    root, ed = _editor_or_skip()
    files = {}
    for i, nm in enumerate(('s1', 's2')):
        p = tmp_path / f'{nm}.fcs'
        p.write_bytes(bytes(range(256)) * (i + 3))   # any bytes: only hashed
        files[nm] = str(p)
        df = pd.DataFrame({'CD3': np.linspace(0, 1, 50)})
        ed._samples[nm] = SimpleNamespace(name=nm, path=str(p), data=df,
                                          fluor_channels=['CD3'],
                                          channel_labels={'CD3': 'CD3'})
        ed._sample_order.append(nm)
    shown = []
    for fn in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(ui_audit.messagebox, fn,
                            lambda title, msg, _f=fn, **k: shown.append((_f, msg)))
    answers = {'signer': 'Dr Test', 'meaning': 'Reviewed and approved'}
    import tkinter.simpledialog as sd
    monkeypatch.setattr(sd, 'askstring',
                        lambda title, prompt, **k: answers['signer']
                        if 'Signer' in prompt else answers['meaning'])
    target = {}
    monkeypatch.setattr(ui_audit.filedialog, 'asksaveasfilename',
                        lambda **k: target['save'])
    monkeypatch.setattr(ui_audit.filedialog, 'askopenfilename',
                        lambda **k: target['open'])
    ed._audit('load', files=2)
    ed._audit('gate.add', name='CD3+')
    yield SimpleNamespace(root=root, ed=ed, files=files, shown=shown,
                          target=target, tmp=tmp_path, answers=answers)
    root.destroy()


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def test_history_window_lists_and_live_refreshes(env):
    env.ed._show_audit_window()
    win = env.ed._audit_window
    assert len(win._tv.get_children()) == 2
    env.ed._audit('gate.delete', name='CD3+')          # recorded while open
    rows = [win._tv.item(i, 'values') for i in win._tv.get_children()]
    assert [r[2] for r in rows] == ['load', 'gate.add', 'gate.delete']


def test_export_json_csv_md(env):
    env.ed._show_audit_window()
    win = env.ed._audit_window
    for fmt in ('json', 'csv', 'md'):
        env.target['save'] = str(env.tmp / f'trail.{fmt}')
        win._export(fmt)
        assert os.path.isfile(env.target['save'])
    doc = json.load(open(env.tmp / 'trail.json', encoding='utf-8'))
    assert doc['format'] == 'openflo-audit'
    assert [e['action'] for e in doc['entries']] == ['load', 'gate.add']
    csv_text = open(env.tmp / 'trail.csv', encoding='utf-8').read()
    assert 'gate.add' in csv_text and 'CD3+' in csv_text
    assert 'gate.add' in open(env.tmp / 'trail.md', encoding='utf-8').read()


def test_sign_then_verify_then_detect_tamper(env):
    env.ed._show_audit_window()
    win = env.ed._audit_window
    rec_path = str(env.tmp / 'record.json')
    env.target['save'] = rec_path
    win._sign_record()
    rec = json.load(open(rec_path, encoding='utf-8'))
    assert {n: f['sha256'] for n, f in rec['files'].items()} == \
        {n: _sha(p) for n, p in env.files.items()}
    assert rec['signatures'][0]['signer'] == 'Dr Test'
    assert os.path.isfile(str(env.tmp / 'record.md'))
    # the signing itself is recorded in the trail (after the manifest)
    assert env.ed._audit_log.entries()[-1]['action'] == 'compliance.sign'
    assert env.shown[-1][0] == 'showinfo' and '2 files hashed' in env.shown[-1][1]

    env.target['open'] = rec_path
    win._verify_record()
    kind, msg = env.shown[-1]
    assert kind == 'showinfo' and msg.startswith('Overall: VALID')

    with open(env.files['s2'], 'ab') as fh:              # tamper with s2
        fh.write(b'\x00')
    win._verify_record()
    kind, msg = env.shown[-1]
    assert kind == 'showwarning' and 'INVALID / TAMPERED' in msg
    assert msg.rstrip().endswith('Changed/missing data files: s2')


def test_sign_cancelled_at_signer_prompt_writes_nothing(env):
    env.ed._show_audit_window()
    env.answers['signer'] = None
    env.target['save'] = str(env.tmp / 'never.json')
    env.ed._audit_window._sign_record()
    assert not os.path.exists(env.tmp / 'never.json')
