"""The group-test choice shared by the comparison windows.

A "Test" box and a "Paired by donor" tick, read as the keyword arguments
stats.compare_groups takes. The default is a t-test on transformed values:
at the group sizes flow experiments use (3 to 5 samples), a rank test
cannot come out significant (Mann-Whitney's smallest p at 3 vs 3 is 0.1),
and a t-test on the logit (percentages) or arcsinh (intensities) scale is
the usual choice there.
"""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import ttk

from .ui_tips import tip

T_TRANSFORMED = 't-test, transformed'
RANK = 'Rank test'
T_RAW = 't-test, raw values'
TESTS = (T_TRANSFORMED, RANK, T_RAW)


def test_kwargs(choice, transform):
    """``{parametric, transform}`` for compare_groups from the box's
    `choice`; `transform` is the one that suits the metric."""
    if choice == RANK:
        return {'parametric': False, 'transform': None}
    if choice == T_RAW:
        return {'parametric': True, 'transform': None}
    return {'parametric': True, 'transform': transform}


def pair_key(sample, group, tokens=()):
    """The donor a sample belongs to, read from its name: the name with its
    group's label and every grouping token taken out (case-insensitive),
    then separators trimmed. 'D1_Stim' and 'D1_Ctrl' both give 'd1'."""
    key = str(sample).lower()
    for word in [str(group), *tokens]:
        w = word.strip().lower()
        if w:
            key = key.replace(w, ' ')
    key = re.sub(r'\.fcs$', '', key)
    return re.sub(r'[\s_\-.,;:()\[\]]+', ' ', key).strip()


def add_test_controls(editor, parent, on_change=None):
    """Pack the Test box and the Paired tick into `parent`; returns
    ``(test_var, paired_var)``."""
    ttk.Label(parent, text="Test:").pack(side='left')
    test_var = tk.StringVar(master=parent, value=T_TRANSFORMED)
    box = ttk.Combobox(parent, textvariable=test_var, width=17,
                       state='readonly', values=TESTS)
    box.pack(side='left', padx=(2, 4))
    tip(editor, box,
        "t-test, transformed (default): Welch's t-test (ANOVA for more "
        "groups) on percentages as log odds, counts as log and intensities "
        "on the arcsinh scale. It is the usual choice at 3 to 5 samples a "
        "group. Rank test: Mann-Whitney / Kruskal-Wallis, which needs at "
        "least 4 samples a group to ever reach p < 0.05. t-test, raw "
        "values: the t-test on the numbers as they are.")
    paired_var = tk.BooleanVar(master=parent, value=False)
    chk = ttk.Checkbutton(parent, text="Paired by donor",
                          variable=paired_var)
    chk.pack(side='left', padx=(0, 8))
    tip(editor, chk,
        "Compare each donor with itself: samples are matched by their name "
        "with the group taken out (D1_Stim with D1_Ctrl). Two groups get a "
        "paired t-test (Wilcoxon signed-rank with the rank test), more get "
        "a repeated-measures ANOVA (Friedman). Differential abundance takes "
        "the donor out of the model the same way. A sample with no partner "
        "is left out, and the summary names it.")
    if on_change is not None:
        box.bind('<<ComboboxSelected>>', lambda *_: on_change())
        chk.configure(command=on_change)
    return test_var, paired_var


def power_lines(res):
    """Summary lines about pairing and the smallest p the test could give."""
    lines = []
    if res.get('paired'):
        lines.append(f"  Paired by donor: {res.get('n_pairs') or 0} "
                     "donor(s) in every group")
        if res.get('unpaired'):
            lines.append("  Left out, no partner in every group: "
                         + ', '.join(map(str, res['unpaired'])))
    mp = res.get('min_p')
    if res.get('underpowered'):
        lines.append(
            f"  With these group sizes this test can't give p below "
            f"{mp:.3g}, so it can't come out significant at 0.05 however "
            "different the groups are. Use the t-test, add samples, or "
            "pair by donor.")
    return lines
