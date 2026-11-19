"""Which single-stain control file goes with which channel: ``ControlMatcher``.

Core module, no Tk. The compensation editor's Auto-fill uses it, and so does
a Pipeline Workspace run that builds a matrix from a row's or group's comp
beads (scope audit L3), so both pair files and channels the same way.
"""
from __future__ import annotations

import os
import re


class ControlMatcher:
    """Pairs control FCS file names with channel names (classmethods only)."""

    # Most labs name a control after the dye, not the brand.
    _VENDOR_PREFIXES = ('Horizon ', 'BD ', 'eFluor ', 'Brilliant ',
                        'Alexa Fluor ', 'AF', 'Super Bright ')

    # A word ends at a separator, between letters and digits, and where
    # lower case turns upper: 'CD3FLA' -> CD 3 FLA, 'compFLB' -> comp FLB,
    # 'PerCP-Cy5.5' -> Per CP Cy 5 5. An all-capitals run stays whole.
    _WORD = re.compile(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+')

    @classmethod
    def _name_tokens(cls, name):
        """Lower-case words of `name` (see `_WORD`): 'FLB-Cy7 Stained' ->
        ['flb', 'cy', '7', 'stained']. Separators never matter, so 'FLB-Cy7',
        'FLB_Cy7', 'FLB Cy7' and 'FLBCy7' compare equal."""
        return [w.lower() for w in cls._WORD.findall(name)]

    @staticmethod
    def _runs(tokens):
        """Every run of consecutive whole `tokens`, joined."""
        return {''.join(tokens[i:j]) for i in range(len(tokens))
                for j in range(i + 1, len(tokens) + 1)}

    @classmethod
    def _channel_keys(cls, channel):
        """(dye, keys, whole) for `channel`. `dye` is the channel name without
        its -A/-H/-W suffix and any bracketed detector or laser ('FLB (YG)-A'),
        joined ('FLB-Cy7-A' -> 'flbcy7'): channels with the same dye are one
        fluor and share a control. A key is a run of the dye's words
        ('flbcy7', 'flbcy', 'flb', 'cy7', 'cy'); a file matches it only as a run
        of WHOLE words, so 'flb' never matches inside 'flbcy7' or 'xflb'.
        The vendor prefix is not a key on its own ('Horizon' would claim every
        Horizon dye), and neither is a single character or a digit run inside
        a longer dye name (the '5' of PerCP-Cy5-5). `whole` holds the keys
        that name the entire dye, with and without its vendor prefix."""
        s = re.sub(r'\s*\([^)]*\)', '', channel).strip()
        s = re.sub(r'-[AHW]$', '', s, flags=re.IGNORECASE)
        dye = ''.join(cls._name_tokens(s))
        for prefix in cls._VENDOR_PREFIXES:
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix):]
                break
        toks = cls._name_tokens(s)
        keys = {dye} if dye else set()
        for i in range(len(toks)):
            for j in range(i + 1, len(toks) + 1):
                k = ''.join(toks[i:j])
                if j - i == len(toks) or (len(k) >= 2 and not k.isdigit()):
                    keys.add(k)
        return dye, keys, {dye, ''.join(toks)} - {''}

    # A detector name: letters, then a wavelength or a number ('B530',
    # 'YG582', 'UV379', 'BL1'), optionally a band width ('B530/30').
    _DETECTOR = re.compile(r'[A-Za-z]{1,3}\d+(?:[-/]\d+)?')

    @classmethod
    def _bracket_dye_keys(cls, channel):
        """`_channel_keys` of the dye a channel names in brackets after a
        detector ('B530 (FLA)-A' -> FLA, 'YG582 (FLB)-A' -> FLB), or None
        for a channel not written that way. _channel_keys strips the bracket,
        which is right for a laser ('FLB (YG)-A') or a marker ('FLA (CD3)-A')
        but left these with only the detector."""
        m = re.fullmatch(r'\s*(.*?)\s*\(([^)]*)\)\s*(-[AHW])?\s*', channel,
                         re.IGNORECASE)
        if (not m or not m.group(2).strip()
                or not cls._DETECTOR.fullmatch(m.group(1))):
            return None
        return cls._channel_keys(m.group(2).strip() + (m.group(3) or ''))

    @classmethod
    def _match_controls(cls, channels, files):
        """(results, ambiguous, missed) for `_autofill_from_dir`: results maps
        a channel to its file name; ambiguous and missed are the notes for
        the channels left empty."""
        words = {f: cls._name_tokens(os.path.splitext(f)[0]) for f in files}
        runs = {f: cls._runs(words[f]) for f in files}
        info = cls._control_keys(channels, files, runs)
        # The longest key any channel finds in each file.
        longest = {f: max((len(k) for _d, ks, _w in info.values() for k in ks
                           if k in runs[f]), default=0) for f in files}
        claims = cls._control_claims(info, files, runs, longest)
        cands, contested = cls._split_claims(channels, claims, info, runs,
                                             longest, words)
        return cls._pick_controls(channels, cands, contested)

    @classmethod
    def _control_keys(cls, channels, files, runs):
        """`_channel_keys` of each channel, with a detector outside the
        brackets that names no file here and a dye inside them
        ('B530 (FLA)-A') taking that dye's keys. A dye outside
        ('FLB (YG)-A', 'FLA (CD3)-A') is no detector name and keeps its
        keys; one that looks like one ('FLD (CD3)-A') keeps them whenever a
        file here names it."""
        info = {ch: cls._channel_keys(ch) for ch in channels}
        for ch in channels:
            alt = cls._bracket_dye_keys(ch)
            if alt and not any(k in runs[f] for k in info[ch][1]
                               for f in files):
                info[ch] = alt
        return info

    @staticmethod
    def _control_claims(info, files, runs, longest):
        """{file: {dye: [channels]}}: each channel claims the files that
        spell its longest key found here, when no channel finds a longer
        key in that file."""
        claims = {}
        for ch, (dye, ks, _whole) in info.items():
            top = max((len(k) for k in ks if any(k in runs[f] for f in files)),
                      default=0)
            if not top:
                continue
            for f in files:
                if longest[f] == top and any(
                        len(k) == top and k in runs[f] for k in ks):
                    claims.setdefault(f, {}).setdefault(dye, []).append(ch)
        return claims

    @staticmethod
    def _split_claims(channels, claims, info, runs, longest, words):
        """(cands, contested): cands maps a channel to [(leftover, file)] for
        the files only its dye claims; contested to [(file, rivals)] for
        files several dyes claim equally."""
        cands = {ch: [] for ch in channels}
        contested = {ch: [] for ch in channels}
        for f, by_dye in claims.items():
            if len(by_dye) > 1:
                # The dye this file names entirely, if exactly one: 'flc' is
                # all of FLC-A but only part of FLC-Cy7-A.
                named = [d for d, chs in by_dye.items()
                         if any(len(w) == longest[f] and w in runs[f]
                                for w in info[chs[0]][2])]
                if len(named) == 1:
                    by_dye = {named[0]: by_dye[named[0]]}
            chans = [ch for chs in by_dye.values() for ch in chs]
            for ch in chans:
                if len(by_dye) == 1:
                    cands[ch].append((len(''.join(words[f])) - longest[f], f))
                else:
                    contested[ch].append((f, chans))
        return cands, contested

    @staticmethod
    def _pick_controls(channels, cands, contested):
        """(results, ambiguous, missed): a channel takes the candidate with
        the least of its name left over; a tie, or a file only contested,
        leaves it empty with a note."""
        results, ambiguous, missed = {}, [], []
        for ch in channels:
            got = sorted(cands[ch])
            if got and (len(got) == 1 or got[0][0] < got[1][0]):
                results[ch] = got[0][1]
            elif got:
                ambiguous.append(f"{ch} matched {len(got)} files")
            elif contested[ch]:
                f, rivals = contested[ch][0]
                ambiguous.append(f"{f} fits {' and '.join(rivals)} equally")
            else:
                missed.append(ch)
        return results, ambiguous, missed

    @staticmethod
    def _autofill_message(n_matched, n_channels, ambiguous, missed):
        """The status line after an autofill."""
        msg_parts = [f"Matched {n_matched}/{n_channels} channels"]
        if ambiguous:
            msg_parts.append("ambiguous, left empty: "
                             + '; '.join(dict.fromkeys(ambiguous[:3])))
        if missed:
            msg_parts.append(f"unmatched: {', '.join(missed)}")
        return '  •  '.join(msg_parts)
