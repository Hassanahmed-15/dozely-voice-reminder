"""Restricted decoding: find which known tablet name and which time phrase fit the audio best.

Instead of writing out whatever sounds it hears, the decoder asks of each candidate
phrase "how well can this phrase explain some stretch of the audio?" Everything outside
that stretch is absorbed by a filler that accepts any sound. The score is the log
probability lost by forcing the phrase in, so 0 means a perfect fit.

Candidates come from text, so a tablet typed into the app becomes recognizable
immediately, with no retraining.
"""
from dataclasses import dataclass

import numpy as np

import phones
from generate_commands import DRUGS, HELD_OUT_DRUGS

NEG = -1e9


@dataclass
class Candidate:
    label: object          # what this phrase means, e.g. "Panadol" or (8, 30)
    text: str
    ids: list
    lang: str = ""
    period: str = ""       # set when the phrase itself says am/pm


class Spotter:
    """Scores many candidate phrases against one utterance at once."""

    def __init__(self, candidates: list[Candidate]):
        self.candidates = candidates
        k = len(candidates)
        s_max = 2 * max(len(c.ids) for c in candidates) + 1
        self.idx = np.zeros((k, s_max), dtype=np.int64)
        self.valid = np.zeros((k, s_max), dtype=bool)
        self.skip = np.zeros((k, s_max), dtype=bool)
        self.end = np.zeros(k, dtype=np.int64)
        self.n_phones = np.array([sum(i != phones.TOKEN_ID[phones.SPACE] for i in c.ids) for c in candidates], dtype=np.float64)
        for j, c in enumerate(candidates):
            lab = [0]
            for i in c.ids:
                lab += [i, 0]
            s = len(lab)
            self.idx[j, :s] = lab
            self.valid[j, :s] = True
            for q in range(2, s):
                self.skip[j, q] = lab[q] != 0 and lab[q] != lab[q - 2]
            self.end[j] = s - 1

    def score(self, lp: np.ndarray) -> np.ndarray:
        """lp: (frames, tokens) log probabilities. Returns one score per candidate (<= 0)."""
        lp = lp.astype(np.float64)
        rows = np.arange(len(self.candidates))
        best = lp.max(1)
        emit = np.where(self.valid[None], lp[:, self.idx], NEG)  # (T, K, S)

        def finished(a):
            return np.maximum(a[rows, self.end], a[rows, self.end - 1])

        alpha = np.full(self.idx.shape, NEG)
        alpha[:, 0], alpha[:, 1] = emit[0, :, 0], emit[0, :, 1]
        pre, post = best[0], np.full(len(rows), NEG)
        pad1 = np.full((len(rows), 1), NEG)
        pad2 = np.full((len(rows), 2), NEG)
        for t in range(1, len(lp)):
            prev = alpha
            a = np.maximum(prev, np.concatenate([pad1, prev[:, :-1]], 1))
            a = np.maximum(a, np.where(self.skip, np.concatenate([pad2, prev[:, :-2]], 1), NEG))
            a[:, 0] = np.maximum(a[:, 0], pre)
            a[:, 1] = np.maximum(a[:, 1], pre)
            post = np.maximum(post, finished(prev)) + best[t]
            alpha = a + emit[t]
            pre = pre + best[t]
        return np.maximum(post, finished(alpha)) - best.sum()

    def rank(self, lp: np.ndarray, bonus: float = 0.0):
        """Candidates sorted best first. `bonus` per phone favours phrases that explain more audio."""
        raw = self.score(lp)
        total = raw + bonus * self.n_phones
        order = np.argsort(-total)
        return [(self.candidates[i], float(total[i]), float(raw[i])) for i in order]


_ID_CACHE: dict = {}


def _ids(text: str, voice: str) -> list:
    key = (text, voice)
    if key not in _ID_CACHE:
        _ID_CACHE[key] = [phones.TOKEN_ID[t] for t in phones.to_tokens(text, voice)]
    return _ID_CACHE[key]


def tablet_candidates(names=None) -> list[Candidate]:
    """names: list of (label, [(text, espeak_voice), ...]). Defaults to the built-in list."""
    if names is None:
        names = [(latin, [(latin.lower(), "en-us"), (urdu, "ur")]) for latin, urdu, _ in DRUGS + HELD_OUT_DRUGS]
    out, seen = [], set()
    for label, spellings in names:
        for text, voice in spellings:
            ids = [i for i in _ids(text, voice) if i != phones.TOKEN_ID[phones.SPACE]]
            if ids and (label, tuple(ids)) not in seen:
                seen.add((label, tuple(ids)))
                out.append(Candidate(label, text, ids))
    return out


UR_1_59 = ("ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس بیس "
           "اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس تیس اکتیس بتیس تینتیس چونتیس پینتیس "
           "چھتیس سینتیس اڑتیس انتالیس چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس سینتالیس "
           "اڑتالیس انچاس پچاس اکیاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ").split()
_EN_ONES = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty"}


def en_number(n: int) -> str:
    if n < 20:
        return _EN_ONES[n - 1]
    tens, ones = divmod(n, 10)
    return _EN_TENS[tens * 10] + ("" if ones == 0 else " " + _EN_ONES[ones - 1])


def en_minute(m: int) -> str:
    return f"oh {_EN_ONES[m - 1]}" if m < 10 else en_number(m)


def time_candidates() -> list[Candidate]:
    """Spoken clock times: Urdu, English, and English numbers inside Urdu ("ten baje").
    Labels are (hour 1-12, minute)."""
    sp = phones.TOKEN_ID[phones.SPACE]
    cands = []

    def add(label, lang, *parts, period=""):
        ids = []
        for text, voice in parts:
            ids += ([sp] if ids else []) + _ids(text, voice)
        cands.append(Candidate(label, " ".join(t for t, _ in parts), ids, lang, period))

    for h in range(1, 13):
        u, e = UR_1_59[h - 1], en_number(h)
        nxt_u, nxt_e = UR_1_59[h % 12], en_number(h % 12 + 1)
        # Urdu
        add((h, 0), "ur", (f"{u} بجے", "ur"))
        add((h, 30), "ur", (f"ساڑھے {u} بجے", "ur"))
        add((h, 15), "ur", (f"سوا {u} بجے", "ur"))
        add((h, 45), "ur", (f"پونے {nxt_u} بجے", "ur"))
        for m in range(1, 60):
            add((h, m), "ur", (f"{u} بج کر {UR_1_59[m - 1]} منٹ", "ur"))
            add((h, m), "ur", (f"{u} بج کر {UR_1_59[m - 1]}", "ur"))
        # English numbers said inside Urdu
        add((h, 0), "ur", (e, "en-us"), ("بجے", "ur"))
        for m in range(1, 60):
            add((h, m), "ur", (f"{e} {en_minute(m)}", "en-us"), ("بجے", "ur"))
        # English. Every phrase carries an anchor ("at", "am/pm", "o'clock"), because a bare
        # number such as "nine" turns up in ordinary speech far too often to mean a time.
        add((h, 0), "en", (e, "en-us"), ("o'clock", "en-us"))
        for m in range(0, 60):
            t = (e, "en-us") if m == 0 else (f"{e} {en_minute(m)}", "en-us")
            add((h, m), "en", ("at", "en-us"), t)
            for ap in ("am", "pm"):
                add((h, m), "en", t, (f"{ap[0]} m", "en-us"), period=ap)
                add((h, m), "en", ("at", "en-us"), t, (f"{ap[0]} m", "en-us"), period=ap)
        add((h, 30), "en", (f"half past {e}", "en-us"))
        add((h, 15), "en", (f"quarter past {e}", "en-us"))
        add((h, 45), "en", (f"quarter to {nxt_e}", "en-us"))
    add((1, 30), "ur", ("ڈیڑھ بجے", "ur"))
    add((2, 30), "ur", ("ڈھائی بجے", "ur"))
    add((12, 0), "noon", ("noon", "en-us"))
    add((12, 0), "midnight", ("midnight", "en-us"))
    return cands


URDU_PERIODS = [("morning", "صبح"), ("afternoon", "دوپہر"), ("evening", "شام"), ("night", "رات")]


def urdu_period_candidates() -> list[Candidate]:
    return [Candidate(label, text, _ids(text, "ur"), "ur") for label, text in URDU_PERIODS]


def decide_period(lp: np.ndarray, time: Candidate, urdu_periods: "Spotter") -> str:
    """For English, judge am/pm together with the chosen time ("nine forty p m"), because the
    words alone are too short to tell apart. For Urdu, spot subah/dopahar/shaam/raat."""
    if time.period:
        return time.period
    if time.lang == "noon":
        return "pm"
    if time.lang == "midnight":
        return "am"
    if time.lang == "en":
        pair = Spotter([Candidate(p, f"{time.text} {p[0]} m", _ids(f"{time.text} {p[0]} m", "en-us"), "en") for p in ("am", "pm")])
        return pair.rank(lp)[0][0].label
    return urdu_periods.rank(lp)[0][0].label


# Chosen on the tuning half of the held-out synthetic commands (see evaluate_commands.py).
TABLET_BONUS = 4.0
TIME_BONUS = 2.0


class Understander:
    """Turns model output into a tablet, a time and a period, each with a fit score.

    The fit is the log probability lost per phone by forcing the phrase in: near 0 is a
    clean fit, and strongly negative means the phrase is probably not in the audio.
    """

    def __init__(self, tablet_names=None):
        self.tablets = Spotter(tablet_candidates(tablet_names))
        self.times = Spotter(time_candidates())
        self.urdu_periods = Spotter(urdu_period_candidates())

    def __call__(self, lp: np.ndarray) -> dict:
        tab, _, tab_raw = self.tablets.rank(lp, TABLET_BONUS)[0]
        tim, _, tim_raw = self.times.rank(lp, TIME_BONUS)[0]
        return {
            "tablet": tab.label, "tablet_fit": tab_raw / max(1, len(tab.ids)),
            "hour": tim.label[0], "minute": tim.label[1], "time_phrase": tim.text,
            "time_fit": tim_raw / max(1, len(tim.ids)),
            "period": decide_period(lp, tim, self.urdu_periods),
        }


def load_thresholds() -> dict:
    """Cut-offs from calibrate.py; below them the phrase is treated as not present."""
    import json
    from pathlib import Path
    path = Path(__file__).with_name("thresholds.json")
    return json.loads(path.read_text()) if path.exists() else {"tablet": -9e9, "time": -9e9}


def accept(u: dict, thresholds: dict) -> dict:
    """Blank out the tablet and/or time when they do not fit well enough to trust."""
    u = dict(u)
    u["tablet_found"] = u["tablet_fit"] >= thresholds["tablet"]
    u["time_found"] = u["time_fit"] >= thresholds["time"]
    return u
