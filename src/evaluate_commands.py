"""Measure tablet and time accuracy on the held-out synthetic commands.

The 1,016 held-out commands (6 tablet names and 1 voice never trained on) are split in
two by clip number: the first half is used to choose settings, and the second half is
scored once with those settings.

    python src/evaluate_commands.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import decoder
import features as fx
import transcribe as ours

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
BONUSES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]


@torch.no_grad()
def log_probs(model, rows):
    mels = np.memmap(FEAT / "mels.bin", dtype=np.float16, mode="r").reshape(-1, fx.N_MELS)
    out = []
    for r in rows:
        m = fx.normalize(np.array(mels[r["offset"]: r["offset"] + r["frames"]], dtype=np.float32))
        lp, _ = model(torch.from_numpy(m)[None], torch.tensor([len(m)]))
        out.append(lp[0].numpy())
    return out


def rankings(spotter, lps):
    raw = [spotter.score(lp) for lp in lps]
    return raw


def pick(spotter, raw, bonus):
    return spotter.candidates[int(np.argmax(raw + bonus * spotter.n_phones))].label


def accuracy(rows, preds, key):
    return sum(p == key(r) for p, r in zip(preds, rows)) / len(rows)


def main():
    rows = [json.loads(l) for l in open(FEAT / "manifest.jsonl", encoding="utf-8")]
    test = [r for r in rows if r["source"] == "command" and r["split"] == "test"]
    test.sort(key=lambda r: r["offset"])
    tune, report = test[0::2], test[1::2]
    model = ours.load_model()
    tab, tim, per = decoder.Spotter(decoder.tablet_candidates()), decoder.Spotter(decoder.time_candidates()), decoder.Spotter(decoder.urdu_period_candidates())
    lexicon = ours._lexicon()

    def run(subset):
        lps = log_probs(model, subset)
        return lps, [tab.score(lp) for lp in lps], [tim.score(lp) for lp in lps], None

    lps, t_raw, h_raw, p_raw = run(tune)
    want_tab = lambda r: r["drug"]
    want_time = lambda r: (r["hour"], r["minute"])
    want_per = lambda r: r["period"]
    best_tab = max(BONUSES, key=lambda b: accuracy(tune, [pick(tab, x, b) for x in t_raw], want_tab))
    best_time = max(BONUSES, key=lambda b: accuracy(tune, [pick(tim, x, b) for x in h_raw], want_time))
    print(f"settings chosen on the tuning half: tablet bonus {best_tab}, time bonus {best_time}")

    lps, t_raw, h_raw, p_raw = run(report)
    p_tab = [pick(tab, x, best_tab) for x in t_raw]
    time_cands = [tim.candidates[int(np.argmax(x + best_time * tim.n_phones))] for x in h_raw]
    p_time = [c.label for c in time_cands]
    p_per = [decoder.decide_period(lp, c, per) for lp, c in zip(lps, time_cands)]
    base_tab = [ours.best_tablet(ours.recognize_lp(lp), lexicon, threshold=9.0)[0] for lp in lps]

    def line(name, subset_idx):
        sub = [report[i] for i in subset_idx]
        acc = lambda preds, key: sum(preds[i] == key(report[i]) for i in subset_idx) / len(subset_idx)
        both = sum(p_tab[i] == want_tab(report[i]) and p_time[i] == want_time(report[i]) and p_per[i] == want_per(report[i]) for i in subset_idx) / len(subset_idx)
        print(f"{name:34s} n={len(sub):4d}  tablet {acc(p_tab, want_tab):6.1%} (before: {acc(base_tab, want_tab):6.1%})"
              f"  time {acc(p_time, want_time):6.1%}  period {acc(p_per, want_per):6.1%}  all three {both:6.1%}")

    print("\nscored once on the reporting half:")
    idx = range(len(report))
    line("all held-out commands", list(idx))
    for voice in sorted({r["say_voice"] for r in report}):
        line(f"  voice {voice}", [i for i in idx if report[i]["say_voice"] == voice])
    line("  tablet names never trained on", [i for i in idx if report[i]["held_out_drug"]])
    line("  tablet names seen in training", [i for i in idx if not report[i]["held_out_drug"]])
    errors = Counter((report[i]["drug"], p_tab[i]) for i in idx if p_tab[i] != report[i]["drug"])
    print("\nmost common tablet mistakes (said -> picked):", errors.most_common(6))


if __name__ == "__main__":
    main()
