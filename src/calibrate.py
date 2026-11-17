"""Choose the "is it really there?" cut-offs for tablet and time detection.

Positives: the tuning half of the held-out synthetic commands (they always contain both).
Negatives: ordinary validation sentences from Common Voice and LibriSpeech, which almost
never contain a tablet name or a clock time. The user's real test clips are not used.

The cut-off is the lowest fit at which at most 5% of the negatives would be accepted.

    python src/calibrate.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import decoder
import transcribe as ours
from evaluate_commands import FEAT, log_probs

MAX_FALSE_ALARMS = 0.05


def fits(understand, lps):
    out = [understand(lp) for lp in lps]
    return np.array([u["tablet_fit"] for u in out]), np.array([u["time_fit"] for u in out]), out


def main():
    rows = [json.loads(l) for l in open(FEAT / "manifest.jsonl", encoding="utf-8")]
    test = sorted((r for r in rows if r["source"] == "command" and r["split"] == "test"), key=lambda r: r["offset"])
    positives = test[0::2]
    negatives = [r for r in rows if r["split"] == "val" and r["source"] in ("common_voice", "librispeech")]
    random.Random(3).shuffle(negatives)
    negatives = negatives[:600]

    model = ours.load_model()
    understand = decoder.Understander()
    pos_tab, pos_time, pos_u = fits(understand, log_probs(model, positives))
    neg_tab, neg_time, _ = fits(understand, log_probs(model, negatives))

    result = {}
    for name, pos, neg, correct in [
        ("tablet", pos_tab, neg_tab, np.array([u["tablet"] == r["drug"] for u, r in zip(pos_u, positives)])),
        ("time", pos_time, neg_time, np.array([(u["hour"], u["minute"]) == (r["hour"], r["minute"]) for u, r in zip(pos_u, positives)])),
    ]:
        cut = float(np.quantile(neg, 1 - MAX_FALSE_ALARMS))
        kept = pos >= cut
        result[name] = round(cut, 3)
        print(f"{name}: cut-off {cut:.2f} | ordinary sentences wrongly accepted {np.mean(neg >= cut):.1%} | "
              f"commands accepted {kept.mean():.1%}, of which correct {correct[kept].mean():.1%}")
        print(f"    fit ranges: commands median {np.median(pos):.2f}, ordinary sentences median {np.median(neg):.2f}")
    out = Path(__file__).with_name("thresholds.json")
    out.write_text(json.dumps(result, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
