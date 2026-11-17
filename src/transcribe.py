"""Run the trained model on audio files and show what it heard.

    python src/transcribe.py path/to/folder_or_files ...

For each file it prints the phones, a readable Roman spelling of them, and the closest
known tablet name, if any is close enough.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import librosa
import numpy as np
import torch

import features as fx
import phones
from generate_commands import DRUGS, HELD_OUT_DRUGS
from model import SpeechModel

ROOT = Path(__file__).resolve().parent.parent
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".webm"}
ROMAN = {"ʃ": "sh", "ʧ": "ch", "ʤ": "j", "ə": "a", "ɛ": "e", "ɪ": "i", "ʊ": "u", "ɔ": "o", "|": " "}


def load_model(path=ROOT / "checkpoints" / "model_best.pt"):
    c = torch.load(path, map_location="cpu")
    model = SpeechModel(**c["config"])
    model.load_state_dict(c["model"])
    model.eval()
    return model


def read_audio(path: Path) -> np.ndarray:
    """Decode any common format to 16 kHz mono with ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(fx.SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    wav = np.frombuffer(raw, dtype=np.float32).copy()
    intervals = librosa.effects.split(wav, top_db=40, frame_length=400, hop_length=160)
    if len(intervals):
        wav = wav[max(0, intervals[0][0] - 1600): intervals[-1][1] + 1600]
    return wav


@torch.no_grad()
def log_probs(model, wav: np.ndarray) -> np.ndarray:
    mel = torch.from_numpy(fx.normalize(fx.log_mel(wav)))[None]
    lp, _ = model(mel, torch.tensor([mel.shape[1]]))
    return lp[0].numpy()


def recognize(model, wav: np.ndarray) -> list[str]:
    return recognize_lp(log_probs(model, wav))


def recognize_lp(lp: np.ndarray) -> list[str]:
    ids, prev = [], 0
    for t in lp.argmax(-1).tolist():
        if t != prev and t != 0:
            ids.append(t)
        prev = t
    return [phones.VOCAB[i] for i in ids]


def romanize(tokens: list[str]) -> str:
    return "".join(ROMAN.get(t, t) for t in tokens)


def _lexicon():
    lex = []
    for latin, urdu, _ in DRUGS + HELD_OUT_DRUGS:
        variants = {tuple(t for t in phones.to_tokens(latin.lower(), "en-us") if t != phones.SPACE),
                    tuple(t for t in phones.to_tokens(urdu, "ur") if t != phones.SPACE)}
        lex.append((latin, variants))
    return lex


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def best_tablet(tokens: list[str], lexicon, threshold: float = 0.34):
    """Slide each known name over the heard phones; return the closest if it is close enough."""
    heard = [t for t in tokens if t != phones.SPACE]
    best = (9.0, None)
    for name, variants in lexicon:
        for v in variants:
            n = len(v)
            for size in range(max(1, n - 2), n + 3):
                for s in range(0, max(1, len(heard) - size + 1)):
                    d = edit_distance(v, heard[s: s + size]) / n
                    if d < best[0]:
                        best = (d, name)
    return (best[1], best[0]) if best[0] <= threshold else (None, best[0])


def main(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT) if p.is_dir() else [p]
    import decoder
    model, lexicon, understand = load_model(), _lexicon(), decoder.Understander()
    thresholds = decoder.load_thresholds()
    for f in files:
        wav = read_audio(f)
        lp = log_probs(model, wav)
        tokens = recognize_lp(lp)
        u = decoder.accept(understand(lp), thresholds)
        tablet, dist = best_tablet(tokens, lexicon)
        guess = f"{tablet} (difference {dist:.0%})" if tablet else f"none close enough (nearest differs by {dist:.0%})"
        print(f"\n{f.parent.name}/{f.name}  [{len(wav)/fx.SR:.1f}s]")
        print(f"  heard:  {romanize(tokens)}")
        print(f"  phones: {' '.join(tokens)}")
        print(f"  tablet: {guess}")
        tab_txt = u["tablet"] if u["tablet_found"] else f"no tablet (closest {u['tablet']})"
        time_txt = f"{u['hour']}:{u['minute']:02d} {u['period']}" if u["time_found"] else f"no time (closest {u['hour']}:{u['minute']:02d})"
        print(f"  restricted decoder: {tab_txt} (fit {u['tablet_fit']:.2f}) | {time_txt} (fit {u['time_fit']:.2f})")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(ROOT / "data" / "test")])
