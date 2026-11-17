"""Turn raw audio into one feature file plus a manifest, split by speaker.

Output:
  data/features/mels.bin        float16 log-mels, all clips concatenated
  data/features/manifest.jsonl  one row per clip (offset, frames, tokens, split, ...)

No speaker appears in more than one split, so validation and test scores measure how
the model does on voices it has never heard.
"""
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).parent))

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

import features as fx
import phones

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "features"
MIN_SEC, MAX_SEC = 0.4, 16.0
MAX_PHONES_PER_SEC = 22  # faster than people speak, so the transcript cannot match the audio
CORRUPT_TEXT = re.compile(r"[0-9a-f]{32}|twenties|thirties|fourties|fifties|teens|sixties|_feminine|_masculine|\t")


def speaker_split(speaker: str) -> str:
    bucket = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 100
    return "test" if bucket < 6 else "val" if bucket < 12 else "train"


def urdu_rows():
    seen = set()
    stats = Counter()
    for path in sorted((RAW / "urdu_cv").glob("*.parquet")):
        table = pq.ParquetFile(path)
        for batch in table.iter_batches(batch_size=256, columns=["client_id", "sentence", "audio", "up_votes", "down_votes"]):
            for row in batch.to_pylist():
                stats["urdu_rows"] += 1
                key = (row["client_id"], row["sentence"])
                if key in seen:
                    stats["urdu_duplicates"] += 1
                    continue
                seen.add(key)
                # Some rows in the upload have neighbouring TSV rows merged into the
                # transcript, leaving speaker hashes and metadata in the text.
                if CORRUPT_TEXT.search(row["sentence"]) or len(row["sentence"]) > 300:
                    stats["urdu_corrupt_text"] += 1
                    continue
                if row["down_votes"] > row["up_votes"]:
                    stats["urdu_downvoted"] += 1
                    continue
                yield {"bytes": row["audio"]["bytes"], "text": row["sentence"], "voice": "ur", "speaker": "ur" + row["client_id"][:16],
                       "lang": "ur", "source": "common_voice"}
    print(dict(stats), flush=True)


def english_rows():
    base = RAW / "librispeech"
    for trans in sorted(base.glob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            wav = base / f"{uid}.flac"
            if wav.exists():
                yield {"file": str(wav), "text": text.lower(), "voice": "en-us", "speaker": "en" + uid.split("-")[0],
                       "lang": "en", "source": "librispeech"}


def command_rows():
    base = RAW / "commands"
    if not (base / "manifest.jsonl").exists():
        print("no command clips found, skipping them", flush=True)
        return
    for line in open(base / "manifest.jsonl", encoding="utf-8"):
        r = json.loads(line)
        yield {"file": str(base / r["wav"]), "text": r["label"], "voice": r["label_voice"], "speaker": "syn" + r["voice"], "lang": r["lang"],
               "source": "command", "split": r["split"], "drug": r["drug"], "hour": r["hour"], "minute": r["minute"], "period": r["period"],
               "held_out_drug": r["held_out_drug"], "say_voice": r["voice"]}


def work(row: dict):
    try:
        src = io.BytesIO(row.pop("bytes")) if "bytes" in row else row["file"]
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
    except Exception as e:
        return None, f"decode:{type(e).__name__}"
    wav = wav.mean(1)
    if sr != fx.SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=fx.SR)
    sec = len(wav) / fx.SR
    if not MIN_SEC <= sec <= MAX_SEC:
        return None, "duration"
    # Trim silence at the ends, keeping a short margin.
    intervals = librosa.effects.split(wav, top_db=40, frame_length=400, hop_length=160)
    if len(intervals):
        wav = wav[max(0, intervals[0][0] - 1600): intervals[-1][1] + 1600]
    mel = fx.log_mel(wav).astype(np.float16)
    row.pop("file", None)
    row["frames"] = len(mel)
    return (row, mel.tobytes()), None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    def sources():
        yield from urdu_rows()
        yield from english_rows()
        yield from command_rows()

    offset = 0
    drops = Counter()
    with Pool(os.cpu_count() or 1) as pool, open(OUT / "mels.bin", "wb") as bin_f:
        for i, (ok, why) in enumerate(pool.imap(work, sources(), chunksize=32)):
            if ok is None:
                drops[why] += 1
                continue
            row, data = ok
            row["offset"] = offset
            offset += row["frames"]
            bin_f.write(data)
            rows.append(row)
            if len(rows) % 5000 == 0:
                print(f"{len(rows)} clips, {offset/360000:.1f} h", flush=True)

    print("converting text to phones", flush=True)
    unknown: dict = {}
    tokens = phones.to_tokens_many([(r["text"], r["voice"]) for r in rows], unknown=unknown)
    kept, splits = [], Counter()
    for r, t in zip(rows, tokens):
        # CTC needs at least one output frame per label after 4x subsampling.
        if not t or len(t) > r["frames"] // 4 - 1:
            drops["label_length"] += 1
            continue
        if len(t) > r["frames"] / 100 * MAX_PHONES_PER_SEC:
            drops["label_rate"] += 1
            continue
        r["tokens"] = " ".join(t)
        r["split"] = r.get("split") or speaker_split(r["speaker"])
        splits[(r["source"], r["split"])] += r["frames"]
        kept.append(r)
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("kept", len(kept), "dropped", dict(drops))
    print("hours by source and split:")
    for k, v in sorted(splits.items()):
        print("  ", k, round(v / 360000, 2))
    print("unknown phones (top):", Counter(unknown).most_common(12))


if __name__ == "__main__":
    main()
