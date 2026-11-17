"""Download raw speech data into data/raw/.

Urdu:    Common Voice Urdu parquet shards (public, no login).
English: a capped portion of LibriSpeech train-clean-100, streamed so that only
         the audio we keep ever reaches the disk.

Downloaded files are treated as data only: parquet is read as columns, and tar
members are written to paths we build ourselves, never to paths from the archive.
"""
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = {"User-Agent": "voice-reminder-trainer/1.0"}

URDU_REPO = "khawajaaliarshad/common-voice-urdu-processed-expanded"
URDU_FILES = [
    "README.md",
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
    "data/validation-00000-of-00001.parquet",
    "data/test-00000-of-00001.parquet",
]
LIBRI_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRI_MAX_BYTES = int(2.0e9)  # about 25 hours of FLAC


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)
    print(f"got {dest.name} {dest.stat().st_size/1e6:.0f} MB", flush=True)


def download_urdu() -> None:
    base = f"https://huggingface.co/datasets/{URDU_REPO}/resolve/main/"
    for name in URDU_FILES:
        fetch(base + name, RAW / "urdu_cv" / Path(name).name)


def download_english() -> None:
    out = RAW / "librispeech"
    done = out / ".complete"
    if done.exists():
        print("have librispeech", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    flac = re.compile(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+-\d+-\d+)\.flac$")
    trans = re.compile(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/(\d+-\d+)\.trans\.txt$")
    written = clips = 0
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(LIBRI_URL, headers=UA), timeout=120) as r:
        with tarfile.open(fileobj=r, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile():
                    continue
                if mf := flac.match(m.name):
                    if written >= LIBRI_MAX_BYTES:
                        break
                    target = out / f"{mf.group(3)}.flac"
                elif mt := trans.match(m.name):
                    target = out / f"{mt.group(3)}.trans.txt"
                else:
                    continue
                data = tar.extractfile(m).read()
                target.write_bytes(data)
                if target.suffix == ".flac":
                    written += len(data)
                    clips += 1
                    if clips % 2000 == 0:
                        print(f"librispeech {clips} clips {written/1e9:.2f} GB {time.time()-t0:.0f}s", flush=True)
    done.write_text(f"{clips} clips\n")
    print(f"librispeech done: {clips} clips, {written/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "urdu"):
        download_urdu()
    if which in ("all", "english"):
        download_english()
