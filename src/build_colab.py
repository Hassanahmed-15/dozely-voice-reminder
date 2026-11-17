"""Build colab/train_voice_model.ipynb from the files in src/, and optionally pack the
Mac-generated command clips into data/export/commands.zip for upload to Google Drive.

    python src/build_colab.py                 # notebook only
    python src/build_colab.py --pack-commands # also make the zip
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FILES = ["phones.py", "features.py", "model.py", "download_data.py", "generate_commands.py", "prepare.py", "train.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, hidden=False):
    meta = {"cellView": "form"} if hidden else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def build():
    cells = [
        md("""
# Urdu + English voice reminder: train the speech model

This trains a small speech model **from scratch** (random start, no Whisper or other pretrained model).
It hears Urdu (including what people write as Roman Urdu) and English, and outputs phonetic sounds,
so tablet names and times can be matched afterwards.

**Before you start:** *Runtime → Change runtime type → T4 GPU*, then *Runtime → Run all*.

- Data: Common Voice Urdu (CC0), LibriSpeech English, and spoken medicine commands.
- Checkpoints are saved to your Google Drive after every epoch, in `MyDrive/voice_reminder/checkpoints`.
- **If Colab disconnects**, just run all cells again. The data steps repeat (about 20 to 30 minutes),
  then training continues from the last saved epoch.
- Optional: put `commands.zip` (made on the Mac) in `MyDrive/voice_reminder/`. Those clips use more
  natural voices than the robotic ones this notebook generates by itself.
"""),
        code("""
#@title 1. Check the GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "NO GPU: choose Runtime > Change runtime type > T4 GPU"
"""),
        code("""
#@title 2. Settings and Google Drive
EPOCHS = 30      #@param {type:"integer"}
DIM = 144        #@param {type:"integer"}
BLOCKS = 5       #@param {type:"integer"}
RUN_NAME = "colab_model"  #@param {type:"string"}

from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE_DIR = '/content/drive/MyDrive/voice_reminder'
CKPT_DIR = DRIVE_DIR + '/checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs('/content/hijab/src', exist_ok=True)
os.makedirs('/content/hijab/logs', exist_ok=True)
%cd /content/hijab
"""),
        code("""
#@title 3. Install tools
!apt-get -qq install -y espeak-ng > /dev/null && espeak-ng --version
!pip -q install librosa soundfile pyarrow
"""),
        md("## 4. Project code\nThese cells write the same source files used on the Mac."),
    ]
    for name in FILES:
        cells.append(code(f"%%writefile /content/hijab/src/{name}\n" + (SRC / name).read_text(encoding="utf-8")))
    cells += [
        code("""
#@title 5. Download speech data (about 3.6 GB)
!python src/download_data.py all
"""),
        code("""
#@title 6. Medicine command clips: use commands.zip from Drive if present, otherwise generate them
import os, zipfile
zip_path = DRIVE_DIR + '/commands.zip'
os.makedirs('data/raw/commands', exist_ok=True)
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        for m in z.namelist():
            name = os.path.basename(m)
            if name and (name.endswith('.flac') or name == 'manifest.jsonl'):
                with open('data/raw/commands/' + name, 'wb') as f:
                    f.write(z.read(m))
    print('using the clips from', zip_path)
else:
    print('no commands.zip on Drive, generating clips with espeak-ng')
    !python src/generate_commands.py
"""),
        code("""
#@title 7. Build features and phonetic labels
!python src/prepare.py
"""),
        code("""
#@title 8. Train (re-run after a disconnect to continue)
!python src/train.py --epochs {EPOCHS} --dim {DIM} --blocks {BLOCKS} --name {RUN_NAME} --ckpt-dir "{CKPT_DIR}" --resume 2>&1 | tee -a logs/train.log
"""),
        code("""
#@title 9. Results
import json
for line in open(f'logs/{RUN_NAME}.jsonl'):
    r = json.loads(line)
    print(f"epoch {r['epoch']:2d}  urdu errors {r['per_ur']:.1%}  english errors {r['per_en']:.1%}  ({r['minutes']} min)")
print('\\nBest model saved at', f'{CKPT_DIR}/{RUN_NAME}_best.pt')
"""),
        code("""
#@title 10. Download the best model to this computer (optional, it is also on Drive)
from google.colab import files
files.download(f'{CKPT_DIR}/{RUN_NAME}_best.pt')
"""),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "colab" / "train_voice_model.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out)


def pack_commands():
    import soundfile as sf
    base = ROOT / "data" / "raw" / "commands"
    out = ROOT / "data" / "export" / "commands.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(base / "manifest.jsonl", encoding="utf-8")]
    tmp = out.parent / "flac_tmp"
    tmp.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            wav, sr = sf.read(base / r["wav"], dtype="int16")
            name = Path(r["wav"]).with_suffix(".flac").name
            sf.write(tmp / name, wav, sr, format="FLAC")
            z.write(tmp / name, name)
            (tmp / name).unlink()
            r["wav"] = name
        z.writestr("manifest.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.rmdir()
    print("wrote", out, f"{out.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    build()
    if "--pack-commands" in sys.argv:
        pack_commands()
