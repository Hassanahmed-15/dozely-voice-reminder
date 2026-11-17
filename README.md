# Dozely: Urdu + English voice medicine reminders

A voice reminder for tablets. You say or type something like *"dopahar do baje Augmentin lena hai"*
or *"remind me to take Panadol at 9 pm"*, and the app sets an alarm that rings at that time.

The speech model is **trained from scratch**, with no Whisper or other pretrained model.

- **Live test page:** https://dozely-voice.vercel.app (the model runs in your browser)
- **Train it on a free GPU:** [Open the notebook in Colab](https://colab.research.google.com/github/Hassanahmed-15/dozely-voice-reminder/blob/main/colab/train_voice_model.ipynb)
- **Team report (plain-language explanation):** [docs/Dozely_Project_Report.pdf](docs/Dozely_Project_Report.pdf)

## How it works, in one paragraph

The model listens to audio and writes down the **sounds** it hears (phones such as *p, a, n*), not words.
Urdu and English share one alphabet of sounds, so one model handles both, and spoken Roman Urdu is simply Urdu.
A **restricted decoder** then asks which tablet name from the list, and which time phrase, fit those sounds best.
If nothing fits well enough, it says "not found" and the app asks you to confirm or type the reminder.

## Datasets

The data is not stored in this repo, because it is several gigabytes. `src/download_data.py` downloads it.

| Data | What it is | Size used | License | Link |
|---|---|---|---|---|
| Common Voice Urdu | People reading Urdu sentences | 59 h train, 2.4 h validation, 2.9 h test | CC0 | [Hugging Face copy](https://huggingface.co/datasets/khawajaaliarshad/common-voice-urdu-processed-expanded), [official (free account)](https://mozilladatacollective.com) |
| LibriSpeech train-clean-100 | English audiobooks | first 25.8 h | CC BY 4.0 | [openslr.org/12](https://www.openslr.org/12) |
| Spoken medicine commands | 7,425 clips made by `src/generate_commands.py` with system voices | 5.9 h | made by us | generated |
| Real test recordings | 20 phone recordings (12 Urdu, 8 English) | 20 clips | private | not published |

Every split separates speakers: nobody in the test set appears in training.
Six tablet names and one English voice appear only in the test set.

## Repository structure

```
src/                      the whole pipeline (Python)
  download_data.py        download Urdu and English speech data
  generate_commands.py    make spoken medicine commands with system voices
  phones.py               text to sounds (phones), shared by Urdu and English
  features.py             audio to log-mel features
  prepare.py              clean the data, split by speaker, build features and labels
  model.py                the speech model (a small Conformer, 3.1 M parameters)
  train.py                training (Apple GPU, NVIDIA GPU, or CPU), resumable
  transcribe.py           run the model on audio files
  decoder.py              restricted decoder: tablet, time, am/pm, with "not found"
  evaluate_commands.py    accuracy on held-out spoken commands
  calibrate.py            choose the "not found" cut-offs (saves thresholds.json)
  export_web.py           export the model to ONNX and the decoder data to JSON
  build_colab.py          build the Colab notebook from the files in src/
colab/train_voice_model.ipynb   the same training pipeline, for Google Colab
checkpoints/model_best.pt       the trained model
web/                      the test page deployed on Vercel (runs fully in the browser)
logs/                     training history and test results
docs/                     the team report (PDF)
```

## Run it yourself

```bash
brew install espeak-ng ffmpeg            # or apt-get install espeak-ng ffmpeg
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python src/download_data.py    # about 3.6 GB
.venv/bin/python src/generate_commands.py
.venv/bin/python src/prepare.py          # builds data/features (about 5.6 GB)
.venv/bin/python src/train.py --epochs 30

.venv/bin/python src/evaluate_commands.py
.venv/bin/python src/calibrate.py
.venv/bin/python src/transcribe.py path/to/recordings/
.venv/bin/python src/export_web.py       # then deploy web/ to Vercel
```

## Results so far

| Test | Result |
|---|---|
| Sounds right, Urdu validation speakers never heard in training | 83.8% (phone error rate 16.2%) |
| Sounds right, English validation speakers | 89.0% (phone error rate 11.0%) |
| Held-out spoken commands: tablet, time and am/pm all correct | 99.6% (synthetic voices, so much easier than real people) |
| 20 real phone recordings | not scored yet: needs transcripts. The team found the times often wrong and the tablets sometimes wrong. |

## Known limitations

- Real voices are much harder than the training data. Collecting a few hundred real command recordings is the most important next step.
- By voice, only the 42 built-in tablet names are recognized. Typed reminders accept any name.
- The app does not yet understand snooze, delete or "I have taken it" commands by voice.
- Alarms on the test page ring only while the page is open.
