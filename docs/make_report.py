"""Build docs/Dozely_Project_Report.pdf (the plain-language team report).

    .venv/bin/python docs/make_report.py
"""
import json
from pathlib import Path

from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "Dozely_Project_Report.pdf"
REPO = "https://github.com/Hassanahmed-15/dozely-voice-reminder"
SITE = "https://dozely-voice.vercel.app"
COLAB = "https://colab.research.google.com/github/Hassanahmed-15/dozely-voice-reminder/blob/main/colab/train_voice_model.ipynb"

pdfmetrics.registerFont(TTFont("Uni", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"))  # for phone symbols

TEAL = colors.HexColor("#0f766e")
TEAL_SOFT = colors.HexColor("#e3f4f1")
INK = colors.HexColor("#17201c")
MUTED = colors.HexColor("#5d6b65")
LINE = colors.HexColor("#d5ddd8")
AMBER_SOFT = colors.HexColor("#fff4dc")
AMBER = colors.HexColor("#8a5300")

base = dict(fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK)
S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26, leading=31, textColor=INK, spaceAfter=8),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=13, leading=18, textColor=MUTED, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=TEAL, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", **base, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", **base, leftIndent=14, bulletIndent=3, spaceAfter=3),
    "small": ParagraphStyle("small", **{**base, "fontSize": 9, "leading": 12.5, "textColor": MUTED}),
    "cell": ParagraphStyle("cell", **{**base, "fontSize": 9.3, "leading": 12.5}),
    "cellb": ParagraphStyle("cellb", **{**base, "fontName": "Helvetica-Bold", "fontSize": 9.3, "leading": 12.5}),
    "box": ParagraphStyle("box", **base, spaceAfter=4),
    "center": ParagraphStyle("center", **{**base, "alignment": TA_CENTER, "fontSize": 9.5, "leading": 12}),
    "mono": ParagraphStyle("mono", fontName="Courier", fontSize=8.8, leading=12, textColor=INK, leftIndent=8),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def link(url, text=None):
    return f'<link href="{url}" color="#0f766e"><u>{text or url}</u></link>'


def uni(text):
    return f'<font name="Uni">{text}</font>'


def table(rows, widths, header=True, zebra=True):
    data = [[P(c, "cellb" if (header and i == 0) else "cell") if isinstance(c, str) else c for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), TEAL_SOFT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, TEAL)]
    if zebra:
        for i in range(2 if header else 1, len(rows), 2):
            style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f8f7")))
    t.setStyle(TableStyle(style))
    return t


def callout(paragraphs, bg=TEAL_SOFT, edge=TEAL):
    t = Table([[paragraphs]], colWidths=[170 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def flow(steps, width=170 * mm):
    """A row of boxes joined by arrows."""
    cells, widths = [], []
    box_w = (width - (len(steps) - 1) * 7 * mm) / len(steps)
    for i, (title, sub) in enumerate(steps):
        cells.append([P(f"<b>{title}</b>", "center"), P(sub, "small")])
        widths.append(box_w)
        if i < len(steps) - 1:
            cells.append(P("<font size=14 color='#0f766e'>→</font>", "center"))
            widths.append(7 * mm)
    t = Table([cells], colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i in range(0, len(cells), 2):
        st += [("BOX", (i, 0), (i, 0), 0.8, TEAL), ("BACKGROUND", (i, 0), (i, 0), colors.white),
               ("TOPPADDING", (i, 0), (i, 0), 6), ("BOTTOMPADDING", (i, 0), (i, 0), 6)]
    t.setStyle(TableStyle(st))
    return t


def training_chart():
    rows = [json.loads(l) for l in open(ROOT / "logs" / "model.jsonl")]
    d = Drawing(170 * mm, 60 * mm)
    lp = LinePlot()
    lp.x, lp.y, lp.width, lp.height = 14 * mm, 11 * mm, 120 * mm, 41 * mm
    lp.data = [[(r["epoch"] + 1, 100 * r["per_ur"]) for r in rows], [(r["epoch"] + 1, 100 * r["per_en"]) for r in rows]]
    lp.lines[0].strokeColor, lp.lines[1].strokeColor = TEAL, colors.HexColor("#c2410c")
    lp.lines[0].strokeWidth = lp.lines[1].strokeWidth = 1.8
    lp.xValueAxis.valueMin, lp.xValueAxis.valueMax, lp.xValueAxis.valueSteps = 1, 28, [1, 5, 10, 15, 20, 25, 28]
    lp.yValueAxis.valueMin, lp.yValueAxis.valueMax, lp.yValueAxis.valueSteps = 0, 55, [0, 10, 20, 30, 40, 50]
    for ax in (lp.xValueAxis, lp.yValueAxis):
        ax.labels.fontName, ax.labels.fontSize, ax.strokeColor = "Helvetica", 8, MUTED
    lp.yValueAxis.gridStrokeColor, lp.yValueAxis.visibleGrid = LINE, True
    d.add(lp)
    d.add(String(14 * mm, 55 * mm, "Sounds the model gets wrong (%) after each round of training", fontName="Helvetica-Bold", fontSize=9, fillColor=INK))
    d.add(String(74 * mm, 3 * mm, "training round (epoch)", fontName="Helvetica", fontSize=8, fillColor=MUTED, textAnchor="middle"))
    lg = Legend()
    lg.x, lg.y = 140 * mm, 42 * mm
    lg.fontName, lg.fontSize, lg.alignment = "Helvetica", 8.5, "right"
    lg.colorNamePairs = [(TEAL, "Urdu (16.2% at the end)"), (colors.HexColor("#c2410c"), "English (11.0% at the end)")]
    d.add(lg)
    return d


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 12 * mm, "Dozely voice reminder · project report")
    canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    s = []
    # ---------------- cover / summary ----------------
    s += [Spacer(1, 18 * mm), P("Dozely voice reminder", "title"),
          P("What we built, how it works, and what is left to do, explained in plain words.", "subtitle"),
          table([["Where to find things", ""],
                 ["Code (GitHub, public)", link(REPO)],
                 ["Live test page", link(SITE)],
                 ["Train on a free GPU", link(COLAB, "Open the notebook in Google Colab")],
                 ["Date", "17 November 2026"]], [45 * mm, 125 * mm]),
          Spacer(1, 8 * mm), P("The short version", "h1")]
    s += bullets([
        "We are building an app that reminds you to take your tablets. You tell it, by <b>voice or typing</b>, in "
        "<b>Urdu, Roman Urdu or English</b>, which tablet and what time, and an alarm rings at that time.",
        "The project rule is that <b>no pretrained model</b> may be used (for example no Whisper). So we trained our own "
        "speech model from zero, using about <b>90 hours</b> of public speech recordings.",
        "The model writes down the <b>sounds</b> it hears, not words. A second step, the <b>restricted decoder</b>, "
        "then picks the tablet name and the time that best fit those sounds, or says \"not found\".",
        "On test commands spoken by computer voices it gets the tablet, the time and am/pm all right <b>99.6%</b> of the "
        "time. On <b>real people's recordings it is much weaker</b>, and that is the main thing left to fix.",
        "There is a <b>working test website</b> where the model runs inside the browser, plus a <b>Colab notebook</b> to "
        "retrain it on a free GPU.",
    ])
    s += [Spacer(1, 4 * mm), callout([P("<b>The most useful next step:</b> collect a few hundred real recordings of people "
                                       "saying reminder commands, each with the exact sentence written down. Real voices are "
                                       "what the model is missing (see section 9).", "box")], AMBER_SOFT, AMBER), PageBreak()]

    # ---------------- 1. what the app does ----------------
    s += [P("1. What the app should do", "h1"),
          P("A person adds a schedule either by typing or by speaking, for example:"),
          *bullets(["<i>\"Dopahar do baje Augmentin lena hai\"</i> (Roman Urdu: take Augmentin at 2 in the afternoon)",
                    "<i>\"Remind me to take Panadol at 9 pm\"</i> (English)",
                    "The same sentence written in Urdu script also works when typed."]),
          P("The app understands the <b>tablet name</b> and the <b>time</b>, shows them so the person can confirm or fix "
            "them, saves the reminder, and at the exact minute plays an alarm until the person taps <b>Taken</b> or "
            "<b>Snooze</b>."),
          P("2. The one rule: no pretrained models", "h1"),
          P("Many apps use a ready-made speech model such as Whisper, which a big company trained on hundreds of "
            "thousands of hours of audio. Our project is not allowed to do that. Everything the model knows, it learned "
            "from data we chose, starting from random numbers. This makes the job much harder, because we have about "
            "90 hours of speech instead of hundreds of thousands. The whole design below is shaped by that rule."),
          P("3. The big idea: listen for sounds, not words", "h1"),
          P("Instead of learning thousands of Urdu and English words, the model learns about <b>30 basic sounds</b> "
            "(linguists call them <i>phones</i>), like <i>p</i>, <i>a</i>, <i>sh</i> and <i>ch</i>. Both languages are "
            "written with the same set of sounds. For example:"),
          table([["Said", "Sounds the model should write"],
                 ["Panadol", uni("p ɛ n a d ɔ l")],
                 ["baje (o'clock)", uni("b ə ʤ ɛ")],
                 ["remind me", uni("r ɪ m a ɪ n d | m i")]], [45 * mm, 125 * mm]),
          Spacer(1, 3 * mm),
          P("The bar <b>|</b> marks a gap between words. We get the correct sounds for every training sentence "
            "automatically with a free rule-based tool called <b>espeak-ng</b> (it follows spelling rules, it is not a "
            "trained AI model). Three good things follow:"),
          *bullets(["<b>One model for both languages</b>, and for sentences that mix them, like \"Panadol subah aath baje\".",
                    "<b>Roman Urdu needs nothing extra.</b> When people speak, Roman Urdu is just Urdu. It only matters for typing.",
                    "<b>New tablet names work without retraining</b>, because any name can be turned into sounds."]),
          PageBreak()]

    # ---------------- 4. data ----------------
    s += [P("4. The data we trained on", "h1"),
          table([["Data", "What it is", "Amount used", "License"],
                 ["Common Voice Urdu", "Volunteers reading Urdu sentences (Mozilla)", "59 h training", "CC0 (free for anything)"],
                 ["LibriSpeech", "English audiobooks", "25.8 h training", "CC BY 4.0"],
                 ["Medicine commands (ours)", "7,425 spoken reminders we generated with computer voices", "5.9 h", "made by us"],
                 ["Total", "", "about 90 hours", ""]], [38 * mm, 70 * mm, 30 * mm, 32 * mm]),
          Spacer(1, 3 * mm),
          P("Download links: " + link("https://huggingface.co/datasets/khawajaaliarshad/common-voice-urdu-processed-expanded", "Common Voice Urdu (Hugging Face copy)")
            + ", " + link("https://www.openslr.org/12", "LibriSpeech") + ". The full official Common Voice Urdu (about 295 hours) "
            "needs a free account at " + link("https://mozilladatacollective.com", "Mozilla Data Collective") + ". The data is "
            "not stored on GitHub because it is several gigabytes; the script <font name='Courier'>src/download_data.py</font> fetches it."),
          P("Why we made our own medicine commands", "h2"),
          P("Public recordings almost never contain sentences like \"Panadol subah aath baje\". So we wrote sentence "
            "templates in Urdu and English, filled them with 42 tablet names and random times, and had the computer read them "
            "aloud with 17 different voices (Mac voices and espeak-ng). These voices are only used to <i>make</i> training "
            "audio; they are not part of the app."),
          P("Keeping the test honest", "h2"),
          *bullets(["<b>Split by speaker.</b> Nobody whose voice is in the test set appears in training, so test scores show how "
                    "the model does on new voices. (The dataset's own split had 112 people in both training and test, so we did not use it.)",
                    "<b>Held-back tablets and voice.</b> Six tablet names (such as Ventolin and Thyroxine) and one English voice appear only "
                    "in the test set, to check that the model handles names it never heard."]),
          P("Problems we found in the data and fixed", "h2"),
          table([["Problem", "What we did"],
                 ["6 Urdu transcripts had other rows' information (speaker IDs, ages) glued into the text", "Removed them"],
                 ["The sound converter added nonsense syllables to the end of Urdu text in one mode", "Changed how text is passed in"],
                 ["The \"j\" sound (as in <i>baje</i>) was being dropped", "Fixed the sound list"],
                 ["Nasal vowels (as in <i>mein</i>, <i>nahin</i>) were deleted about 41,000 times", "Fixed, then rebuilt all labels"],
                 ["Some clips were too short, too long, or had text that could not match the audio", "Dropped 758 of 80,761 clips"]],
                [110 * mm, 60 * mm]),
          PageBreak()]

    # ---------------- 5. model ----------------
    s += [P("5. The model, in simple terms", "h1"),
          flow([("Microphone audio", "16,000 samples a second"), ("Picture of sound", "80 frequency bands, 100 times a second"),
                ("Speech model", "5 Conformer layers, 3.1 million numbers"), ("Sounds", "one of 31 symbols every 0.04 s"),
                ("Restricted decoder", "tablet + time, or \"not found\"")]),
          Spacer(1, 5 * mm),
          *bullets([
              "<b>Picture of sound (log-mel features).</b> The audio is cut into short slices, and for each slice we measure how "
              "loud each of 80 pitch bands is, roughly the way the human ear hears.",
              "<b>Conformer layers.</b> A popular design for speech. Each layer looks at nearby sounds (convolution) and at the "
              "whole sentence at once (attention). Ours is deliberately small, about <b>3.1 million</b> numbers (Whisper Large has "
              "over 1,500 million), so it trains on a laptop and runs on a phone.",
              "<b>CTC training.</b> The model is never told exactly when each sound starts. It is only given the list of sounds "
              "for the whole sentence, and the CTC method lets it work out the timing by itself.",
              "<b>Making training harder on purpose.</b> Each time a clip is used it is slightly changed: a bit faster or slower "
              "(±10%), some background hiss added, and small blocks of time or pitch hidden (SpecAugment). This stops the model "
              "from memorizing and helps with real microphones.",
          ]),
          P("6. How we trained it", "h1"),
          table([["Setting", "Value", "In plain words"],
                 ["Computer", "MacBook (Apple M4), its built-in GPU", "No cloud needed"],
                 ["Training rounds", "28 epochs (30 planned)", "The model saw all 90 hours 28 times"],
                 ["Time", "about 9.5 minutes per round, 4.5 hours in total", "It took longer in practice because the laptop went to sleep"],
                 ["Learning speed", "AdamW optimizer, one-cycle schedule, top rate 0.0015", "Starts gently, speeds up, then slows to a stop"],
                 ["Batch size", "about 4 minutes of audio per step", ""],
                 ["Model saved", "after every round; the best one is kept", "checkpoints/model_best.pt"]],
                [32 * mm, 66 * mm, 72 * mm]),
          Spacer(1, 4 * mm), training_chart(),
          P("The error drops fast at first and then flattens, which means more rounds of the same training would not help much. "
            "Training stopped two rounds early because of a small counting bug in the learning-speed schedule. The learning speed "
            "was already almost zero by then, so it made no real difference, and the bug is fixed.", "small"),
          PageBreak()]

    # ---------------- 7. decoder ----------------
    s += [P("7. From sounds to a reminder: the restricted decoder", "h1"),
          P("Writing down sounds freely is error-prone. \"Panadol\" might come out as \"benadal\". But the app does not need "
            "<i>any</i> sentence, only a tablet and a time. So instead of asking \"what was said?\", the decoder asks: "
            "<b>\"Which tablet from our list, and which time phrase, fits this audio best?\"</b> Since \"benadal\" is far closer "
            "to Panadol than to any other tablet, the right answer wins."),
          *bullets([
              "<b>Tablets:</b> 42 names, each in English and Urdu spelling (83 spellings in total).",
              "<b>Times:</b> 5,836 ways of saying a time: Urdu (<i>aath baje</i>, <i>saade aath</i>, <i>sawa</i>, <i>paune</i>, "
              "<i>dhai</i>, <i>baj kar bees minute</i>), English (<i>at nine</i>, <i>nine forty five pm</i>, <i>half past</i>, "
              "<i>quarter to</i>, <i>noon</i>), and English numbers inside Urdu (<i>ten baje</i>).",
              "<b>Every time phrase needs an anchor word</b> such as <i>baje</i>, <i>at</i>, <i>pm</i> or <i>o'clock</i>. Without "
              "this, the word \"do\" in \"laga do\" was mistaken for \"2 o'clock\".",
              "<b>Morning or evening:</b> Urdu uses <i>subah, dopahar, shaam, raat</i>. For English, am/pm are judged together with "
              "the time (\"nine p m\" versus \"nine a m\"), because the words alone are too short to tell apart.",
              "<b>\"Not found\":</b> every answer gets a fit score. Below a cut-off, the decoder says it found no tablet or no "
              "time, and the app asks the person instead of guessing. We set the cut-offs so that only 5% of 600 ordinary "
              "sentences (with no tablet or time in them) are wrongly accepted.",
          ]),
          P("Results on held-out spoken commands", "h2"),
          P("These are 506 computer-voice commands that were never used for training or for choosing settings "
            "(we chose settings on a separate half)."),
          table([["", "Free reading of sounds", "Restricted decoder"],
                 ["Tablet correct", "87.4%", "99.6%"],
                 ["Time correct", "-", "100%"],
                 ["Tablet, time and am/pm all correct", "-", "99.6%"],
                 ["Tablets never seen in training", "84.7%", "99.5%"]], [80 * mm, 45 * mm, 45 * mm]),
          Spacer(1, 3 * mm),
          callout([P("<b>Careful:</b> these are computer voices, which are clean and regular. They show the method works, "
                     "not that the app is ready for real people.", "box")], AMBER_SOFT, AMBER),
          P("8. Testing on real recordings", "h1"),
          P("We recorded 20 real phone clips (12 Urdu, 8 English). They include not only new reminders but also commands such as "
            "snooze, delete, \"what medicines today?\" and \"I have taken it\". They are kept private and are not on GitHub."),
          *bullets(["The free reading of sounds was <b>very different</b> from what was actually said.",
                    "With the restricted decoder, it found a tablet in 9 clips and a time in 7. We do not yet know how many of "
                    "these are right, because the clips have no written transcripts. From listening, the team found the "
                    "<b>times often wrong and the tablets sometimes wrong</b>.",
                    "We have deliberately <b>not tuned anything on these 20 clips</b>, so that they stay an honest test."]),
          PageBreak()]

    # ---------------- 9. why real voices are hard ----------------
    s += [P("9. Why real voices are harder, and how to fix it", "h1"),
          P("The model learned from people reading sentences calmly, from audiobooks, and from computer voices. A real person "
            "speaking a quick command into a phone, with a room echo and their own accent, sounds different from all of these. "
            "This is the biggest gap. In order of how much they should help:"),
          table([["Step", "Why it helps", "Effort"],
                 ["1. Record real commands (200 to 500 clips from 10 to 20 people), each with the exact sentence written down",
                  "Gives the model the kind of audio it actually has to handle, and lets us measure accuracy properly", "Medium, and the most valuable"],
                 ["2. Write down what each of the 20 test clips says (file: data/test/transcripts.txt)", "Lets us score the model and find exact problems", "Small"],
                 ["3. Use the full Common Voice Urdu (about 295 hours instead of 59)", "Five times more Urdu speech", "Free account, then retrain"],
                 ["4. Train a bigger model on Colab's GPU", "More capacity to learn from more data", "Change two numbers in the notebook"],
                 ["5. Teach it command types: add, snooze, delete, taken, what is due", "Stops it from inventing a time in a snooze command", "Medium"]],
                [70 * mm, 68 * mm, 32 * mm]),
          Spacer(1, 3 * mm),
          P("The website already helps with step 1: after every voice try it offers <b>Download this recording</b>."),
          P("10. The test website", "h1"),
          P(f"Address: {link(SITE)}. The whole model runs <b>inside the browser</b> (converted to a format called ONNX), so "
            "it works on a phone or a laptop without a server, and <b>recordings never leave the device</b>."),
          *bullets(["Open the page and wait for <b>Model ready</b> (the model is about 12 MB).",
                    "Tap the yellow banner once so the alarm is allowed to make sound.",
                    "Tap the microphone and speak, upload a recording, or type a reminder.",
                    "Check the <b>Is this right?</b> box, fix anything wrong, choose AM or PM if asked, then <b>Save</b>.",
                    "Press <b>Test</b> next to a reminder to hear the alarm right away."]),
          P("Limits: alarms only ring while the page is open (use <b>Keep screen on</b> on a phone). By voice it only knows the "
            "42 built-in tablets; typing accepts any name. The alarm is a beep, because we have not yet decided how the spoken "
            "reminder voice will work. We checked that the browser version gives exactly the same answers as the Python "
            "version on all 20 real clips, but recording from the microphone in a real browser has not been tested yet.", "small"),
          PageBreak()]

    # ---------------- 11. colab ----------------
    s += [P("11. The Colab notebook (train_voice_model.ipynb)", "h1"),
          P(f"The notebook runs the same training on Google's free GPU. Open it from {link(COLAB, 'this link')}, choose "
            "<b>Runtime → Change runtime type → T4 GPU</b>, then <b>Runtime → Run all</b>. What each part does:"),
          table([["Cell", "What it does"],
                 ["1. Check the GPU", "Confirms a GPU is switched on"],
                 ["2. Settings and Google Drive", "Training length and model size (EPOCHS, DIM, BLOCKS); connects Drive, where the model is saved after every round"],
                 ["3. Install tools", "Installs espeak-ng and a few Python libraries"],
                 ["4. Project code", "Writes the same Python files that are in src/ (they are copied in automatically, so they always match)"],
                 ["5. Download speech data", "About 3.6 GB of Urdu and English recordings"],
                 ["6. Medicine commands", "Uses commands.zip from Drive if you put it there (nicer voices made on a Mac); otherwise makes them with espeak-ng"],
                 ["7. Build features", "Cleans the data, splits it by speaker, and turns audio into the \"picture of sound\""],
                 ["8. Train", "Trains and saves after every round. If Colab disconnects, run all again and it continues where it stopped"],
                 ["9 and 10. Results, download", "Shows the error for each round, then downloads the best model to your computer"]], [45 * mm, 125 * mm]),
          Spacer(1, 3 * mm),
          P("The notebook's GPU-specific speed-up (half-precision maths) could not be tested on the Mac, so if cell 8 shows an "
            "error or the loss says <font name='Courier'>nan</font>, report it. Free Colab sessions end after a few hours; "
            "because progress is saved to Drive, nothing is lost.", "small"),
          P("12. What is in the GitHub repository", "h1"),
          table([["File or folder", "What it is for"],
                 ["src/download_data.py", "Downloads the Urdu and English speech data"],
                 ["src/generate_commands.py", "Makes the 7,425 spoken medicine commands"],
                 ["src/phones.py", "Turns text into sounds (the shared Urdu + English alphabet)"],
                 ["src/features.py", "Turns audio into the \"picture of sound\""],
                 ["src/prepare.py", "Cleans data, splits by speaker, builds features and labels"],
                 ["src/model.py", "The speech model itself"],
                 ["src/train.py", "Training (Mac GPU, NVIDIA GPU or CPU); can stop and resume"],
                 ["src/decoder.py", "The restricted decoder (tablet, time, am/pm, \"not found\")"],
                 ["src/evaluate_commands.py, calibrate.py", "Measure accuracy; choose the \"not found\" cut-offs"],
                 ["src/transcribe.py", "Runs the model on a folder of recordings"],
                 ["src/export_web.py, build_colab.py", "Make the website files and the Colab notebook"],
                 ["checkpoints/model_best.pt", "The trained model"],
                 ["web/, colab/, logs/", "Test website (Vercel), Colab notebook, training history and results"]], [62 * mm, 108 * mm]),
          PageBreak()]

    # ---------------- 13. run / 14. glossary ----------------
    s += [P("13. Running it yourself", "h1"),
          P("On a Mac or Linux computer with Python 3.12:"),
          *[P(line, "mono") for line in [
              "brew install espeak-ng ffmpeg",
              "python3.12 -m venv .venv",
              ".venv/bin/pip install -r requirements.txt",
              ".venv/bin/python src/download_data.py      # about 3.6 GB",
              ".venv/bin/python src/generate_commands.py",
              ".venv/bin/python src/prepare.py            # needs about 6 GB of disk",
              ".venv/bin/python src/train.py --epochs 30  # about 5 hours on an M4 Mac",
              ".venv/bin/python src/transcribe.py my_recordings/"]],
          Spacer(1, 4 * mm),
          P("14. Words used in this report", "h1"),
          table([["Word", "Meaning"],
                 ["Phone", "One basic speech sound, like p, a or sh"],
                 ["Epoch (training round)", "One pass of the model over all the training data"],
                 ["Phone error rate", "Out of every 100 sounds, how many the model got wrong. Lower is better"],
                 ["Validation / test set", "Recordings kept aside to check the model on voices it never trained on"],
                 ["CTC", "A training method that lets the model work out the timing of sounds by itself"],
                 ["Conformer", "A type of neural network layer that is good at speech"],
                 ["Restricted decoder", "Our step that only accepts known tablet names and time phrases"],
                 ["Fit score", "How cleanly a tablet or time phrase matches the audio. Near 0 is a clean match"],
                 ["Pretrained model", "A model someone else already trained on huge data, such as Whisper. Not allowed in this project"],
                 ["ONNX", "A file format that lets the model run inside a web browser"]], [45 * mm, 125 * mm]),
          Spacer(1, 6 * mm),
          callout([P("<b>Questions to decide as a team</b>", "box"),
                   *bullets(["How should the reminder <b>speak</b>? Options: recorded human phrases plus the user's own voice for the "
                             "tablet name, training our own text-to-speech model (days of work, robotic at first), or the phone's "
                             "built-in voice (easiest, but it is a pretrained engine)."], "bullet"),
                   *bullets(["Who records the real command clips, and which tablet names matter most to our users?"], "bullet")])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Dozely voice reminder: project report", author="Dozely team", subject="Plain-language project report")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
