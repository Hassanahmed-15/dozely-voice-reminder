"""Synthesize spoken reminder commands so the model hears tablet names and clock times.

Public speech corpora almost never contain "Panadol subah aath baje". These clips use
the system voices purely to make training audio. Six tablet names and one English
voice are withheld for testing, so the test set contains things never trained on.

Every clip records its intended tablet and time for task-level evaluation later.
"""
import json
import random
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "commands"

# (Latin, Urdu script, Devanagari)
DRUGS = [
    ("Panadol", "پیناڈول", "पैनाडोल"), ("Augmentin", "آگمینٹن", "ऑगमेंटिन"), ("Brufen", "بروفین", "ब्रूफेन"),
    ("Disprin", "ڈسپرین", "डिस्प्रिन"), ("Flagyl", "فلیجل", "फ्लैजिल"), ("Risek", "رائزیک", "राइज़ेक"),
    ("Omeprazole", "اومیپرازول", "ओमेप्राज़ोल"), ("Metformin", "میٹفارمن", "मेटफॉर्मिन"), ("Amoxil", "اموکسل", "अमोक्सिल"),
    ("Ponstan", "پونسٹان", "पोंस्टान"), ("Calpol", "کیلپول", "कैलपोल"), ("Aspirin", "ایسپرین", "एस्पिरिन"),
    ("Insulin", "انسولین", "इंसुलिन"), ("Glucophage", "گلوکوفیج", "ग्लूकोफेज"), ("Concor", "کونکور", "कॉनकोर"),
    ("Lipitor", "لپیٹر", "लिपिटर"), ("Zyrtec", "زرٹیک", "ज़िरटेक"), ("Softin", "سافٹن", "सॉफ्टिन"),
    ("Arinac", "ارینیک", "एरिनैक"), ("Neurobion", "نیوروبیون", "न्यूरोबियॉन"), ("Surbex", "سربیکس", "सरबेक्स"),
    ("Caltrate", "کیلٹریٹ", "कैल्ट्रेट"), ("Nexium", "نیکسیم", "नेक्सियम"), ("Cipro", "سپرو", "सिप्रो"),
    ("Velosef", "ویلوسیف", "वेलोसेफ"), ("Klaricid", "کلیرسڈ", "क्लैरिसिड"), ("Losartan", "لوسارٹن", "लोसार्टन"),
    ("Amlodipine", "ایملوڈپین", "एम्लोडिपिन"), ("Atorvastatin", "اٹورواسٹیٹن", "एटोरवास्टेटिन"),
    ("Gaviscon", "گیوسکان", "गैविस्कॉन"), ("Tramal", "ٹرامل", "ट्रामल"), ("Xanax", "زینیکس", "ज़ैनेक्स"),
    ("Lexotanil", "لیکسوٹینل", "लेक्सोटैनिल"), ("Cetirizine", "سیٹیریزین", "सेटिरिज़ीन"),
    ("Paracetamol", "پیراسیٹامول", "पैरासिटामोल"), ("Vitamin D", "وٹامن ڈی", "विटामिन डी"),
]
HELD_OUT_DRUGS = [
    ("Ventolin", "وینٹولن", "वेंटोलिन"), ("Thyroxine", "تھائیروکسن", "थायरोक्सिन"), ("Mucaine", "میوکین", "म्यूकेन"),
    ("Nuberol", "نیوبرول", "न्यूबेरोल"), ("Motilium", "موٹیلیم", "मोटिलियम"), ("Entamizole", "اینٹامیزول", "एंटामिज़ोल"),
]

UR_NUM = "ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ".split()
HI_NUM = "एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह".split()
EN_NUM = "one two three four five six seven eight nine ten eleven twelve".split()
# (name, Urdu, Hindi, English, is_pm_for_hours)
PERIODS = [
    ("morning", "صبح", "सुबह", "am"), ("afternoon", "دوپہر", "दोपहर", "pm"),
    ("evening", "شام", "शाम", "pm"), ("night", "رات", "रात", "pm"),
]
UR_MIN = {5: "پانچ", 10: "دس", 15: "پندرہ", 20: "بیس", 25: "پچیس", 40: "چالیس", 50: "پچاس"}
HI_MIN = {5: "पाँच", 10: "दस", 15: "पंद्रह", 20: "बीस", 25: "पच्चीस", 40: "चालीस", 50: "पचास"}
EN_MIN = {5: "oh five", 10: "ten", 15: "fifteen", 20: "twenty", 25: "twenty five", 30: "thirty", 40: "forty", 45: "forty five", 50: "fifty"}

UR_T = ["{d} {p} {t} کھانی ہے", "مجھے {p} {t} {d} یاد دلانا", "{p} {t} {d} کی گولی", "{d} روزانہ {p} {t}",
        "{p} {t} مجھے {d} لینی ہے", "{d} کا الارم {p} {t} لگا دو"]
HI_T = ["{d} {p} {t} खानी है", "मुझे {p} {t} {d} याद दिलाना", "{p} {t} {d} की गोली", "{d} रोज़ाना {p} {t}",
        "{p} {t} मुझे {d} लेनी है", "{d} का अलार्म {p} {t} लगा दो"]
EN_T = ["take {d} at {t}", "remind me to take {d} at {t}", "{d} at {t} every day", "set an alarm for {d} at {t}",
        "i need my {d} tablet at {t}", "{d} tablet at {t}"]

EN_VOICES = ["Daniel", "Samantha", "Moira", "Karen", "Tessa", "Fred", "Ralph", "Kathy", "Junior",
             "Eddy (English (UK))", "Flo (English (US))", "Reed (English (UK))", "Sandy (English (US))", "Shelley (English (UK))"]
EN_TEST_VOICE = "Rishi"  # the South Asian English voice, withheld from training
HINDI_VOICE = ("say", "Lekha")
EN_ENGINE = "say"

if shutil.which("say") is None:
    # Not on a Mac (for example on Colab): use espeak-ng voices for everything.
    EN_VOICES = ["en-us", "en-gb", "en-gb-scotland", "en-gb-x-rp", "en-gb-x-gbcwmd", "en-029",
                 "en-us+f3", "en-us+m3", "en-gb+f4", "en-us+f2", "en-gb+m4"]
    EN_TEST_VOICE = "en-gb-x-gbclan"
    HINDI_VOICE = ("espeak", "hi")
    EN_ENGINE = "espeak"


def ur_hi_time(rng: random.Random, h: int) -> tuple[str, str, int]:
    """Return an Urdu phrase, a matching Hindi phrase, and the minute."""
    style = rng.choice(["plain", "plain", "half", "quarter_past", "quarter_to", "minutes"])
    if style == "half":
        if h == 1:
            return "ڈیڑھ بجے", "डेढ़ बजे", 30
        if h == 2:
            return "ڈھائی بجے", "ढाई बजे", 30
        return f"ساڑھے {UR_NUM[h-1]} بجے", f"साढ़े {HI_NUM[h-1]} बजे", 30
    if style == "quarter_past":
        return f"سوا {UR_NUM[h-1]} بجے", f"सवा {HI_NUM[h-1]} बजे", 15
    if style == "quarter_to":
        return f"پونے {UR_NUM[h-1]} بجے", f"पौने {HI_NUM[h-1]} बजे", 45
    if style == "minutes":
        m = rng.choice(list(UR_MIN))
        return f"{UR_NUM[h-1]} بج کر {UR_MIN[m]} منٹ", f"{HI_NUM[h-1]} बज कर {HI_MIN[m]} मिनट", m
    return f"{UR_NUM[h-1]} بجے", f"{HI_NUM[h-1]} बजे", 0


def make_items(rng: random.Random, n_each: int) -> list[dict]:
    items = []
    for held, drugs, count in [(False, DRUGS, n_each), (True, HELD_OUT_DRUGS, n_each // 8)]:
        for _ in range(count):
            d = rng.choice(drugs)
            hour = rng.randint(1, 12)
            period = rng.choice(PERIODS)
            ti = rng.randrange(len(UR_T))
            tu, th, minute = ur_hi_time(rng, hour)
            # Quarter-to names the coming hour, so the clock hour is one less.
            clock = hour - 1 or 12 if minute == 45 else hour
            ur = UR_T[ti].format(d=d[1], p=period[1], t=tu)
            hi = HI_T[ti].format(d=d[2], p=period[2], t=th)
            base = {"drug": d[0], "hour": clock, "minute": minute, "period": period[0], "held_out_drug": held}
            items.append({**base, "lang": "ur", "engine": "espeak", "voice": "ur", "say": ur, "label": ur, "label_voice": "ur"})
            items.append({**base, "lang": "ur", "engine": HINDI_VOICE[0], "voice": HINDI_VOICE[1], "say": hi, "label": ur, "label_voice": "ur"})

            en_hour = rng.randint(1, 12)
            en_min = rng.choice([0, 0, 0, *EN_MIN])
            ampm = rng.choice(["am", "pm"])
            en_time = f"{EN_NUM[en_hour-1]} {EN_MIN[en_min]} {ampm}" if en_min else f"{EN_NUM[en_hour-1]} {ampm}"
            en = rng.choice(EN_T).format(d=d[0].lower(), t=en_time)
            en_base = {"drug": d[0], "hour": en_hour, "minute": en_min, "period": ampm, "held_out_drug": held}
            voice = EN_TEST_VOICE if rng.random() < 0.08 else rng.choice(EN_VOICES)
            items.append({**en_base, "lang": "en", "engine": EN_ENGINE, "voice": voice, "say": en, "label": en, "label_voice": "en-us"})
    return items


def render(job: tuple[int, dict, random.Random]) -> dict | None:
    i, it, rng = job
    wav = OUT / f"cmd_{i:06d}.wav"
    if it["engine"] == "espeak":
        cmd = ["espeak-ng", "-v", it["voice"], "-s", str(rng.randint(120, 175)), "-p", str(rng.randint(30, 70)), "-w", str(wav), it["say"]]
    else:
        cmd = ["say", "-v", it["voice"], "-r", str(rng.randint(140, 205)), "--data-format=LEI16@16000", "-o", str(wav), it["say"]]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode or not wav.exists() or wav.stat().st_size < 2000:
        return None
    return {**it, "wav": wav.name}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)
    items = make_items(rng, 2200)
    jobs = [(i, it, random.Random(i)) for i, it in enumerate(items)]
    with ThreadPoolExecutor(8) as pool:
        done = [r for r in pool.map(render, jobs) if r]
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in done:
            r["split"] = "test" if r["held_out_drug"] or r["voice"] == EN_TEST_VOICE else "train"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"commands: {len(done)}/{len(items)} rendered")


if __name__ == "__main__":
    main()
