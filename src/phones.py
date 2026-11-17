"""Text to phoneme tokens, shared by Urdu and English.

Both languages go through espeak-ng's rule-based converter and are then folded into
one small phone set. The model therefore has a single output alphabet, so mixed
speech such as "Panadol subah aath baje" has one consistent transcription, and a new
tablet name can be recognised just by converting its spelling to phones.
"""
import re
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor

BLANK = "<b>"
SPACE = "|"

# Fold espeak's detailed IPA into coarser classes. Distinctions that vary with accent
# (vowel length, retroflex r, the several schwa-like vowels) carry little information
# for picking a tablet name and a time, and merging them makes training easier.
FOLD = {
    "ɐ": "ə", "ʌ": "ə", "ɜ": "ə", "ᵻ": "ɪ", "ɨ": "ɪ", "y": "i",
    "ɑ": "a", "æ": "a", "ɒ": "ɔ", "o": "ɔ", "e": "ɛ",
    "ɾ": "r", "ɹ": "r", "ɽ": "r", "ɻ": "r", "ʁ": "r", "ɚ": "ər", "ɝ": "ər",
    "ɡ": "g", "ç": "h", "x": "h", "ɦ": "h", "ʔ": "",
    "ʋ": "v", "ɬ": "l", "ɫ": "l", "ɭ": "l",
    "ɳ": "n", "ɲ": "n", "ŋ": "n",
    "ʂ": "s", "ʒ": "z", "ɣ": "g", "χ": "h", "ħ": "h", "ʕ": "", "q": "k",
    "ʈ": "t", "ɖ": "d", "ð": "d", "θ": "t",
    "c": "ʧ", "ɟ": "ʤ",
    "ʰ": "h", "ʱ": "h", "ʐ": "z",
    # Nasal vowels compose into single characters under NFC, so they must be folded here.
    "ẽ": "ɛ", "õ": "ɔ", "ã": "a", "ũ": "u", "ĩ": "i",
}
MARKS = re.compile(r"\([a-z-]+\)|[ˈˌːˑ‿̩̪̥̬̯̃͡ʲˠˤ\-_.,;:!?'\"]")

VOCAB = [BLANK, SPACE] + list("abdfghijklmnprstuvwzɔəɛɪʃʊʧʤ")
TOKEN_ID = {t: i for i, t in enumerate(VOCAB)}


def espeak(text: str, voice: str) -> str:
    # Text goes in as an argument: with --stdin, espeak-ng reads past the end of
    # non-ASCII input and appends spurious syllables.
    text = " ".join(text.split()).lstrip("-") or " "
    try:
        out = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, text], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        return ""  # an empty result is dropped by the caller
    return " ".join(out.stdout.decode("utf-8", "ignore").split())


def fold(ipa: str, unknown: dict | None = None) -> list[str]:
    ipa = MARKS.sub("", unicodedata.normalize("NFC", ipa))
    ipa = ipa.replace("dʒ", "ʤ").replace("tʃ", "ʧ")
    tokens: list[str] = []
    for ch in ipa:
        if ch == " ":
            if tokens and tokens[-1] != SPACE:
                tokens.append(SPACE)
            continue
        for mapped in FOLD.get(ch, ch):
            if mapped in TOKEN_ID:
                tokens.append(mapped)
            elif unknown is not None:
                unknown[mapped] = unknown.get(mapped, 0) + 1
    while tokens and tokens[-1] == SPACE:
        tokens.pop()
    return tokens


def to_tokens(text: str, voice: str, unknown: dict | None = None) -> list[str]:
    return fold(espeak(text, voice), unknown)


def to_tokens_many(items: list[tuple[str, str]], workers: int = 10, unknown: dict | None = None) -> list[list[str]]:
    """Convert (text, voice) pairs, calling espeak once per distinct pair."""
    distinct = sorted(set(items))
    with ThreadPoolExecutor(workers) as pool:
        ipa = dict(zip(distinct, pool.map(lambda p: espeak(p[0], p[1]), distinct)))
    return [fold(ipa[i], unknown) for i in items]
