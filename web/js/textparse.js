// Understand a typed reminder in English, Roman Urdu or Urdu script: which tablet, what time.

const UR_SCRIPT_NUMS = ("ایک دو تین چار پانچ چھ سات آٹھ نو دس گیارہ بارہ تیرہ چودہ پندرہ سولہ سترہ اٹھارہ انیس بیس " +
  "اکیس بائیس تئیس چوبیس پچیس چھبیس ستائیس اٹھائیس انتیس تیس اکتیس بتیس تینتیس چونتیس پینتیس " +
  "چھتیس سینتیس اڑتیس انتالیس چالیس اکتالیس بیالیس تینتالیس چوالیس پینتالیس چھیالیس سینتالیس " +
  "اڑتالیس انچاس پچاس اکیاون باون ترپن چون پچپن چھپن ستاون اٹھاون انسٹھ").split(" ");

const ROMAN_NUMS = {
  ek: 1, aik: 1, ik: 1, do: 2, teen: 3, tin: 3, char: 4, chaar: 4, chār: 4, panch: 5, paanch: 5, panj: 5,
  chay: 6, chhe: 6, che: 6, chhay: 6, chhey: 6, chey: 6, cheh: 6, chhah: 6, chah: 6, saat: 7, sat: 7,
  aath: 8, ath: 8, aat: 8, aanth: 8, nau: 9, no: 9, now: 9, nao: 9, nou: 9, das: 10, dus: 10, dass: 10,
  gyarah: 11, gyara: 11, giyara: 11, gyaara: 11, gyarha: 11, giyarah: 11, barah: 12, bara: 12, baara: 12, baarah: 12, bärah: 12,
  pandrah: 15, pandra: 15, bees: 20, bis: 20, pachees: 25, pachis: 25, tees: 30, tis: 30, paintees: 35, pantees: 35,
  chalees: 40, chalis: 40, paintalees: 45, pantalis: 45, paintalis: 45, pachaas: 50, pachas: 50, pachpan: 55,
};
const EN_ONES = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(" ");
const EN_TENS = { twenty: 20, thirty: 30, forty: 40, fourty: 40, fifty: 50 };
const BAJE = new Set(["baje", "bje", "bjy", "bajay", "bajey", "bajy", "bajai", "bjey", "bajeh", "bajee", "بجے", "oclock", "o'clock"]);
const BAJ = new Set(["baj", "bj", "بج"]);
const KAR = new Set(["kar", "kr", "ke", "k", "کر", "کے"]);
const SAADE = new Set(["saade", "sade", "sadhe", "sarhe", "saarhe", "saray", "sarhay", "saadhe", "ساڑھے"]);
const SAWA = new Set(["sawa", "sava", "swa", "سوا"]);
const PAUNE = new Set(["paune", "pone", "pauney", "pony", "paunay", "پونے"]);
const DEDH = new Set(["dedh", "derh", "darh", "deedh", "ڈیڑھ"]);
const DHAI = new Set(["dhai", "dhaee", "dhaai", "arhai", "adhai", "dhayi", "ڈھائی"]);
const PERIODS = {
  morning: ["subah", "subha", "sobah", "subh", "saveray", "savere", "morning", "صبح"],
  afternoon: ["dopahar", "dopehar", "dupehar", "dopher", "dophar", "dupahar", "afternoon", "دوپہر"],
  evening: ["shaam", "sham", "evening", "شام"],
  night: ["raat", "rat", "night", "tonight", "رات"],
};

function numberAt(tokens, i) {
  // Returns { value, used } for a number starting at tokens[i], or null.
  const t = tokens[i];
  if (t === undefined) return null;
  if (/^\d{1,2}$/.test(t)) return { value: +t, used: 1 };
  if (t in ROMAN_NUMS) return { value: ROMAN_NUMS[t], used: 1 };
  const ui = UR_SCRIPT_NUMS.indexOf(t);
  if (ui >= 0) return { value: ui + 1, used: 1 };
  const ei = EN_ONES.indexOf(t);
  if (ei >= 0) return { value: ei + 1, used: 1 };
  if (t in EN_TENS) {
    const n = EN_ONES.indexOf(tokens[i + 1]);
    return n >= 0 && n < 9 ? { value: EN_TENS[t] + n + 1, used: 2 } : { value: EN_TENS[t], used: 1 };
  }
  if (t === "oh" && EN_ONES.indexOf(tokens[i + 1]) >= 0 && EN_ONES.indexOf(tokens[i + 1]) < 9) return { value: EN_ONES.indexOf(tokens[i + 1]) + 1, used: 2 };
  return null;
}

const isHour = (n) => n && n.value >= 1 && n.value <= 12;

function findTime(tokens) {
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    let m;
    // 9:30, 9.30, 9:30pm, 9pm
    if ((m = t.match(/^(\d{1,2})(?:[:.](\d{2}))?(am|pm)?$/))) {
      const h = +m[1], min = m[2] ? +m[2] : 0;
      const ap = m[3] || (["am", "pm"].includes(tokens[i + 1]) ? tokens[i + 1] : null);
      const anchored = m[2] || ap || BAJE.has(tokens[i + 1]) || ["at", "@"].includes(tokens[i - 1]);
      if (h <= 23 && min < 60 && anchored) return h > 12 ? { hour: h - 12, minute: min, period: "pm" } : { hour: h === 0 ? 12 : h, minute: min, period: ap || (h === 0 ? "am" : null) };
      continue;
    }
    if (DEDH.has(t)) return { hour: 1, minute: 30 };
    if (DHAI.has(t)) return { hour: 2, minute: 30 };
    if (SAADE.has(t) || SAWA.has(t) || PAUNE.has(t)) {
      const n = numberAt(tokens, i + 1);
      if (isHour(n)) {
        if (SAADE.has(t)) return { hour: n.value, minute: 30 };
        if (SAWA.has(t)) return { hour: n.value, minute: 15 };
        return { hour: n.value === 1 ? 12 : n.value - 1, minute: 45 };
      }
    }
    if (t === "noon") return { hour: 12, minute: 0, period: "pm" };
    if (t === "midnight") return { hour: 12, minute: 0, period: "am" };
    if ((t === "half" || t === "quarter") && (tokens[i + 1] === "past" || tokens[i + 1] === "to")) {
      const n = numberAt(tokens, i + 2);
      if (isHour(n)) {
        if (t === "half") return { hour: n.value, minute: 30 };
        if (tokens[i + 1] === "past") return { hour: n.value, minute: 15 };
        return { hour: n.value === 1 ? 12 : n.value - 1, minute: 45 };
      }
    }
    const n = numberAt(tokens, i);
    if (!isHour(n)) continue;
    const j = i + n.used;
    // "aath baje", "8 baje", "nine o'clock"
    if (BAJE.has(tokens[j])) return { hour: n.value, minute: 0 };
    // "aath baj kar bees minute", "8 bajkar 20"
    const bajkar = BAJ.has(tokens[j]) && KAR.has(tokens[j + 1]) ? 2 : ["bajkar", "bajke", "bjkr", "bajkr"].includes(tokens[j]) ? 1 : 0;
    if (bajkar) {
      const mn = numberAt(tokens, j + bajkar);
      return { hour: n.value, minute: mn && mn.value < 60 ? mn.value : 0 };
    }
    // English words: "at nine", "nine thirty pm", "nine forty five"
    const isEnglish = EN_ONES.includes(tokens[i]);
    if (isEnglish) {
      const mn = numberAt(tokens, j);
      const minute = mn && mn.value < 60 && (EN_ONES.includes(tokens[j]) || tokens[j] in EN_TENS || tokens[j] === "oh") ? mn.value : 0;
      const k = j + (minute ? mn.used : 0);
      const ap = ["am", "pm"].includes(tokens[k]) ? tokens[k] : null;
      if (ap || tokens[i - 1] === "at" || minute) return { hour: n.value, minute, period: ap };
    }
  }
  return null;
}

function findPeriod(tokens) {
  for (const t of tokens) {
    if (t === "am" || t === "pm") return t;
    for (const [p, words] of Object.entries(PERIODS)) if (words.includes(t)) return p;
  }
  return null;
}

function editDistance(a, b) {
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur.push(Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] !== b[j - 1])));
    prev = cur;
  }
  return prev[b.length];
}

function findTablet(tokens, tablets) {
  // tablets: [{ latin, urdu }]; compares single words and pairs ("vitamin d").
  let best = null;
  const grams = [];
  for (let i = 0; i < tokens.length; i++) {
    grams.push(tokens[i]);
    if (i + 1 < tokens.length) grams.push(tokens[i] + " " + tokens[i + 1]);
  }
  for (const tab of tablets) {
    for (const name of [tab.latin.toLowerCase(), tab.urdu].filter(Boolean)) {
      for (const g of grams) {
        if (g.length < 3) continue;
        const d = editDistance(g, name) / name.length;
        const limit = name.length <= 4 ? 0 : 0.25;
        if (d <= limit && (!best || d < best.d)) best = { d, name: tab.latin };
      }
    }
  }
  return best ? best.name : null;
}

/** Returns { tablet, hour, minute, period } with null where something was not found. */
export function parseText(text, tablets) {
  const tokens = text
    .toLowerCase()
    .replace(/o'\s?clock/g, "oclock")
    .replace(/(\d)\s*(a\.?m\.?|p\.?m\.?)(?=\s|$)/g, (_, d, ap) => `${d} ${ap.replace(/\./g, "")}`)
    .replace(/\b(a|p)\.m\./g, "$1m")
    .replace(/[،۔,!?؟"()]/g, " ")
    .replace(/(\D)[.](?=\s|$)/g, "$1 ")
    .split(/\s+/)
    .filter(Boolean);
  const time = findTime(tokens);
  return {
    tablet: findTablet(tokens, tablets),
    hour: time ? time.hour : null,
    minute: time ? time.minute : null,
    period: (time && time.period) || findPeriod(tokens),
  };
}

/** 24-hour clock from hour 1-12 plus am/pm or an Urdu period. null when it is ambiguous. */
export function to24(hour, period) {
  if (hour == null) return null;
  const h = hour % 12;
  switch (period) {
    case "am": return h;
    case "pm": return h + 12;
    case "morning": return h;
    case "afternoon": return hour === 12 ? 12 : hour >= 11 ? hour : h + 12;
    case "evening": return h + 12;
    case "night": return hour === 12 ? 0 : hour <= 4 ? hour : h + 12;
    default: return null;
  }
}
