import { Features, trimSilence, decodeTo16k, encodeWav } from "./audio.js";
import { Understander } from "./decoder.js";
import { parseText, to24 } from "./textparse.js";

const $ = (id) => document.getElementById(id);
const store = {
  get(key, fallback) { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} },
};
const pad = (n) => String(n).padStart(2, "0");
const tick = () => new Promise((r) => setTimeout(r, 30));
const PERIOD_WORD = { morning: "subah", afternoon: "dopahar", evening: "shaam", night: "raat", am: "AM", pm: "PM" };

let assets, features, understander, session;
let reminders = store.get("dozely.reminders", []);
let pending = null;

// ---------- model ----------
async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
    ort.env.wasm.numThreads = 1;
    assets = await (await fetch("assets.json")).json();
    features = new Features(assets);
    understander = new Understander(assets);
    session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
    $("model-pill").textContent = "Model ready";
    $("model-pill").classList.add("ready");
    $("mic").disabled = false;
    $("tablet-list").innerHTML = allTablets().map((t) => `<option value="${t.latin}">`).join("");
  } catch (e) {
    $("model-pill").textContent = "Model failed to load";
    status("Could not load the speech model: " + e.message, true);
  }
}

const allTablets = () => [...assets.tablet_names, ...store.get("dozely.myTablets", []).map((n) => ({ latin: n, urdu: "" }))];

async function understandAudio(wav) {
  status("Understanding…");
  await tick();
  const t0 = performance.now();
  const trimmed = trimSilence(wav);
  const { data, frames } = features.compute(trimmed);
  const out = await session.run({ mel: new ort.Tensor("float32", data, [1, frames, assets.n_mels]) });
  const lp = out.log_probs;
  const u = understander.understand(lp.data, lp.dims[1], lp.dims[2]);
  u.ms = Math.round(performance.now() - t0);
  u.seconds = (trimmed.length / assets.sr).toFixed(1);
  return u;
}

// ---------- input ----------
function status(msg, bad = false) {
  $("status").textContent = msg;
  $("status").style.color = bad ? "var(--bad)" : "";
}

let recorder = null, chunks = [], stopTimer = null;
$("mic").addEventListener("click", async () => {
  unlockSound();
  if (recorder && recorder.state === "recording") return recorder.stop();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: true } });
    chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    recorder.onstop = async () => {
      clearTimeout(stopTimer);
      stream.getTracks().forEach((t) => t.stop());
      $("mic").classList.remove("recording");
      $("mic-hint").textContent = "Tap the mic, speak, then tap again to stop.";
      const blob = new Blob(chunks, { type: recorder.mimeType });
      await handleAudio(await blob.arrayBuffer());
    };
    recorder.start();
    $("mic").classList.add("recording");
    $("mic-hint").textContent = "Listening… tap again when you finish (stops by itself after 12 seconds).";
    status("");
    stopTimer = setTimeout(() => recorder.state === "recording" && recorder.stop(), 12000);
  } catch (e) {
    status("Microphone not available: " + e.message, true);
  }
});

$("file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) await handleAudio(await f.arrayBuffer());
  e.target.value = "";
});

async function handleAudio(buf) {
  try {
    const wav = await decodeTo16k(buf, assets.sr);
    if (wav.length < assets.sr * 0.3) return status("That recording was too short. Try again.", true);
    const u = await understandAudio(wav);
    const time = u.time_found ? { hour: u.hour, minute: u.minute, period: u.period } : null;
    showConfirm({ source: "voice", tablet: u.tablet_found ? u.tablet : null, time, u, wav });
    status(`Done in ${u.ms} ms.`);
  } catch (e) {
    status("Could not read that audio: " + e.message, true);
  }
}

$("parse").addEventListener("click", () => {
  unlockSound();
  const text = $("text").value.trim();
  if (!text) return status("Type a reminder first.", true);
  const r = parseText(text, allTablets());
  showConfirm({ source: "text", tablet: r.tablet, time: r.hour == null ? null : { hour: r.hour, minute: r.minute, period: r.period }, text });
  status("");
});
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("parse").click(); } });
document.querySelectorAll(".examples span").forEach((s) => s.addEventListener("click", () => { $("text").value = s.textContent; $("parse").click(); }));

// ---------- confirm ----------
function showConfirm(p) {
  pending = p;
  $("confirm").classList.add("show");
  $("c-tablet").value = p.tablet ?? "";
  flag("c-tablet-flag", p.tablet ? ["ok", "Recognized"] : ["warn", p.source === "voice" ? `Not sure (closest: ${p.u.tablet}). Please type it.` : "No tablet found. Please type it."]);
  $("ampm").classList.remove("show");
  $("c-time").value = "";
  if (p.time) {
    const h24 = to24(p.time.hour, p.time.period);
    const said = `${p.time.hour}:${pad(p.time.minute)}${p.time.period ? " " + PERIOD_WORD[p.time.period] : ""}`;
    if (h24 == null) {
      $("c-time").value = `${pad(p.time.hour % 12)}:${pad(p.time.minute)}`;
      $("ampm").classList.add("show");
      flag("c-time-flag", ["warn", `Heard ${said}. AM or PM?`]);
    } else {
      $("c-time").value = `${pad(h24)}:${pad(p.time.minute)}`;
      flag("c-time-flag", ["ok", `Heard ${said}`]);
    }
  } else {
    flag("c-time-flag", ["warn", "No time found. Please set it."]);
  }
  const dl = $("download");
  if (p.wav) {
    dl.href = URL.createObjectURL(encodeWav(p.wav, assets.sr));
    dl.download = `dozely_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`;
    dl.style.display = "";
  } else dl.style.display = "none";
  $("details").style.display = p.u ? "" : "none";
  if (p.u) {
    const u = p.u;
    $("details-body").innerHTML =
      `<p>Free reading of the sounds (${u.seconds} s): <code dir="auto">${escapeHtml(u.heard) || "(nothing)"}</code></p>` +
      `<p>Best tablet match: <b>${u.tablet}</b>, fit ${u.tablet_fit.toFixed(2)} (needs ${assets.thresholds.tablet.toFixed(2)} or higher)<br>` +
      `Best time match: <b>${escapeHtml(u.time_phrase)}</b> → ${u.hour}:${pad(u.minute)} ${PERIOD_WORD[u.period] ?? ""}, fit ${u.time_fit.toFixed(2)} (needs ${assets.thresholds.time.toFixed(2)} or higher)</p>` +
      `<p>A fit near 0 means the phrase matches the audio cleanly.</p>`;
  }
  $("confirm").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function flag(id, [cls, msg]) {
  $(id).className = "flag " + cls;
  $(id).textContent = msg;
}
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

document.querySelectorAll("#ampm button").forEach((b) =>
  b.addEventListener("click", () => {
    const [h, m] = $("c-time").value.split(":").map(Number);
    const h12 = h % 12;
    $("c-time").value = `${pad(b.dataset.ap === "pm" ? h12 + 12 : h12)}:${pad(m)}`;
    $("ampm").classList.remove("show");
    flag("c-time-flag", ["ok", `Set to ${b.dataset.ap.toUpperCase()}`]);
  }),
);

$("save").addEventListener("click", () => {
  const tablet = $("c-tablet").value.trim();
  const time = $("c-time").value;
  if (!tablet) return flag("c-tablet-flag", ["warn", "Please enter the tablet name."]);
  if (!time) return flag("c-time-flag", ["warn", "Please set the time."]);
  if ($("ampm").classList.contains("show")) return flag("c-time-flag", ["warn", "Please choose AM or PM."]);
  const [h, m] = time.split(":").map(Number);
  if (!allTablets().some((t) => t.latin.toLowerCase() === tablet.toLowerCase())) {
    store.set("dozely.myTablets", [...store.get("dozely.myTablets", []), tablet]);
  }
  reminders.push({ id: Date.now(), tablet, h, m, lastFired: null, snoozeUntil: null });
  saveReminders();
  closeConfirm();
  status(`Saved: ${tablet} at ${fmt(h, m)}.`);
});
$("cancel").addEventListener("click", closeConfirm);
function closeConfirm() { $("confirm").classList.remove("show"); pending = null; }

// ---------- reminders ----------
const fmt = (h, m) => `${(h % 12) || 12}:${pad(m)} ${h < 12 ? "AM" : "PM"}`;
function saveReminders() { store.set("dozely.reminders", reminders); render(); }

function render() {
  const sorted = [...reminders].sort((a, b) => a.h * 60 + a.m - (b.h * 60 + b.m));
  $("empty").style.display = sorted.length ? "none" : "";
  $("list").innerHTML = sorted.map((r) => `
    <li>
      <div class="time-big">${fmt(r.h, r.m)}</div>
      <div class="grow"><div dir="auto"><b>${escapeHtml(r.tablet)}</b></div><div class="sub">${r.snoozeUntil ? "Snoozed until " + new Date(r.snoozeUntil).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "Every day"}</div></div>
      <button class="btn small" data-test="${r.id}">Test</button>
      <button class="btn small" data-del="${r.id}" aria-label="Delete">✕</button>
    </li>`).join("");
  $("list").querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", () => { reminders = reminders.filter((r) => r.id != b.dataset.del); saveReminders(); }));
  $("list").querySelectorAll("[data-test]").forEach((b) => b.addEventListener("click", () => { unlockSound(); fire(reminders.find((r) => r.id == b.dataset.test), true); }));
}

const todayKey = (d) => `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`;
setInterval(() => {
  const now = new Date();
  for (const r of reminders) {
    const due = now.getHours() === r.h && now.getMinutes() === r.m && r.lastFired !== todayKey(now);
    const snoozeDue = r.snoozeUntil && Date.now() >= r.snoozeUntil;
    if ((due || snoozeDue) && !active) {
      if (due) r.lastFired = todayKey(now);
      r.snoozeUntil = null;
      saveReminders();
      fire(r);
    }
  }
}, 1000);

// ---------- alarm ----------
let audioCtx = null, active = null, beepTimer = null, stopAfter = null;
function unlockSound() {
  if (!audioCtx) {
    try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch {}
  }
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
  $("sound-banner").classList.remove("show");
}
$("sound-banner").addEventListener("click", unlockSound);

function beep() {
  if (!audioCtx) return;
  const t = audioCtx.currentTime;
  [[880, 0], [1175, 0.18], [880, 0.36], [1175, 0.54]].forEach(([f, dt]) => {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = "sine"; o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t + dt);
    g.gain.exponentialRampToValueAtTime(0.5, t + dt + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dt + 0.16);
    o.connect(g).connect(audioCtx.destination);
    o.start(t + dt); o.stop(t + dt + 0.18);
  });
  navigator.vibrate?.([200, 100, 200]);
}

function fire(r, test = false) {
  if (!r) return;
  active = { r, test };
  $("alarm-title").textContent = r.tablet;
  $("alarm-sub").textContent = `${fmt(r.h, r.m)} · Time to take your tablet${test ? " (test)" : ""}`;
  $("overlay").classList.add("show");
  beep();
  beepTimer = setInterval(beep, 1500);
  stopAfter = setTimeout(stopAlarm, 2 * 60 * 1000);
  document.title = `⏰ ${r.tablet}`;
  if ("Notification" in window && Notification.permission === "granted") {
    try { new Notification("Dozely", { body: `Time to take ${r.tablet}`, tag: "dozely-" + r.id, requireInteraction: true }); } catch {}
  }
}
function stopAlarm() {
  clearInterval(beepTimer); clearTimeout(stopAfter);
  $("overlay").classList.remove("show");
  document.title = "Dozely Voice Reminders";
  active = null;
}
$("taken").addEventListener("click", stopAlarm);
$("snooze").addEventListener("click", () => {
  if (active && !active.test) { active.r.snoozeUntil = Date.now() + 10 * 60 * 1000; saveReminders(); }
  stopAlarm();
});

// ---------- settings ----------
$("notify").addEventListener("click", async () => {
  if (!("Notification" in window)) return status("This browser does not support notifications.", true);
  const p = await Notification.requestPermission();
  $("notify").textContent = p === "granted" ? "Notifications on" : "Notifications blocked";
});
if ("Notification" in window && Notification.permission === "granted") $("notify").textContent = "Notifications on";

let wakeLock = null;
$("wake").addEventListener("click", async () => {
  try {
    if (wakeLock) { await wakeLock.release(); wakeLock = null; $("wake").textContent = "Keep screen on"; return; }
    wakeLock = await navigator.wakeLock.request("screen");
    $("wake").textContent = "Screen stays on";
    wakeLock.addEventListener("release", () => { wakeLock = null; $("wake").textContent = "Keep screen on"; });
  } catch { status("This browser cannot keep the screen on.", true); }
});

render();
loadModel();
