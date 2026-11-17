// Audio front end, matching src/features.py and the trimming in src/transcribe.py.

export class Features {
  constructor(assets) {
    this.nFft = assets.n_fft;
    this.hop = assets.hop;
    this.nMels = assets.n_mels;
    this.nBins = this.nFft / 2 + 1;
    this.mel = new Float64Array(this.nMels * this.nBins);
    assets.mel.forEach((row, m) => row.forEach((v, b) => (this.mel[m * this.nBins + b] = v)));
    const N = this.nFft;
    this.window = new Float64Array(N);
    for (let n = 0; n < N; n++) this.window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / (N - 1)); // np.hanning
    this.cos = new Float64Array(this.nBins * N);
    this.sin = new Float64Array(this.nBins * N);
    for (let k = 0; k < this.nBins; k++)
      for (let n = 0; n < N; n++) {
        const a = (2 * Math.PI * k * n) / N;
        this.cos[k * N + n] = Math.cos(a);
        this.sin[k * N + n] = Math.sin(a);
      }
  }

  /** Normalized log-mels for 16 kHz mono audio: { data: Float32Array(frames * 80), frames }. */
  compute(wav) {
    const { nFft: N, hop, nMels, nBins } = this;
    if (wav.length < N) {
      const p = new Float32Array(N);
      p.set(wav);
      wav = p;
    }
    const frames = 1 + Math.floor((wav.length - N) / hop);
    const out = new Float64Array(frames * nMels);
    const x = new Float64Array(N);
    const power = new Float64Array(nBins);
    for (let f = 0; f < frames; f++) {
      const o = f * hop;
      for (let n = 0; n < N; n++) x[n] = wav[o + n] * this.window[n];
      for (let k = 0; k < nBins; k++) {
        let re = 0, im = 0;
        const base = k * N;
        for (let n = 0; n < N; n++) {
          re += x[n] * this.cos[base + n];
          im -= x[n] * this.sin[base + n];
        }
        power[k] = re * re + im * im;
      }
      for (let m = 0; m < nMels; m++) {
        let s = 0;
        const base = m * nBins;
        for (let k = 0; k < nBins; k++) s += power[k] * this.mel[base + k];
        out[f * nMels + m] = Math.log(Math.max(s, 1e-10));
      }
    }
    // Zero mean, unit variance per band over the utterance.
    const data = new Float32Array(frames * nMels);
    for (let m = 0; m < nMels; m++) {
      let mean = 0;
      for (let f = 0; f < frames; f++) mean += out[f * nMels + m];
      mean /= frames;
      let v = 0;
      for (let f = 0; f < frames; f++) v += (out[f * nMels + m] - mean) ** 2;
      const sd = Math.sqrt(v / frames) + 1e-5;
      for (let f = 0; f < frames; f++) data[f * nMels + m] = (out[f * nMels + m] - mean) / sd;
    }
    return { data, frames };
  }
}

/** Same as librosa.effects.split(top_db=40, frame_length=400, hop_length=160) plus a 0.1 s margin. */
export function trimSilence(wav, frameLength = 400, hop = 160, topDb = 40, margin = 1600) {
  const half = frameLength / 2;
  const nFrames = 1 + Math.floor(wav.length / hop);
  const mse = new Float64Array(nFrames);
  let max = 0;
  for (let i = 0; i < nFrames; i++) {
    let s = 0;
    const start = i * hop - half;
    for (let n = 0; n < frameLength; n++) {
      const j = start + n;
      if (j >= 0 && j < wav.length) s += wav[j] * wav[j];
    }
    mse[i] = s / frameLength;
    if (mse[i] > max) max = mse[i];
  }
  const ref = 10 * Math.log10(Math.max(1e-10, max));
  let first = -1, last = -1;
  for (let i = 0; i < nFrames; i++) {
    if (10 * Math.log10(Math.max(1e-10, mse[i])) - ref > -topDb) {
      if (first < 0) first = i;
      last = i;
    }
  }
  if (first < 0) return wav;
  const start = Math.min(first * hop, wav.length);
  const end = Math.min((last + 1) * hop, wav.length);
  return wav.slice(Math.max(0, start - margin), end + margin);
}

/** Browser only: decode any audio file or recording to 16 kHz mono. */
export async function decodeTo16k(arrayBuffer, sr = 16000) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
  ctx.close();
  const length = Math.max(1, Math.ceil(decoded.duration * sr));
  const off = new OfflineAudioContext(1, length, sr);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0).slice();
}

export function encodeWav(samples, sr = 16000) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, "data"); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true);
  return new Blob([buf], { type: "audio/wav" });
}
