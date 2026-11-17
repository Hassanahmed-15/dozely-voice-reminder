// Restricted decoder, matching src/decoder.py: which known tablet and which time phrase fit best.

const NEG = -1e9;

export class Spotter {
  constructor(cands, spaceId) {
    this.cands = cands;
    const K = cands.length;
    let L = 0;
    for (const c of cands) L = Math.max(L, c.ids.length);
    const S = 2 * L + 1;
    Object.assign(this, { K, S });
    this.lab = new Int32Array(K * S);
    this.valid = new Uint8Array(K * S);
    this.skip = new Uint8Array(K * S);
    this.end = new Int32Array(K);
    this.nPh = new Float64Array(K);
    cands.forEach((c, j) => {
      const lab = [0];
      for (const i of c.ids) lab.push(i, 0);
      for (let q = 0; q < lab.length; q++) {
        this.lab[j * S + q] = lab[q];
        this.valid[j * S + q] = 1;
        if (q >= 2) this.skip[j * S + q] = lab[q] !== 0 && lab[q] !== lab[q - 2] ? 1 : 0;
      }
      this.end[j] = lab.length - 1;
      this.nPh[j] = c.ids.filter((i) => i !== spaceId).length;
    });
  }

  /** lp: Float32Array(T * V) log probabilities. Returns the fit of each candidate (<= 0). */
  score(lp, T, V) {
    const { K, S, lab, valid, skip, end } = this;
    const best = new Float64Array(T);
    let bestSum = 0;
    for (let t = 0; t < T; t++) {
      let m = -Infinity;
      for (let v = 0; v < V; v++) if (lp[t * V + v] > m) m = lp[t * V + v];
      best[t] = m;
      bestSum += m;
    }
    let alpha = new Float64Array(K * S).fill(NEG);
    let next = new Float64Array(K * S);
    const post = new Float64Array(K).fill(NEG);
    for (let j = 0; j < K; j++) {
      const b = j * S;
      alpha[b] = lp[lab[b]];
      alpha[b + 1] = valid[b + 1] ? lp[lab[b + 1]] : NEG;
    }
    let pre = best[0];
    for (let t = 1; t < T; t++) {
      const row = t * V;
      for (let j = 0; j < K; j++) {
        const b = j * S, e = end[j];
        const fin = Math.max(alpha[b + e], alpha[b + e - 1]);
        post[j] = Math.max(post[j], fin) + best[t];
        for (let q = 0; q < S; q++) {
          if (!valid[b + q]) { next[b + q] = NEG; continue; }
          let a = alpha[b + q];
          if (q >= 1 && alpha[b + q - 1] > a) a = alpha[b + q - 1];
          if (q >= 2 && skip[b + q] && alpha[b + q - 2] > a) a = alpha[b + q - 2];
          if (q <= 1 && pre > a) a = pre;
          next[b + q] = a + lp[row + lab[b + q]];
        }
      }
      pre += best[t];
      [alpha, next] = [next, alpha];
    }
    const out = new Float64Array(K);
    for (let j = 0; j < K; j++) {
      const b = j * S, e = end[j];
      out[j] = Math.max(post[j], Math.max(alpha[b + e], alpha[b + e - 1])) - bestSum;
    }
    return out;
  }

  best(lp, T, V, bonus = 0) {
    const raw = this.score(lp, T, V);
    let bi = 0, bt = -Infinity;
    for (let j = 0; j < this.K; j++) {
      const tot = raw[j] + bonus * this.nPh[j];
      if (tot > bt) { bt = tot; bi = j; }
    }
    return { cand: this.cands[bi], raw: raw[bi] };
  }
}

export class Understander {
  constructor(assets) {
    this.a = assets;
    const sp = assets.space_id;
    this.tablets = new Spotter(assets.tablets, sp);
    this.times = new Spotter(assets.times, sp);
    this.urduPeriods = new Spotter(assets.urdu_periods, sp);
  }

  period(lp, T, V, time) {
    if (time.period) return time.period;
    if (time.lang === "noon") return "pm";
    if (time.lang === "midnight") return "am";
    if (time.lang === "en") {
      const pair = new Spotter([{ label: "am", ids: time.ampm.am }, { label: "pm", ids: time.ampm.pm }], this.a.space_id);
      const s = pair.score(lp, T, V);
      return s[0] >= s[1] ? "am" : "pm";
    }
    return this.urduPeriods.best(lp, T, V).cand.label;
  }

  understand(lp, T, V) {
    const tab = this.tablets.best(lp, T, V, this.a.tablet_bonus);
    const tim = this.times.best(lp, T, V, this.a.time_bonus);
    const u = {
      tablet: tab.cand.label,
      tablet_fit: tab.raw / Math.max(1, tab.cand.ids.length),
      hour: tim.cand.label[0],
      minute: tim.cand.label[1],
      time_phrase: tim.cand.text,
      time_fit: tim.raw / Math.max(1, tim.cand.ids.length),
      period: this.period(lp, T, V, tim.cand),
    };
    u.tablet_found = u.tablet_fit >= this.a.thresholds.tablet;
    u.time_found = u.time_fit >= this.a.thresholds.time;
    u.heard = this.heard(lp, T, V);
    return u;
  }

  /** Free (unrestricted) reading of the sounds, shown for information. */
  heard(lp, T, V) {
    let prev = 0, s = "";
    for (let t = 0; t < T; t++) {
      let bi = 0, bm = -Infinity;
      for (let v = 0; v < V; v++) if (lp[t * V + v] > bm) { bm = lp[t * V + v]; bi = v; }
      if (bi !== prev && bi !== 0) {
        const tok = this.a.vocab[bi];
        s += this.a.roman[tok] ?? tok;
      }
      prev = bi;
    }
    return s.replace(/\s+/g, " ").trim();
  }
}
