"""Export everything the browser page needs into web/:
  model.onnx   the trained model
  assets.json  mel filterbank, phone list, decoder phrases (with phones), cut-offs, tablets
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

import decoder
import features as fx
import phones
import transcribe as ours
from generate_commands import DRUGS, HELD_OUT_DRUGS

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):  # (1, frames, 80) normalized log-mels
        # One utterance at a time, so there is no padding and no mask is needed.
        m = self.model
        x = m.sub.proj(m.sub.conv(mel.unsqueeze(1)).permute(0, 2, 1, 3).flatten(2))
        x = x + m.pos[:, : x.size(1)]
        for b in m.blocks:
            x = x + 0.5 * b.ff1(x)
            x = x + attention(b.attn, b.attn_norm(x))
            x = x + b.conv(x)
            x = x + 0.5 * b.ff2(x)
            x = b.norm(x)
        return torch.log_softmax(m.out(x), -1)


def attention(mha, y):
    """nn.MultiheadAttention written out with the same weights, using -1 for the time axis so
    the exported model accepts recordings of any length."""
    h = mha.num_heads
    d = mha.embed_dim // h
    q, k, v = torch.nn.functional.linear(y, mha.in_proj_weight, mha.in_proj_bias).chunk(3, -1)
    q, k, v = (t.reshape(1, -1, h, d).transpose(1, 2) for t in (q, k, v))
    w = torch.softmax(q @ k.transpose(-1, -2) / d ** 0.5, -1)
    out = (w @ v).transpose(1, 2).reshape(1, -1, h * d)
    return mha.out_proj(out)


def export_model():
    model = ours.load_model()
    w = Wrapper(model).eval()
    x = torch.randn(1, 300, fx.N_MELS)
    torch.onnx.export(w, (x,), str(WEB / "model.onnx"), input_names=["mel"], output_names=["log_probs"],
                      dynamic_axes={"mel": {1: "frames"}, "log_probs": {1: "steps"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(WEB / "model.onnx"))
    for T in (57, 300, 911):
        x = torch.randn(1, T, fx.N_MELS)
        ref = model(x, torch.tensor([T]))[0].detach().numpy()  # the original model, not the wrapper
        got = sess.run(None, {"mel": x.numpy()})[0]
        print(f"onnx check T={T}: max difference {np.abs(ref - got).max():.2e}")


def cand_json(c):
    return {"label": c.label, "text": c.text, "ids": c.ids, "lang": c.lang, "period": c.period}


def export_assets():
    und = decoder.Understander()
    times = []
    for c in und.times.candidates:
        d = cand_json(c)
        if c.lang == "en" and not c.period:  # the am/pm pair decide_period would build
            d["ampm"] = {p: decoder._ids(f"{c.text} {p[0]} m", "en-us") for p in ("am", "pm")}
        times.append(d)
    assets = {
        "sr": fx.SR, "n_fft": fx.N_FFT, "hop": fx.HOP, "n_mels": fx.N_MELS,
        "mel": np.round(fx._MEL, 7).tolist(),
        "vocab": phones.VOCAB, "space_id": phones.TOKEN_ID[phones.SPACE],
        "roman": ours.ROMAN,
        "tablet_bonus": decoder.TABLET_BONUS, "time_bonus": decoder.TIME_BONUS,
        "thresholds": decoder.load_thresholds(),
        "tablets": [cand_json(c) for c in und.tablets.candidates],
        "tablet_names": [{"latin": l, "urdu": u} for l, u, _ in DRUGS + HELD_OUT_DRUGS],
        "times": times,
        "urdu_periods": [cand_json(c) for c in und.urdu_periods.candidates],
    }
    (WEB / "assets.json").write_text(json.dumps(assets, ensure_ascii=False, separators=(",", ":")))
    print("assets.json", round((WEB / "assets.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    WEB.mkdir(exist_ok=True)
    export_model()
    export_assets()
