"""Train the speech model. Runs on an NVIDIA GPU (Colab), the Apple GPU, or the CPU.

Safe to stop and resume with --resume: a checkpoint is written after every epoch.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

import features as fx
import phones
from model import SpeechModel

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


class Data:
    def __init__(self):
        self.rows = [json.loads(l) for l in open(FEAT / "manifest.jsonl", encoding="utf-8")]
        for r in self.rows:
            r["ids"] = [phones.TOKEN_ID[t] for t in r["tokens"].split()]
        self._mels = None

    @property
    def mels(self):
        # Opened lazily so each data-loader worker maps the file itself rather than
        # receiving a pickled copy of several gigabytes.
        if self._mels is None:
            self._mels = np.memmap(FEAT / "mels.bin", dtype=np.float16, mode="r").reshape(-1, fx.N_MELS)
        return self._mels

    def __getstate__(self):
        return {**self.__dict__, "_mels": None}

    def split(self, name, source=None):
        return [r for r in self.rows if r["split"] == name and (source is None or r["source"] == source)]

    def load(self, r, train: bool) -> np.ndarray:
        m = np.array(self.mels[r["offset"]: r["offset"] + r["frames"]], dtype=np.float32)
        if train:
            rate = random.uniform(0.9, 1.1)  # tempo: nearest-frame resampling along time
            idx = np.clip((np.arange(int(len(m) / rate)) * rate).astype(int), 0, len(m) - 1)
            m = m[idx]
            if random.random() < 0.5:  # add a noise floor, in the power domain
                level = m.mean() + random.uniform(-4.0, 0.0)
                m = np.logaddexp(m, level + np.random.randn(*m.shape) * 0.5)
        m = fx.normalize(m)
        if train:  # SpecAugment
            for _ in range(2):
                w = random.randint(0, 15)
                s = random.randint(0, fx.N_MELS - w)
                m[:, s: s + w] = 0
            for _ in range(2):
                w = random.randint(0, min(40, len(m) // 8))
                s = random.randint(0, len(m) - w)
                m[s: s + w] = 0
        return m


def batches(rows, max_frames: int, shuffle: bool):
    order = sorted(rows, key=lambda r: r["frames"] + (random.random() * 40 if shuffle else 0))
    out, cur, longest = [], [], 0
    for r in order:
        longest = max(longest, r["frames"])
        if cur and longest * (len(cur) + 1) > max_frames:
            out.append(cur)
            cur, longest = [], r["frames"]
        cur.append(r)
    if cur:
        out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(data, batch, train):
    mels = [data.load(r, train) for r in batch]
    lens = torch.tensor([len(m) for m in mels])
    x = torch.zeros(len(mels), int(lens.max()), fx.N_MELS)
    for i, m in enumerate(mels):
        x[i, : len(m)] = torch.from_numpy(m)
    y = torch.tensor([t for r in batch for t in r["ids"]])
    ylen = torch.tensor([len(r["ids"]) for r in batch])
    return x, lens, y, ylen, sum(r["frames"] for r in batch)


class BatchSet(torch.utils.data.Dataset):
    def __init__(self, data, batch_list, train):
        self.data, self.batch_list, self.train = data, batch_list, train

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, i):
        return collate(self.data, self.batch_list[i], self.train)


def seed_worker(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def greedy(log_probs, lengths):
    out = []
    for seq, n in zip(log_probs.argmax(-1).cpu().tolist(), lengths.cpu().tolist()):
        ids, prev = [], 0
        for t in seq[:n]:
            if t != prev and t != 0:
                ids.append(t)
            prev = t
        out.append(ids)
    return out


@torch.no_grad()
def evaluate(model, data, rows, device, max_frames=30000):
    """Phone error rate: edits needed to fix the output, per reference phone."""
    model.eval()
    errors = total = 0
    for b in batches(rows, max_frames, False):
        x, lens, _, _, _ = collate(data, b, False)
        lp, out_lens = model(x.to(device), lens.to(device))
        for ids, r in zip(greedy(lp.float(), out_lens), b):
            errors += edit_distance(ids, r["ids"])
            total += len(r["ids"])
    return errors / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--dim", type=int, default=144)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--max-frames", type=int, default=24000)
    ap.add_argument("--limit", type=int, default=0, help="use only this many training clips (for speed tests)")
    ap.add_argument("--max-seconds", type=float, default=0, help="stop cleanly after this long")
    ap.add_argument("--workers", type=int, default=-1, help="data-loading processes (default: 2 on NVIDIA, 0 elsewhere)")
    ap.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints"))
    ap.add_argument("--name", default="model")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    random.seed(7)
    torch.manual_seed(7)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_amp = device == "cuda"  # half precision on NVIDIA GPUs; the loss stays in full precision
    workers = args.workers if args.workers >= 0 else (min(2, os.cpu_count() or 1) if device == "cuda" else 0)
    data = Data()
    train_rows = data.split("train")
    if args.limit:
        train_rows = random.sample(train_rows, args.limit)
    val_ur = data.split("val", "common_voice")
    val_en = data.split("val", "librispeech")
    hours = sum(r["frames"] for r in train_rows) / 360000
    print(f"train clips {len(train_rows)} ({hours:.1f} h), val urdu {len(val_ur)}, val english {len(val_en)}, device {device}", flush=True)

    model = SpeechModel(len(phones.VOCAB), dim=args.dim, blocks=args.blocks).to(device)
    print("parameters", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(batches(train_rows, args.max_frames, False))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch + 50, pct_start=0.12)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    last_path, best_path = ckpt_dir / f"{args.name}_last.pt", ckpt_dir / f"{args.name}_best.pt"
    start, best = 0, 9.0
    if args.resume and last_path.exists():
        c = torch.load(last_path, map_location=device)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        start, best = c["epoch"] + 1, c["best"]
        print(f"resumed at epoch {start}, best {best:.4f}", flush=True)

    log = open(ROOT / "logs" / f"{args.name}.jsonl", "a")
    began = time.time()
    for epoch in range(start, args.epochs):
        model.train()
        t0, loss_sum, n, audio = time.time(), 0.0, 0, 0
        loader = torch.utils.data.DataLoader(
            BatchSet(data, batches(train_rows, args.max_frames, True), True), batch_size=None, shuffle=False,
            num_workers=workers, worker_init_fn=seed_worker, persistent_workers=False, pin_memory=device == "cuda")
        for x, lens, y, ylen, frames in loader:
            x, lens = x.to(device, non_blocking=True), lens.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                lp, out_lens = model(x, lens)
            lp = lp.float().transpose(0, 1)
            if device == "mps":  # CTC loss is not implemented on the Apple GPU
                lp, out_lens = lp.cpu(), out_lens.cpu()
            else:
                y, ylen = y.to(device), ylen.to(device)
            loss = F.ctc_loss(lp, y, out_lens, ylen, blank=0, zero_infinity=True, reduction="sum") / ylen.sum()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            # Shuffled batching can give a few more batches per epoch than the estimate,
            # so stop stepping the schedule at its end instead of raising an error.
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            loss_sum += loss.item()
            n += 1
            audio += frames
            if n % 100 == 0:
                print(f"epoch {epoch} step {n}/{steps_per_epoch} loss {loss_sum/n:.3f} speed {audio/100/(time.time()-t0):.0f}x realtime", flush=True)
        per_ur = evaluate(model, data, val_ur, device)
        per_en = evaluate(model, data, val_en, device)
        score = (per_ur + per_en) / 2
        rec = dict(epoch=epoch, loss=round(loss_sum / max(n, 1), 4), per_ur=round(per_ur, 4), per_en=round(per_en, 4),
                   minutes=round((time.time() - t0) / 60, 1), lr=sched.get_last_lr()[0])
        print(rec, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        state = dict(model=model.state_dict(), config=model.config, epoch=epoch, best=min(best, score), opt=opt.state_dict(), sched=sched.state_dict(), vocab=phones.VOCAB)
        torch.save(state, str(last_path) + ".tmp")
        os.replace(str(last_path) + ".tmp", last_path)  # never leave a half-written checkpoint
        if score < best:
            best = score
            torch.save(dict(model=model.state_dict(), config=model.config, epoch=epoch, score=score, vocab=phones.VOCAB), best_path)
        if args.max_seconds and time.time() - began > args.max_seconds:
            print("time budget reached", flush=True)
            break


if __name__ == "__main__":
    main()
