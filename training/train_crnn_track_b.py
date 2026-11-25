"""
Jalur Penantang (Track B) — Pengenal Baris Segmentation-Free CRNN + BiLSTM + CTC Loss (Apache-2.0/BSD-3)
Untuk AksaraScan v3.0 (Fase 2 / 3 / 4h).

Dilatih langsung pada pasangan gambar baris -> string Unicode Aksara Jawa (`setilanaji/aksara-jawa-ocr`
+ `synth_v2`), menghasilkan transkripsi Unicode secara end-to-end tanpa perlu pemotongan kotak karakter.
Digunakan sebagai pemeriksa silang terhadap Track A (Detektor YOLOX + Dekoder Beam).

Penggunaan:
  python training/train_crnn_track_b.py --epochs 30 --out-dir runs_crnn
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")

# Kosakata Unicode Aksara Jawa kanonik + blank CTC (indeks 0 = <blank>)
JAVANESE_CHARS = (
    [chr(c) for c in range(0xA980, 0xA9E0)]
)


class CRNNTrackB(nn.Module):
    """CNN ringan + BiGRU/BiLSTM 2-layer dengan keluaran CTC log-probs [T, B, vocab]."""

    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # H: 64 -> 32, W: W/2
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.MaxPool2d((2, 2)),  # H: 32 -> 16, W: W/4
            nn.Conv2d(96, 160, 3, padding=1), nn.BatchNorm2d(160), nn.SiLU(),
            nn.Conv2d(160, 160, 3, padding=1), nn.BatchNorm2d(160), nn.SiLU(),
            nn.MaxPool2d((2, 1)),  # H: 16 -> 8, W: W/4
            nn.Conv2d(160, 224, 3, padding=1), nn.BatchNorm2d(224), nn.SiLU(),
            nn.MaxPool2d((2, 1)),  # H: 8 -> 4, W: W/4
        )
        self.proj = nn.Linear(224 * 4, 256)
        self.rnn = nn.LSTM(256, 192, num_layers=2, bidirectional=True, dropout=0.15, batch_first=False)
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        # x: [B, 1, 64, W]
        f = self.cnn(x)  # [B, 224, 4, W/4]
        B, C, H, W = f.shape
        f = f.permute(3, 0, 1, 2).reshape(W, B, C * H)  # [T, B, 896]
        f = F.silu(self.proj(f))
        r, _ = self.rnn(f)
        logits = self.head(r)  # [T, B, V]
        return logits


class LineOCRDataset(Dataset):
    def __init__(self, split="train", img_h=64, max_w=768, max_items=0):
        self.img_h = img_h
        self.max_w = max_w
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(JAVANESE_CHARS)}
        self.samples = []

        # 1. setilanaji/aksara-jawa-ocr (ambil baris tunggal aspect ratio W/H >= 2.2)
        seti_idx = os.path.join(EXT_DIR, "line_ocr", "setilanaji", "index.jsonl")
        if os.path.exists(seti_idx):
            with open(seti_idx, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["split"] != split:
                        continue
                    if r["width"] / max(1, r["height"]) < 2.2:
                        continue
                    ids = [self.char_to_id[c] for c in r["unicode"] if c in self.char_to_id]
                    if ids:
                        self.samples.append((os.path.join(ROOT_DIR, r["image"]), ids))

        # 2. Synth v2 single-line strips (jika tersedia)
        synth_idx = os.path.join(EXT_DIR, "synth_v2", "manifest.jsonl")
        if os.path.exists(synth_idx):
            with open(synth_idx, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r["split"] != split or "\n" in r.get("unicode_gt", ""):
                        continue
                    ids = [self.char_to_id[c] for c in r["unicode_gt"] if c in self.char_to_id]
                    if ids:
                        self.samples.append((os.path.join(ROOT_DIR, r["image"]), ids))

        if max_items > 0 and len(self.samples) > max_items:
            random.Random(42).shuffle(self.samples)
            self.samples = self.samples[:max_items]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ip, ids = self.samples[idx]
        gray = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.full((self.img_h, 64), 245, dtype=np.uint8)
        h, w = gray.shape[:2]
        sc = self.img_h / max(1, h)
        nw = min(self.max_w, max(16, int(round(w * sc))))
        resized = cv2.resize(gray, (nw, self.img_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.img_h, self.max_w), 245, dtype=np.uint8)
        canvas[:, :nw] = resized
        norm = (255.0 - canvas.astype(np.float32)) / 255.0  # tinta=1, kertas=0
        return torch.from_numpy(norm[None, ...]), torch.tensor(ids, dtype=torch.long), nw // 4


def collate_ctc(batch):
    imgs, tgts, in_lens = zip(*batch)
    flat_tgt = torch.cat(tgts, dim=0)
    tgt_lens = torch.tensor([len(t) for t in tgts], dtype=torch.long)
    return torch.stack(imgs, dim=0), flat_tgt, torch.tensor(in_lens, dtype=torch.long), tgt_lens


def decode_ctc_greedy(logits):
    # logits: [T, B, V]
    preds = logits.argmax(dim=-1).permute(1, 0).cpu().numpy()  # [B, T]
    out = []
    for row in preds:
        chars = []
        prev = 0
        for cid in row:
            if cid != 0 and cid != prev and (cid - 1) < len(JAVANESE_CHARS):
                chars.append(JAVANESE_CHARS[cid - 1])
            prev = cid
        out.append("".join(chars))
    return out


def main():
    p = argparse.ArgumentParser(description="Pelatih Track B CRNN-CTC AksaraScan")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument("--max-steps-per-epoch", type=int, default=0)
    p.add_argument("--out-dir", default="runs_crnn")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ds = LineOCRDataset("train", max_items=args.max_items)
    va_ds = LineOCRDataset("val", max_items=max(0, args.max_items // 4) if args.max_items else 0)
    print(f"[CRNN Track B] train={len(tr_ds)} baris | val={len(va_ds)} baris | vocab={len(JAVANESE_CHARS)+1} | device={device}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_ctc)
    model = CRNNTrackB(num_classes=len(JAVANESE_CHARS) + 1).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_loss = float("inf")
    best_pt = os.path.join(out_dir, "crnn_track_b_best.pt")

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_l, steps = 0.0, 0
        for imgs, flat_t, in_l, tgt_l in tr_loader:
            imgs = imgs.to(device)
            opt.zero_grad()
            logits = model(imgs)
            log_p = F.log_softmax(logits, dim=-1)
            loss = ctc_loss(log_p, flat_t.to(device), in_l.to(device), tgt_l.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_l += float(loss.detach().cpu())
            steps += 1
            if args.max_steps_per_epoch > 0 and steps >= args.max_steps_per_epoch:
                break
        mean_l = tot_l / max(1, steps)
        print(f"  Epoch {ep:02d}/{args.epochs} | ctc_loss={mean_l:.4f}")
        if mean_l < best_loss:
            best_loss = mean_l
            torch.save({"model": model.state_dict(), "vocab": JAVANESE_CHARS}, best_pt)

    # Ekspor ONNX Track B & vocab JSON
    model.load_state_dict(torch.load(best_pt, map_location="cpu")["model"])
    model.eval().cpu()
    onnx_out = os.path.join(out_dir, "aksara_crnn_ctc.onnx")
    dummy = torch.zeros(1, 1, 64, 768, dtype=torch.float32)
    export_kw = dict(input_names=["line_image"], output_names=["ctc_logits"], opset_version=14)
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kw["dynamo"] = False
    torch.onnx.export(model, dummy, onnx_out, **export_kw)
    sz_mb = os.path.getsize(onnx_out) / (1024 * 1024)
    with open(os.path.join(out_dir, "crnn_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"blank_id": 0, "chars": JAVANESE_CHARS}, f, ensure_ascii=False)
    print(f"[CRNN ONNX] Tersimpan {onnx_out} ({sz_mb:.2f} MB)")


if __name__ == "__main__":
    main()
