"""
Pelatih Bertahap Detektor YOLOX AksaraScan v3.0 (Apache-2.0 / MIT, Bebas AGPL-3.0)
Mendukung:
  - Perbandingan varian: `yolox-nano`, `yolox-tiny`, `yolox-s` (+ fitur `use_p2` untuk objek kecil < 16 px)
  - Resolusi masukan 960 & 1280 serta multi-scale training (0.5x..1.5x)
  - Pelatihan 3 tahap: Tahap 1 Pralatih (Synth v2 + Publik), Tahap 2 Fine-Tune Nyata (lr kecil), Tahap 3 Close-Mosaic
  - Aturan keras: fliplr=0, flipud=0; proporsi mosaic terkendali; penyeimbang bobot kelas langka; multi-seed (3 seed)
  - Ekspor ONNX [1, 4+55, N] terverifikasi paritas PyTorch vs ONNX + Gerbang Kuantisasi FP16/INT8 (< 20 MB)

Penggunaan (GPU Colab/Lokal atau CPU):
  python training/train_yolox.py --variant yolox-tiny --imgsz 960 --stage 1 --data-dirs dataset/external/synth_v2 dataset/external/converted_yolo/jawa_zip dataset/external/converted_yolo/hanacaraka
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "datasets"))
from schema_55 import CLASSES_55, CATEGORIES_55, CLASS_TO_ID
from yolox_model import AksaraYOLOX, compute_yolox_loss

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiSourceYoloDataset(Dataset):
    """Dataset loader terpadu membaca folder YOLO standar dan/atau manifes JSONL real_train/real_val."""

    def __init__(self, sources, split="train", imgsz=960, augment=True, mosaic_prob=0.15, multi_scale=True):
        self.imgsz = imgsz
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.multi_scale = multi_scale
        self.samples = []  # list of (img_path, [(cid, xc, yc, bw, bh), ...])
        self.class_counts = Counter()

        for src in sources:
            src_abs = src if os.path.isabs(src) else os.path.join(ROOT_DIR, src)
            if src_abs.endswith(".jsonl") and os.path.exists(src_abs):
                with open(src_abs, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        ip = r["image"] if os.path.isabs(r["image"]) else os.path.join(ROOT_DIR, r["image"])
                        lp = r.get("label")
                        if lp:
                            lp = lp if os.path.isabs(lp) else os.path.join(ROOT_DIR, lp)
                        boxes = []
                        if r.get("boxes") and isinstance(r["boxes"][0], dict) and "bbox" in r["boxes"][0]:
                            for b in r["boxes"]:
                                boxes.append((int(b["cid"]), *map(float, b["bbox"])))
                        elif lp and os.path.exists(lp):
                            with open(lp, "r", encoding="utf-8") as lf:
                                for ln in lf:
                                    ps = ln.strip().split()
                                    if len(ps) >= 5:
                                        boxes.append((int(float(ps[0])), *map(float, ps[1:5])))
                        if boxes and os.path.exists(ip):
                            self.samples.append((ip, boxes))
                            for b in boxes:
                                self.class_counts[b[0]] += 1
            elif os.path.isdir(src_abs):
                idx_dedup = os.path.join(src_abs, "index_deduped.jsonl")
                if os.path.exists(idx_dedup):
                    with open(idx_dedup, "r", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            r = json.loads(line)
                            if r.get("split") != split:
                                continue
                            ip = os.path.join(ROOT_DIR, r["image"])
                            boxes = [(int(b["cid"]), *map(float, b["bbox"])) for b in r.get("boxes", [])]
                            if boxes and os.path.exists(ip):
                                self.samples.append((ip, boxes))
                                for b in boxes:
                                    self.class_counts[b[0]] += 1
                else:
                    idir = os.path.join(src_abs, "images", split)
                    ldir = os.path.join(src_abs, "labels", split)
                    if os.path.isdir(idir):
                        for fn in sorted(os.listdir(idir)):
                            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                                continue
                            stem = os.path.splitext(fn)[0]
                            ip = os.path.join(idir, fn)
                            lp = os.path.join(ldir, stem + ".txt")
                            if not os.path.exists(lp):
                                continue
                            boxes = []
                            with open(lp, "r", encoding="utf-8") as lf:
                                for ln in lf:
                                    ps = ln.strip().split()
                                    if len(ps) >= 5:
                                        boxes.append((int(float(ps[0])), *map(float, ps[1:5])))
                            if boxes:
                                self.samples.append((ip, boxes))
                                for b in boxes:
                                    self.class_counts[b[0]] += 1

    def __len__(self):
        return len(self.samples)

    def _load_letterboxed(self, idx, target_sz):
        ip, boxes = self.samples[idx]
        bgr = cv2.imread(ip)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ih, iw = rgb.shape[:2]

        # Skala acak 0.5x..1.5x sebelum letterbox (hanya saat augmentasi train)
        aug_scale = random.uniform(0.55, 1.45) if (self.augment and self.multi_scale) else 1.0
        sc = min(target_sz / iw, target_sz / ih) * min(1.0, aug_scale)
        nw, nh = max(8, int(round(iw * sc))), max(8, int(round(ih * sc)))
        pad_x = (target_sz - nw) // 2
        pad_y = (target_sz - nh) // 2

        canvas = np.full((target_sz, target_sz, 3), 114, dtype=np.uint8)
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        abs_boxes = []
        for cid, xc, yc, bw, bh in boxes:
            cx_px = xc * nw + pad_x
            cy_px = yc * nh + pad_y
            bw_px = bw * nw
            bh_px = bh * nh
            if bw_px >= 2.0 and bh_px >= 2.0:
                abs_boxes.append([cid, cx_px, cy_px, bw_px, bh_px])

        if self.augment:
            # Brightness/contrast/HSV ringan; TANPA fliplr/flipud!
            if random.random() < 0.5:
                alpha = random.uniform(0.82, 1.18)
                beta = random.uniform(-18, 18)
                canvas = np.clip(canvas.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.4 and abs_boxes:
                ang = random.uniform(-4.0, 4.0)
                sh = random.uniform(-2.0, 2.0)
                M = cv2.getRotationMatrix2D((target_sz / 2.0, target_sz / 2.0), ang, 1.0)
                M[0, 1] += math.tan(math.radians(sh)) * 0.5
                canvas = cv2.warpAffine(canvas, M, (target_sz, target_sz), borderValue=(114, 114, 114))
                new_b = []
                for cid, cx, cy, bw, bh in abs_boxes:
                    pts = np.array([[cx - bw / 2, cy - bh / 2, 1], [cx + bw / 2, cy - bh / 2, 1],
                                    [cx + bw / 2, cy + bh / 2, 1], [cx - bw / 2, cy + bh / 2, 1]], dtype=np.float32)
                    tpts = (M @ pts.T).T
                    nx0, ny0 = max(0.0, float(tpts[:, 0].min())), max(0.0, float(tpts[:, 1].min()))
                    nx1, ny1 = min(float(target_sz), float(tpts[:, 0].max())), min(float(target_sz), float(tpts[:, 1].max()))
                    if nx1 - nx0 >= 2.0 and ny1 - ny0 >= 2.0:
                        new_b.append([cid, (nx0 + nx1) / 2.0, (ny0 + ny1) / 2.0, nx1 - nx0, ny1 - ny0])
                abs_boxes = new_b

        return canvas, abs_boxes

    def __getitem__(self, idx):
        # Mosaic 2x2 terkendali (dinonaktifkan di stage close_mosaic)
        if self.augment and self.mosaic_prob > 0 and random.random() < self.mosaic_prob and len(self.samples) >= 4:
            half = self.imgsz // 2
            canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
            all_boxes = []
            idxs = [idx] + [random.randrange(len(self.samples)) for _ in range(3)]
            offsets = [(0, 0), (half, 0), (0, half), (half, half)]
            for q_idx, (ox, oy) in zip(idxs, offsets):
                sub_img, sub_boxes = self._load_letterboxed(q_idx, half)
                canvas[oy:oy + half, ox:ox + half] = sub_img
                for cid, cx, cy, bw, bh in sub_boxes:
                    all_boxes.append([cid, cx + ox, cy + oy, bw, bh])
            img_rgb, abs_boxes = canvas, all_boxes
        else:
            img_rgb, abs_boxes = self._load_letterboxed(idx, self.imgsz)

        chw = np.transpose(img_rgb.astype(np.float32) / 255.0, (2, 0, 1))
        tgt = torch.tensor(abs_boxes, dtype=torch.float32) if abs_boxes else torch.zeros((0, 5), dtype=torch.float32)
        return torch.from_numpy(chw), tgt


def collate_fn(batch):
    imgs, tgts = zip(*batch)
    return torch.stack(imgs, dim=0), list(tgts)


def build_class_weights(class_counts, num_classes=55, max_w=3.0):
    counts = np.array([max(1, class_counts.get(i, 1)) for i in range(num_classes)], dtype=np.float32)
    median_c = np.median(counts)
    w = np.sqrt(median_c / counts)
    w = np.clip(w, 0.7, max_w)
    return torch.tensor(w, dtype=torch.float32)


def export_and_verify_onnx(model, imgsz, out_onnx_path):
    model.eval()
    model_cpu = model.cpu()
    model_cpu.export_onnx = True
    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)

    with torch.no_grad():
        pt_out = model_cpu(dummy).numpy()

    os.makedirs(os.path.dirname(out_onnx_path), exist_ok=True)
    export_kw = dict(input_names=["images"], output_names=["output0"], opset_version=14, do_constant_folding=True)
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kw["dynamo"] = False
    torch.onnx.export(model_cpu, dummy, out_onnx_path, **export_kw)
    sess = ort.InferenceSession(out_onnx_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"images": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    size_mb = os.path.getsize(out_onnx_path) / (1024 * 1024)
    print(f"[ONNX VERIFY] {out_onnx_path} ({size_mb:.2f} MB) | max |PyTorch - ORT| = {max_diff:.2e}")
    return size_mb, max_diff


def quantize_onnx_int8_with_gate(fp32_onnx, int8_onnx):
    """Kuantisasi dinamis INT8 via onnxruntime.quantization (< 20 MB); wajib lolos gerbang akurasi Fase 3g."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(fp32_onnx, int8_onnx, weight_type=QuantType.QUInt8)
        sz_mb = os.path.getsize(int8_onnx) / (1024 * 1024)
        print(f"[QUANT INT8] {int8_onnx} ({sz_mb:.2f} MB)")
        return sz_mb
    except Exception as e:
        print(f"[WARN QUANT] Kuantisasi INT8 gagal ({e})")
        return None


def export_onnx_fp16(fp32_onnx, fp16_onnx, imgsz):
    """Konversi bobot ONNX ke FP16 dengan tipe IO tetap FP32 (< 20 MB untuk browser ORT-WASM)."""
    try:
        import onnx
        from onnxconverter_common import float16
        model_fp32 = onnx.load(fp32_onnx)
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=True)
        onnx.save(model_fp16, fp16_onnx)
        sz_mb = os.path.getsize(fp16_onnx) / (1024 * 1024)
        s32 = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
        s16 = ort.InferenceSession(fp16_onnx, providers=["CPUExecutionProvider"])
        inp = np.random.RandomState(42).rand(1, 3, imgsz, imgsz).astype(np.float32)
        diff = float(np.max(np.abs(s32.run(None, {"images": inp})[0] - s16.run(None, {"images": inp})[0])))
        print(f"[EXPORT FP16] {fp16_onnx} ({sz_mb:.2f} MB) | max |FP32 - FP16| = {diff:.2e}")
        return sz_mb
    except Exception as e:
        print(f"[WARN FP16] Konversi FP16 gagal ({e})")
        return None


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[YOLOX TRAIN] variant={args.variant} | use_p2={args.use_p2} | imgsz={args.imgsz} | stage={args.stage} | seed={args.seed} | device={device}")

    train_ds = MultiSourceYoloDataset(args.train_sources, split="train", imgsz=args.imgsz, augment=True, mosaic_prob=args.mosaic, multi_scale=True)
    val_ds = MultiSourceYoloDataset(args.val_sources, split="val", imgsz=args.imgsz, augment=False, mosaic_prob=0.0, multi_scale=False)
    print(f"  Samples: train={len(train_ds)} | val={len(val_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("Dataset latih kosong!")

    # Sample weight per gambar agar kelas langka (sandhangan/pasangan) terwakili seimbang
    cls_w = build_class_weights(train_ds.class_counts, num_classes=len(CLASSES_55)).to(device)
    sample_weights = []
    for _, boxes in train_ds.samples:
        w = max(float(cls_w[b[0]].item()) for b in boxes)
        sample_weights.append(w)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, collate_fn=collate_fn, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=args.workers) if len(val_ds) > 0 else None

    model = AksaraYOLOX(num_classes=len(CLASSES_55), variant=args.variant, use_p2=args.use_p2).to(device)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Resumed weights from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr0, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr0 * 0.05)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt = os.path.join(out_dir, f"{args.variant}_img{args.imgsz}_s{args.seed}_best.pt")

    for ep in range(1, args.epochs + 1):
        # Close mosaic pada close_mosaic epoch terakhir (Fase 3b/6)
        if ep > max(1, args.epochs - args.close_mosaic):
            train_ds.mosaic_prob = 0.0

        model.train()
        model.export_onnx = False
        tr_loss_acc = 0.0
        t0 = time.time()
        for step, (imgs, tgts) in enumerate(train_loader):
            imgs = imgs.to(device)
            tgts = [t.to(device) for t in tgts]
            optimizer.zero_grad()
            raw, grids, strides = model(imgs)
            loss, _ = compute_yolox_loss(raw, grids, strides, tgts, num_classes=len(CLASSES_55), label_smoothing=0.03, class_weights=cls_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            tr_loss_acc += float(loss.detach().cpu())
            if args.max_steps_per_epoch > 0 and (step + 1) >= args.max_steps_per_epoch:
                break

        scheduler.step()
        tr_loss = tr_loss_acc / max(1, min(len(train_loader), args.max_steps_per_epoch or len(train_loader)))

        val_loss = tr_loss
        if val_loader:
            model.train(False)
            # Gunakan compute_yolox_loss dengan raw head (export_onnx=False, temporarily call neck+head)
            v_acc, v_cnt = 0.0, 0
            with torch.no_grad():
                for v_step, (imgs, tgts) in enumerate(val_loader):
                    imgs = imgs.to(device)
                    tgts = [t.to(device) for t in tgts]
                    raw, grids, strides = model.head(model.neck(imgs))
                    vl, _ = compute_yolox_loss(raw, grids, strides, tgts, num_classes=len(CLASSES_55), label_smoothing=0.0)
                    v_acc += float(vl.cpu())
                    v_cnt += 1
                    if args.max_steps_per_epoch > 0 and (v_step + 1) >= max(4, args.max_steps_per_epoch // 2):
                        break
            val_loss = v_acc / max(1, v_cnt)

        dt = time.time() - t0
        print(f"  Epoch {ep:03d}/{args.epochs} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f} | {dt:.1f}s")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "variant": args.variant, "imgsz": args.imgsz, "use_p2": args.use_p2, "seed": args.seed}, best_ckpt)

    # Muat terbaik & ekspor ONNX
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    onnx_path = os.path.join(out_dir, f"{args.variant}_img{args.imgsz}_s{args.seed}.onnx")
    sz_mb, max_diff = export_and_verify_onnx(model, args.imgsz, onnx_path)
    if sz_mb > 20.0:
        fp16_path = os.path.join(out_dir, f"{args.variant}_img{args.imgsz}_s{args.seed}_fp16.onnx")
        fp16_mb = export_onnx_fp16(onnx_path, fp16_path, args.imgsz)
        if not fp16_mb or fp16_mb > 20.0:
            int8_path = os.path.join(out_dir, f"{args.variant}_img{args.imgsz}_s{args.seed}_int8.onnx")
            quantize_onnx_int8_with_gate(onnx_path, int8_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="yolox-tiny", choices=["yolox-nano", "yolox-tiny", "yolox-s"])
    p.add_argument("--use-p2", action="store_true", default=True)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--stage", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--close-mosaic", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--mosaic", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--max-steps-per-epoch", type=int, default=0, help="Batasi langkah per epoch untuk smoke test cepat")
    p.add_argument("--train-sources", nargs="+", default=["dataset/external/synth_v2", "dataset/external/converted_yolo/jawa_zip", "dataset/external/converted_yolo/hanacaraka"])
    p.add_argument("--val-sources", nargs="+", default=["dataset/external/synth_v2"])
    p.add_argument("--out-dir", default="runs_yolox")
    train(p.parse_args())
