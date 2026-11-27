"""
Pelatih Track A2 — Stage-2 Reference Prototype Metric Verifier (`aksara_embed.onnx <= 3 MB`)
untuk AksaraScan v4.0 (55 Kelas Tier-1 Aksara Jawa + 1 Kelas Penolak `non_aksara` = 56 Kelas).

Fitur Arsitektur & Pelatihan:
  - Input: [B, 3, 64, 64] FP32 (rentang [-1, 1], aspect-preserving median-border pad)
  - Backbone ringan Depthwise-Separable ConvNet + Squeeze-Excite + Spatial Quadrant Mixer (< 1.5 MB ONNX)
  - Sub-Center Cosine Margin Head (3 sub-pusat per kelas) untuk mengakomodasi gaya cetak + tulisan tangan
  - Output ONNX:
      1. `emb`: [B, 64] vektor embedding L2-normalisasi untuk pencocokan cosine ke `reference_prototypes.json`
      2. `logits`: [B, 56] skor klasifikasi terkalibrasi (0..54 kelas Aksara, 55 = non_aksara)
  - Pemisahan ketat: hanya melatih di `split="train"` & validasi di `split="val"` dari `manifests_v4`.

Penggunaan:
  python -u training/train_stage2_embedder.py --epochs 24 --out-dir runs_stage2
"""
import argparse
from collections import Counter, defaultdict
import json
import math
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
from schema_55 import CLASSES_55

NON_AKSARA_CID = 55
NUM_CLASSES = 56  # 0..54 Aksara Tier-1 + 55 non_aksara


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def letterbox_crop_patch(bgr_crop, out_size=64):
    """Ubah ukuran crop menjaga rasio aspek dengan padding warna median tepi gambar."""
    if bgr_crop is None or bgr_crop.size == 0:
        return np.full((out_size, out_size, 3), 235, dtype=np.uint8)
    if bgr_crop.ndim == 2:
        bgr_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_GRAY2BGR)
    h, w = bgr_crop.shape[:2]
    if h < 2 or w < 2:
        return np.full((out_size, out_size, 3), 235, dtype=np.uint8)

    # Estimasi warna latar dari tepi crop agar batas padding mulus (kertas/dinding/dark-mode)
    top = bgr_crop[0, :, :]
    bot = bgr_crop[-1, :, :]
    lft = bgr_crop[:, 0, :]
    rgt = bgr_crop[:, -1, :]
    border_px = np.concatenate([top, bot, lft, rgt], axis=0)
    bg_color = np.median(border_px, axis=0).astype(np.uint8)

    # Sisakan ~4 px margin aman di ukuran target agar struktur tepi tak terpotong
    usable = out_size - 4
    sc = min(usable / float(w), usable / float(h))
    nw = max(2, min(usable, int(round(w * sc))))
    nh = max(2, min(usable, int(round(h * sc))))
    interp = cv2.INTER_AREA if (nw < w or nh < h) else cv2.INTER_LINEAR
    resized = cv2.resize(bgr_crop, (nw, nh), interpolation=interp)

    canvas = np.empty((out_size, out_size, 3), dtype=np.uint8)
    canvas[:, :] = bg_color
    x0 = (out_size - nw) // 2
    y0 = (out_size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def patch_to_tensor(bgr_64):
    rgb = cv2.cvtColor(bgr_64, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return np.transpose(rgb, (2, 0, 1))


def load_stage2_records(split="train", max_per_real_single=4000, max_synth_per_class=260, seed=42):
    """
    Muat indeks crop dari manifests_v4 secara deterministik & bebas kebocoran.
    Mengembalikan list of dict: {image, bbox_norm (atau None full patch), cid, domain}.
    """
    rng = random.Random(seed)
    mdir = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4")
    records = []

    def _read_jsonl(fname):
        fp = os.path.join(mdir, fname)
        out = []
        if not os.path.exists(fp):
            return out
        with open(fp, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip():
                    out.append(json.loads(ln))
        return out

    # 1. Tulisan tangan nyata terisolasi: hanacaraka_v4 + jawa_zip_v4
    for fname, dom, cap in [
        ("hanacaraka_v4.jsonl", "hw_hanacaraka", 10000),
        ("jawa_zip_v4.jsonl", "hw_jawazip", max_per_real_single if split == "train" else 2000),
    ]:
        rows = [r for r in _read_jsonl(fname) if r.get("split") == split and r.get("qa_status") != "reject"]
        if split == "train" and len(rows) > cap:
            rng.shuffle(rows)
            rows = rows[:cap]
        for r in rows:
            ip = os.path.join(ROOT_DIR, r["image"])
            if not os.path.exists(ip):
                continue
            for b in r.get("boxes", []):
                cid = int(b["cid"])
                if 0 <= cid < len(CLASSES_55):
                    records.append({"image": ip, "bbox": b["bbox"], "cid": cid, "domain": dom})

    # 2. Sukukata & Sandhangan tulisan tangan nyata: jawa_syllables_v4
    for r in _read_jsonl("jawa_syllables_v4.jsonl"):
        if r.get("split") != split or r.get("qa_status") == "reject":
            continue
        ip = os.path.join(ROOT_DIR, r["image"])
        if not os.path.exists(ip):
            continue
        for b in r.get("boxes", []):
            cid = int(b["cid"])
            if 0 <= cid < len(CLASSES_55):
                records.append({"image": ip, "bbox": b["bbox"], "cid": cid, "domain": "hw_syllables"})

    # 3. Baris naskah beranotasi guru: setilanaji_teacher_v4 (hanya train)
    if split == "train":
        for r in _read_jsonl("setilanaji_teacher_v4.jsonl"):
            if r.get("split") != split or r.get("qa_status") == "reject":
                continue
            ip = os.path.join(ROOT_DIR, r["image"])
            if not os.path.exists(ip):
                continue
            for b in r.get("boxes", []):
                cid = int(b["cid"])
                if 0 <= cid < len(CLASSES_55):
                    records.append({"image": ip, "bbox": b["bbox"], "cid": cid, "domain": "real_manuscript"})

    # 4. Sintetis elastis multi-kondisi v4: synth_v4 (sampling berimbang per kelas!)
    by_cls = defaultdict(list)
    synth_neg_imgs = []
    for r in _read_jsonl("synth_v4.jsonl"):
        if r.get("split") != split:
            continue
        ip = os.path.join(ROOT_DIR, r["image"])
        if not os.path.exists(ip):
            continue
        dom = "synth_tps" if "tps" in str(r.get("mode", "")).lower() else "synth_printed"
        boxes = r.get("boxes", [])
        if not boxes:
            synth_neg_imgs.append(ip)
            continue
        for b in boxes:
            cid = int(b["cid"])
            if 0 <= cid < len(CLASSES_55):
                by_cls[cid].append({"image": ip, "bbox": b["bbox"], "cid": cid, "domain": dom})

    cap_cls = max_synth_per_class if split == "train" else max(40, max_synth_per_class // 3)
    for cid, items in by_cls.items():
        if len(items) > cap_cls:
            rng.shuffle(items)
            items = items[:cap_cls]
        records.extend(items)

    # 5. Kelas penolak non_aksara (cid = 55): patch Latin/dinding/garis buku + irisan persegi + hasil tambang FP
    for r in _read_jsonl("non_aksara_distractors_v4.jsonl"):
        if r.get("split") != split:
            continue
        ip = os.path.join(ROOT_DIR, r["image"])
        if os.path.exists(ip):
            dom_tag = f"neg_{r.get('distractor_class', 'patch')}"
            records.append({"image": ip, "bbox": None, "cid": NON_AKSARA_CID, "domain": dom_tag})
            # Tambah sub-jendela mendekati persegi agar cocok dgn crop detektor 1-2 karakter Latin
            records.append({"image": ip, "bbox": [rng.uniform(0.20, 0.50), 0.50, rng.uniform(0.18, 0.34), 0.94], "cid": NON_AKSARA_CID, "domain": "neg_latin_sq_win"})
            records.append({"image": ip, "bbox": [rng.uniform(0.50, 0.80), 0.50, rng.uniform(0.18, 0.34), 0.94], "cid": NON_AKSARA_CID, "domain": "neg_latin_sq_win"})

    if split == "train":
        for r in _read_jsonl("stage2_hard_neg_train_only.jsonl"):
            if r.get("split") != "train":
                continue
            ip = os.path.join(ROOT_DIR, r["image"])
            if os.path.exists(ip):
                records.append({"image": ip, "bbox": None, "cid": NON_AKSARA_CID, "domain": f"neg_{r.get('distractor_class', 'hard_sq')}"})

    # Ekstrak potongan multi-jendela persegi dari halaman negatif murni synth_v4 (Latin + tembok + buku bergaris)
    for nip in synth_neg_imgs:
        for _ in range(6 if split == "train" else 4):
            cx = rng.uniform(0.10, 0.90)
            cy = rng.uniform(0.12, 0.88)
            bw = rng.uniform(0.035, 0.11)
            bh = rng.uniform(0.06, 0.18)
            records.append({"image": nip, "bbox": [cx, cy, bw, bh], "cid": NON_AKSARA_CID, "domain": "neg_fullpage_subcrop"})

    return records


class Stage2CropDataset(Dataset):
    """Ekstraktor crop 64x64 dengan pra-ekstraksi ROI ke RAM agar pelatihan CPU super cepat."""

    def __init__(self, records, img_size=64, augment=True, max_roi_side=88):
        self.records = records
        self.img_size = img_size
        self.augment = augment
        self.class_counts = Counter(r["cid"] for r in records)
        self.domain_counts = Counter(r["domain"] for r in records)

        # Pra-baca & ekstrak ROI (+24% margin untuk jitter augmentasi) sekali di awal
        # Kelompokkan per file agar setiap gambar dibaca tepat 1x dari disk!
        by_file = defaultdict(list)
        for i, r in enumerate(records):
            by_file[r["image"]].append((i, r["bbox"]))

        self.rois = [None] * len(records)
        self.cids = [r["cid"] for r in records]
        self.static_tensors = None

        for ip, items in by_file.items():
            img = cv2.imread(ip)
            if img is None:
                fallback = np.full((img_size, img_size, 3), 235, dtype=np.uint8)
                for idx, _ in items:
                    self.rois[idx] = fallback
                continue
            H, W = img.shape[:2]
            for idx, bbox in items:
                if bbox is None:
                    crop = img
                else:
                    cx, cy, bw, bh = bbox
                    pad = 0.24 if augment else 0.08
                    px1 = max(0, int(round((cx - bw * (0.5 + pad)) * W)))
                    py1 = max(0, int(round((cy - bh * (0.5 + pad)) * H)))
                    px2 = min(W, int(round((cx + bw * (0.5 + pad)) * W)))
                    py2 = min(H, int(round((cy + bh * (0.5 + pad)) * H)))
                    crop = img[py1:py2, px1:px2] if (px2 - px1 >= 3 and py2 - py1 >= 3) else img
                ch, cw = crop.shape[:2]
                if max(ch, cw) > max_roi_side:
                    sc = max_roi_side / float(max(ch, cw))
                    crop = cv2.resize(crop, (max(2, int(round(cw * sc))), max(2, int(round(ch * sc)))), interpolation=cv2.INTER_AREA)
                else:
                    crop = crop.copy()
                self.rois[idx] = crop

        if not augment:
            # Pra-hitung seluruh tensor validasi 64x64 sekali (< 65 MB RAM)
            self.static_tensors = np.stack([patch_to_tensor(letterbox_crop_patch(roi, img_size)) for roi in self.rois], axis=0).astype(np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if self.static_tensors is not None:
            return torch.from_numpy(self.static_tensors[idx]), self.cids[idx]

        roi = self.rois[idx]
        H, W = roi.shape[:2]
        # Jitter & tight-to-loose crop acak dari ROI ber-margin 24%
        if self.records[idx]["bbox"] is not None and H >= 8 and W >= 8:
            # Margin asli ~24% di tiap sisi, trim 0..16% per sisi untuk variasi bbox detektor
            trim_l = random.uniform(0.0, 0.14)
            trim_r = random.uniform(0.0, 0.14)
            trim_t = random.uniform(0.0, 0.14)
            trim_b = random.uniform(0.0, 0.14)
            x1 = int(round(W * trim_l))
            x2 = max(x1 + 4, int(round(W * (1.0 - trim_r))))
            y1 = int(round(H * trim_t))
            y2 = max(y1 + 4, int(round(H * (1.0 - trim_b))))
            crop = roi[y1:y2, x1:x2]
        else:
            crop = roi

        patch = letterbox_crop_patch(crop, self.img_size)

        # Rotasi ringan +-6 derajat & penipisan/penebalan goresan; DILARANG flip horizontal/vertikal!
        if random.random() < 0.45:
            ang = random.uniform(-6.0, 6.0)
            med = np.median(patch[0], axis=0).tolist()
            M = cv2.getRotationMatrix2D((self.img_size / 2.0, self.img_size / 2.0), ang, random.uniform(0.92, 1.08))
            patch = cv2.warpAffine(patch, M, (self.img_size, self.img_size), borderValue=med)
        if random.random() < 0.22:
            k = random.choice([3, 5])
            patch = cv2.GaussianBlur(patch, (k, k), random.uniform(0.35, 1.0))
        if random.random() < 0.25:
            ker = np.ones((2, 2), np.uint8)
            patch = cv2.erode(patch, ker, iterations=1) if random.random() < 0.5 else cv2.dilate(patch, ker, iterations=1)
        if random.random() < 0.40:
            alpha = random.uniform(0.78, 1.25)
            beta = random.uniform(-22, 22)
            tint = np.array([random.uniform(0.90, 1.10) for _ in range(3)], dtype=np.float32)
            patch = np.clip(patch.astype(np.float32) * alpha * tint + beta, 0, 255).astype(np.uint8)
        if random.random() < 0.12:
            patch = 255 - patch

        return torch.from_numpy(patch_to_tensor(patch)), self.cids[idx]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        s = x.mean(dim=(2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class SepConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_se=True):
        super().__init__()
        self.use_res = (stride == 1 and in_c == out_c)
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c) if use_se else nn.Identity()

    def forward(self, x):
        y = F.relu(self.bn1(self.dw(x)), inplace=True)
        y = F.relu(self.bn2(self.pw(y)), inplace=True)
        y = self.se(y)
        return x + y if self.use_res else y


class AksaraStage2Embedder(nn.Module):
    """
    Metric Embedding ConvNet sangat ringan (~1.1 MB ONNX FP32) untuk Stage-2 Prototype Matching
    serta penolakan distraktor non-aksara (Latin / garis dinding).
    """

    def __init__(self, num_classes=NUM_CLASSES, emb_dim=64, subcenters=3, cosine_scale=22.0):
        super().__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.subcenters = subcenters
        self.cosine_scale = cosine_scale

        # 64 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 32 -> 16 -> 8 -> 4
        self.stage1 = nn.Sequential(SepConvBlock(32, 64, stride=2), SepConvBlock(64, 64, stride=1))
        self.stage2 = nn.Sequential(SepConvBlock(64, 128, stride=2), SepConvBlock(128, 128, stride=1))
        self.stage3 = nn.Sequential(SepConvBlock(128, 192, stride=2), SepConvBlock(192, 192, stride=1))

        # Pencampur spasial kuadran 4x4 agar orientasi kiri/kanan/atas/bawah sandhangan tetap terjaga
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(192, 192, 4, stride=1, padding=0, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(384, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        # Bobot sub-center cosine classifier: [num_classes * subcenters, emb_dim]
        self.prototypes = nn.Parameter(torch.randn(num_classes * subcenters, emb_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x):
        f = self.stage3(self.stage2(self.stage1(self.stem(x))))  # [B, 192, 4, 4]
        g_avg = f.mean(dim=(2, 3))
        g_sp = self.spatial_dw(f).flatten(1)
        emb = self.proj(torch.cat([g_avg, g_sp], dim=1))
        emb = F.normalize(emb, p=2, dim=1)

        w = F.normalize(self.prototypes, p=2, dim=1)
        sim_sub = F.linear(emb, w).view(-1, self.num_classes, self.subcenters)
        cos_cls = sim_sub.max(dim=2).values
        logits = cos_cls * self.cosine_scale
        return emb, logits


def subcenter_cosface_loss(emb, logits_raw, prototypes, targets, num_classes=NUM_CLASSES, subcenters=3, scale=22.0, margin=0.18, class_weights=None):
    w = F.normalize(prototypes, p=2, dim=1)
    sim_sub = F.linear(emb, w).view(-1, num_classes, subcenters)
    cos_cls = sim_sub.max(dim=2).values
    one_hot = F.one_hot(targets, num_classes).float()
    cos_m = cos_cls - one_hot * margin
    logits_m = cos_m * scale
    return F.cross_entropy(logits_m, targets, weight=class_weights, label_smoothing=0.05)


@torch.no_grad()
def evaluate_split(model, records, va_loader, device):
    model.eval()
    preds_all, tgts_all = [], []
    for imgs, cids in va_loader:
        imgs = imgs.to(device)
        _emb, logits = model(imgs)
        pr = F.softmax(logits, dim=1).cpu().numpy()
        preds_all.append(np.argmax(pr, axis=1))
        tgts_all.append(cids.numpy())

    preds = np.concatenate(preds_all)
    tgts = np.concatenate(tgts_all)

    # Akurasi per domain
    dom_acc = {}
    for dom in sorted(set(r["domain"] for r in records)):
        idx = [i for i, r in enumerate(records) if r["domain"] == dom]
        dom_acc[dom] = round(float((preds[idx] == tgts[idx]).mean()), 4)

    # Akurasi macro 55 kelas Aksara (di subset aksara)
    aks_mask = tgts < NON_AKSARA_CID
    aks_acc = float((preds[aks_mask] == tgts[aks_mask]).mean()) if aks_mask.any() else 0.0
    cls_accs = []
    for cid in range(NON_AKSARA_CID):
        m = (tgts == cid)
        if m.any():
            cls_accs.append(float((preds[m] == cid).mean()))
    macro55 = float(np.mean(cls_accs)) if cls_accs else 0.0

    # F1 penolakan non_aksara (kelas 55 vs <55)
    is_neg_gt = (tgts == NON_AKSARA_CID)
    is_neg_pr = (preds == NON_AKSARA_CID)
    tp = int((is_neg_gt & is_neg_pr).sum())
    fp = int((~is_neg_gt & is_neg_pr).sum())
    fn = int((is_neg_gt & ~is_neg_pr).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    neg_f1 = (2 * prec * rec / max(1e-8, prec + rec))

    real_idx = [i for i, r in enumerate(records) if r["domain"].startswith("hw_") or r["domain"] == "real_manuscript"]
    real_acc = float((preds[real_idx] == tgts[real_idx]).mean()) if real_idx else aks_acc

    score = 0.45 * real_acc + 0.35 * macro55 + 0.20 * neg_f1
    return {
        "score": round(score, 4),
        "real_hw_acc": round(real_acc, 4),
        "aks_top1_acc": round(aks_acc, 4),
        "macro55_acc": round(macro55, 4),
        "non_aksara_f1": round(neg_f1, 4),
        "non_aksara_prec": round(prec, 4),
        "non_aksara_rec": round(rec, 4),
        "domain_acc": dom_acc,
        "covered_classes": len(cls_accs),
    }


def export_stage2_onnx(model, out_onnx_path):
    model.eval()
    model_cpu = model.cpu()
    dummy = torch.randn(4, 3, 64, 64, dtype=torch.float32)
    with torch.no_grad():
        pt_emb, pt_logits = model_cpu(dummy)

    os.makedirs(os.path.dirname(out_onnx_path), exist_ok=True)
    export_kw = dict(
        input_names=["crops"],
        output_names=["emb", "logits"],
        dynamic_axes={"crops": {0: "batch"}, "emb": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
    )
    if "dynamo" in torch.onnx.export.__code__.co_varnames:
        export_kw["dynamo"] = False
    torch.onnx.export(model_cpu, dummy, out_onnx_path, **export_kw)

    sess = ort.InferenceSession(out_onnx_path, providers=["CPUExecutionProvider"])
    o_emb, o_logits = sess.run(None, {"crops": dummy.numpy()})
    diff_e = float(np.max(np.abs(pt_emb.numpy() - o_emb)))
    diff_l = float(np.max(np.abs(pt_logits.numpy() - o_logits)))
    sz_mb = os.path.getsize(out_onnx_path) / (1024 * 1024)
    print(f"[STAGE-2 ONNX] {out_onnx_path} ({sz_mb:.2f} MB) | diff_emb={diff_e:.2e} | diff_logits={diff_l:.2e}")
    return sz_mb, max(diff_e, diff_l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0025)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-real", type=int, default=4000)
    p.add_argument("--max-synth-per-cls", type=int, default=240)
    p.add_argument("--out-dir", default="runs_stage2")
    p.add_argument("--deploy-onnx", default="app/web_camscanner/aksara_embed.onnx")
    args = p.parse_args()

    set_seed(args.seed)
    if not torch.cuda.is_available():
        torch.set_num_threads(min(12, os.cpu_count() or 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_rec = load_stage2_records("train", max_per_real_single=args.max_per_real, max_synth_per_class=args.max_synth_per_cls, seed=args.seed)
    va_rec = load_stage2_records("val", max_per_real_single=2000, max_synth_per_class=80, seed=args.seed)
    t_cache0 = time.time()
    tr_ds = Stage2CropDataset(tr_rec, img_size=64, augment=True)
    va_ds = Stage2CropDataset(va_rec, img_size=64, augment=False)
    va_loader = DataLoader(va_ds, batch_size=256, shuffle=False, num_workers=0)

    print(f"[STAGE-2 TRAIN] device={device} | threads={torch.get_num_threads()} | train_crops={len(tr_rec)} | val_crops={len(va_rec)} | precache={time.time()-t_cache0:.1f}s")
    print(f"  Train domains={dict(tr_ds.domain_counts)}")
    print(f"  Classes covered in train={len(tr_ds.class_counts)}/{NUM_CLASSES} | non_aksara={tr_ds.class_counts[NON_AKSARA_CID]}")

    # Bobot seimbang per kelas; jaga proporsi non_aksara ~20% agar 55 kelas Aksara tetap terlatih optimal
    counts = np.array([max(1, tr_ds.class_counts.get(c, 1)) for c in range(NUM_CLASSES)], dtype=np.float32)
    cls_w = (np.median(counts) / counts) ** 0.40
    cls_w[NON_AKSARA_CID] = float(np.clip(cls_w[NON_AKSARA_CID], 0.55, 0.85))
    cls_w = np.clip(cls_w, 0.5, 4.0)

    sample_weights = []
    for r in tr_rec:
        w = float(cls_w[r["cid"]])
        if r["domain"] in ("hw_hanacaraka", "hw_syllables", "real_manuscript"):
            w *= 1.45
        elif r["cid"] == NON_AKSARA_CID and ("mined_fp" in r["domain"] or "char_sq" in r["domain"] or "sq" in r["domain"]):
            w *= 1.35
        sample_weights.append(w)

    num_samples_ep = min(len(tr_rec), 14000)
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples_ep, replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, sampler=sampler, num_workers=0, drop_last=False)

    model = AksaraStage2Embedder(num_classes=NUM_CLASSES, emb_dim=64, subcenters=3, cosine_scale=22.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params={n_params:,} (~{n_params*4/(1024*1024):.2f} MB FP32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.03)
    cls_w_tensor = torch.tensor(cls_w, dtype=torch.float32, device=device)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    best_ckpt = os.path.join(out_dir, f"stage2_embed_s{args.seed}_best.pt")
    best_score = -1.0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, total = 0.0, 0, 0
        for imgs, cids in tr_loader:
            imgs = imgs.to(device)
            cids = cids.to(device).long()
            optimizer.zero_grad()
            emb, logits = model(imgs)
            loss = subcenter_cosface_loss(
                emb, logits, model.prototypes, cids,
                num_classes=NUM_CLASSES, subcenters=3, scale=22.0,
                margin=0.18 if ep > 2 else 0.08, class_weights=cls_w_tensor,
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * imgs.size(0)
            correct += int((logits.argmax(dim=1) == cids).sum().cpu())
            total += imgs.size(0)

        scheduler.step()
        tr_loss = loss_sum / max(1, total)
        tr_acc = correct / max(1, total)

        val_m = evaluate_split(model, va_rec, va_loader, device)
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"  Ep {ep:02d}/{args.epochs} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
            f"val_hw={val_m['real_hw_acc']:.4f} val_macro55={val_m['macro55_acc']:.4f} "
            f"neg_f1={val_m['non_aksara_f1']:.4f} score={val_m['score']:.4f} | {dt:.1f}s"
        )
        history.append({"epoch": ep, "tr_loss": round(tr_loss, 4), "tr_acc": round(tr_acc, 4), "lr": cur_lr, "elapsed_s": round(dt, 2), **val_m})
        if val_m["score"] > best_score:
            best_score = val_m["score"]
            torch.save({"model": model.state_dict(), "val_metrics": val_m, "epoch": ep, "seed": args.seed}, best_ckpt)

    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    onnx_local = os.path.join(out_dir, f"aksara_embed_s{args.seed}.onnx")
    sz_mb, diff = export_stage2_onnx(model, onnx_local)

    deploy_abs = args.deploy_onnx if os.path.isabs(args.deploy_onnx) else os.path.join(ROOT_DIR, args.deploy_onnx)
    if sz_mb <= 3.0 and diff < 1e-4:
        os.makedirs(os.path.dirname(deploy_abs), exist_ok=True)
        import shutil
        shutil.copy2(onnx_local, deploy_abs)
        print(f"[DEPLOY STAGE-2] Disalin ke {deploy_abs} ({sz_mb:.2f} MB <= 3.0 MB)")

    hist_path = os.path.join(out_dir, f"stage2_s{args.seed}_history.json")
    with open(hist_path, "w", encoding="utf-8") as hf:
        json.dump({
            "seed": args.seed,
            "epochs": args.epochs,
            "best_epoch": ckpt["epoch"],
            "best_val_metrics": ckpt["val_metrics"],
            "onnx_size_mb": round(sz_mb, 3),
            "onnx_parity_max_diff": diff,
            "history": history,
        }, hf, indent=2)
    print(f"[DONE] Best score={ckpt['val_metrics']} | History={hist_path}")


if __name__ == "__main__":
    main()
