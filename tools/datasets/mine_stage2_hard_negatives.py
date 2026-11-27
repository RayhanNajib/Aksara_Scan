"""
Ekstraktor Hard Negative Berbentuk Persegi/Karakter-Tunggal (Strictly SPLIT=TRAIN ONLY!)
untuk melatih kelas penolak #55 (`non_aksara`) di Stage-2 Metric Verifier.

Sumber (100% split="train", 0 sentuhan val/heldout):
  1. Kotak False Positive nyata dari detektor baseline (aksara_yolo.onnx) pada 155 citra negatif murni synth_v4/train
     (teks Latin murni, kertas bergaris buku, kotak milimeter, tembok retak/bayangan).
  2. Kotak False Positive nyata pada latar di luar kotak GT (IoU < 0.05) citra positif synth_v4/train.
  3. Irisan karakter-tunggal/bigram persegi dari non_aksara_distractors_v4 (split="train").
  4. Render 600 karakter tunggal & pasangan huruf Latin/angka SIL OFL (Caveat, Newsreader, PlusJakartaSans)
     di atas latar garis buku, kisi kotak, tekstur kertas tua, dan guratan dinding.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
import run_eval as re_base
from schema_v4 import compute_dhash64, compute_phash64, compute_sha256_file


LATIN_CHARS_POOL = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?()-/+=#@&")


def _box_iou_list(box, gt_boxes):
    for g in gt_boxes:
        if re_base.box_iou(box, g) > 0.05:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-fp-per-img", type=int, default=12)
    p.add_argument("--n-ofl-latin", type=int, default=650)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    out_img_dir = os.path.join(ROOT_DIR, "dataset", "external", "non_aksara_distractors_v4", "images", "train_hard_sq")
    os.makedirs(out_img_dir, exist_ok=True)
    out_manifest = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4", "stage2_hard_neg_train_only.jsonl")

    meta = json.load(open(os.path.join(ROOT_DIR, "app", "web_camscanner", "yolo_meta.json"), encoding="utf-8"))
    sess = ort.InferenceSession(os.path.join(ROOT_DIR, "app", "web_camscanner", "aksara_yolo.onnx"), providers=["CPUExecutionProvider"])

    synth_manifest = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4", "synth_v4.jsonl")
    train_negs, train_pos = [], []
    with open(synth_manifest, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != "train":
                continue
            if not r.get("boxes"):
                train_negs.append(r)
            else:
                train_pos.append(r)

    records = []
    idx_cnt = 0

    def _save_patch(bgr_crop, tag, prefix):
        nonlocal idx_cnt
        if bgr_crop is None or bgr_crop.shape[0] < 8 or bgr_crop.shape[1] < 8:
            return
        fn = f"{prefix}_{idx_cnt:05d}.jpg"
        idx_cnt += 1
        out_p = os.path.join(out_img_dir, fn)
        cv2.imwrite(out_p, bgr_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "image_id": f"{prefix}_{idx_cnt:05d}",
            "source": "stage2_mined_train_only",
            "distractor_class": tag,
            "license": "SIL OFL 1.1 / CC0",
            "split": "train",
            "writer_id": f"s2neg_{prefix}_{idx_cnt}",
            "cluster_id": f"s2neg_{prefix}_{idx_cnt}",
            "phash": compute_phash64(bgr_crop),
            "dhash": compute_dhash64(bgr_crop),
            "image": f"dataset/external/non_aksara_distractors_v4/images/train_hard_sq/{fn}",
            "sha256": compute_sha256_file(out_p),
        })

    # 1. Tambang seluruh False Positive detektor pada 155 halaman negatif murni TRAIN
    print(f"[MINE HARD NEG] Menambang FP detektor pada {len(train_negs)} halaman negatif train...")
    for r in train_negs:
        bgr = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Cek di resolusi langsung 416px & tiles agar semua skala FP tertangkap
        dets, _ = re_base.run_onnx_letterbox(sess, meta, rgb, imgsz=meta.get("imgsz", 416), conf_thr=0.16)
        dets = sorted(dets, key=lambda d: d["confidence"], reverse=True)[:args.max_fp_per_img]
        ih, iw = bgr.shape[:2]
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            bw, bh = x1 - x0, y1 - y0
            px0 = max(0, int(round(x0 - 0.12 * bw)))
            py0 = max(0, int(round(y0 - 0.12 * bh)))
            px1 = min(iw, int(round(x1 + 0.12 * bw)))
            py1 = min(ih, int(round(y1 + 0.12 * bh)))
            tag = str(r.get("mode", "fp_neg")).replace("neg_", "mined_fp_")
            _save_patch(bgr[py0:py1, px0:px1], tag, "fpneg")

    # 2. Tambang FP luar-kotak-GT pada sampel citra positif synth_v4 TRAIN (teks Latin tercampur & garis pinggir)
    rng.shuffle(train_pos)
    print(f"[MINE HARD NEG] Menambang FP luar-GT pada 120 halaman positif train...")
    for r in train_pos[:120]:
        bgr = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ih, iw = bgr.shape[:2]
        gt_px = [
            [(b["bbox"][0] - b["bbox"][2] / 2) * iw, (b["bbox"][1] - b["bbox"][3] / 2) * ih,
             (b["bbox"][0] + b["bbox"][2] / 2) * iw, (b["bbox"][1] + b["bbox"][3] / 2) * ih]
            for b in r["boxes"]
        ]
        dets, _ = re_base.run_onnx_letterbox(sess, meta, rgb, imgsz=meta.get("imgsz", 416), conf_thr=0.18)
        kept_fp = 0
        for d in dets:
            if not _box_iou_list(d["box"], gt_px):
                x0, y0, x1, y1 = d["box"]
                bw, bh = x1 - x0, y1 - y0
                px0, py0 = max(0, int(round(x0 - 0.1 * bw))), max(0, int(round(y0 - 0.1 * bh)))
                px1, py1 = min(iw, int(round(x1 + 0.1 * bw))), min(ih, int(round(y1 + 0.1 * bh)))
                _save_patch(bgr[py0:py1, px0:px1], "mined_bg_outbox", "fppos")
                kept_fp += 1
                if kept_fp >= 4:
                    break

    # 3. Irisan persegi dari strip non_aksara_distractors_v4 (SPLIT=TRAIN)
    neg_dist_m = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4", "non_aksara_distractors_v4.jsonl")
    with open(neg_dist_m, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != "train":
                continue
            bgr = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            if w > h * 1.4:
                for _ in range(2):
                    win = min(w, int(round(h * rng.uniform(0.85, 1.55))))
                    x0 = rng.randint(0, max(0, w - win))
                    _save_patch(bgr[:, x0:x0 + win], f"sq_{r.get('distractor_class', 'latin')}", "sqstrip")

    # 4. Render 650 karakter-tunggal/pasangan Latin SIL OFL + garis buku/retakan dinding (persegi)
    fonts_dir = os.path.join(ROOT_DIR, "dataset", "external", "fonts_latin_ofl")
    font_files = [os.path.join(fonts_dir, fn) for fn in sorted(os.listdir(fonts_dir)) if fn.lower().endswith((".ttf", ".otf"))]
    print(f"[MINE HARD NEG] Merender {args.n_ofl_latin} patch persegi karakter Latin OFL & garis...")
    for k in range(args.n_ofl_latin):
        sz = rng.randint(44, 76)
        base_tone = rng.randint(205, 248)
        bg = np.full((sz, sz, 3), base_tone, dtype=np.uint8)
        # Tekstur kertas / bayangan gradien / garis buku / dinding
        noise = np_rng.normal(0, rng.uniform(2, 10), (sz, sz, 3))
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.random() < 0.55:
            y_ln = rng.randint(4, sz - 5)
            col_ln = (rng.randint(150, 210), rng.randint(150, 200), rng.randint(140, 195))
            cv2.line(bg, (0, y_ln), (sz, y_ln + rng.randint(-3, 3)), col_ln, rng.randint(1, 2))
        if rng.random() < 0.30:
            x_ln = rng.randint(4, sz - 5)
            cv2.line(bg, (x_ln, 0), (x_ln + rng.randint(-3, 3), sz), (160, 160, 160), 1)

        pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        txt = "".join(rng.choice(LATIN_CHARS_POOL) for _ in range(rng.choice([1, 1, 2])))
        if font_files and rng.random() < 0.85:
            fp = rng.choice(font_files)
            fnt = ImageFont.truetype(fp, rng.randint(int(sz * 0.42), int(sz * 0.72)))
            ink = rng.choice([(25, 25, 30), (30, 60, 150), (140, 35, 35), (55, 55, 55)])
            ox, oy = rng.randint(2, max(3, sz // 3)), rng.randint(0, max(2, sz // 3))
            draw.text((ox, oy), txt, font=fnt, fill=ink)
            tag = "synth_latin_char_sq"
        else:
            # Garis retak dinding / coretan non-aksara murni
            for _ in range(rng.randint(1, 3)):
                p1 = (rng.randint(0, sz), rng.randint(0, sz))
                p2 = (rng.randint(0, sz), rng.randint(0, sz))
                c = rng.randint(40, 120)
                draw.line([p1, p2], fill=(c, c, c), width=rng.randint(1, 3))
            tag = "synth_wall_line_sq"

        out_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        if rng.random() < 0.2:
            out_bgr = cv2.GaussianBlur(out_bgr, (3, 3), rng.uniform(0.4, 1.0))
        _save_patch(out_bgr, tag, "ofllat")

    with open(out_manifest, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE HARD NEG] saved {len(records)} square hard-negative train patches -> {out_manifest}")


if __name__ == "__main__":
    main()
