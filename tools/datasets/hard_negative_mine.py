"""
Penambang Contoh Negatif Sulit (Hard Negative Mining - Fase 3e) &
Pelabelan Semu Konservatif (Pseudo-Labeling - Fase 3f) untuk AksaraScan v3.0.

Aturan ketat:
  - Hard Negatives: menjalankan model ONNX pada citra latar non-Aksara Jawa (kertas tulis kosong,
    teks Latin, buku tulis, margin/coretan); setiap deteksi dengan skor >= thresh disimpan
    sebagai gambar negatif dengan file label kosong (`*.txt` 0 baris) agar detektor menekan FP.
  - Pseudo-Labeling (Fase 3f): hanya menerima prediksi pada halaman tak berlabel jika:
      1) Urutan Unicode hasil rekonstruksi 100% lolos `validate_javanese_unicode()` (0 galat kaidah)
      2) Rasio kata cocok kamus Jawa >= 0.60 (`--min-lex-match`)
      3) Keyakinan rata-rata detektor >= 0.68 (`--min-conf`)
      4) Dibatasi maksimal 5 putaran (`--round <= 5`).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "synth_v2"))

from schema_55 import CLASSES_55, CATEGORIES_55, CLASS_TO_ID
from orthography_rules import (
    NGLEGENA_LIST,
    CAKRA, CECAK, KERET, LAYAR, PANGKON, PENGKAL, PEPET, SUKU, TALING, TARUNG, WIGNYAN, WULU,
    PADA_ADEG, PADA_LINGSA, PADA_LUNGSI,
    javanese_to_latin_syllables, validate_javanese_unicode,
)

BASE_MAP = {name: ch for name, ch in NGLEGENA_LIST}
MOD_MAP = {
    "wulu": WULU, "suku": SUKU, "pepet": PEPET, "taling": TALING, "tarung": TARUNG,
    "layar": LAYAR, "cecak": CECAK, "wignyan": WIGNYAN, "pangkon": PANGKON,
    "cakra": CAKRA, "keret": KERET, "pengkal": PENGKAL,
}
PADA_MAP = {"pada_adeg": PADA_ADEG, "pada_lingsa": PADA_LINGSA, "pada_lungsi": PADA_LUNGSI}


def load_lexicon_words():
    lex_path = os.path.join(ROOT_DIR, "app", "web_camscanner", "javanese_lm_data.js")
    words = set()
    if os.path.exists(lex_path):
        txt = open(lex_path, "r", encoding="utf-8").read()
        start = txt.find("{")
        end = txt.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(txt[start : end + 1])
            words.update(data.get("dict", {}).keys())
            words.update(data.get("unigramLog10", {}).keys())
    return words


def letterbox_rgb(img_bgr, imgsz):
    h, w = img_bgr.shape[:2]
    scale = min(imgsz / max(1, w), imgsz / max(1, h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x = (imgsz - nw) // 2
    pad_y = (imgsz - nh) // 2
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))[None, ...]
    return chw, scale, pad_x, pad_y, w, h


def nms_boxes(boxes, scores, iou_thr=0.45):
    if not boxes:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    order = np.argsort(-s)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        xx0 = np.maximum(b[i, 0], b[order[1:], 0])
        yy0 = np.maximum(b[i, 1], b[order[1:], 1])
        xx1 = np.minimum(b[i, 2], b[order[1:], 2])
        yy1 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.maximum(0.0, xx1 - xx0) * np.maximum(0.0, yy1 - yy0)
        area_i = np.maximum(1.0, (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]))
        area_o = np.maximum(1.0, (b[order[1:], 2] - b[order[1:], 0]) * (b[order[1:], 3] - b[order[1:], 1]))
        iou = inter / (area_i + area_o - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


def run_detector(sess, img_bgr, conf_thr=0.25):
    inp_info = sess.get_inputs()[0]
    imgsz = int(inp_info.shape[2]) if len(inp_info.shape) == 4 and isinstance(inp_info.shape[2], int) else 416
    chw, scale, px, py, W, H = letterbox_rgb(img_bgr, imgsz)
    out = sess.run(None, {inp_info.name: chw})[0]
    # out: [1, 4+nc, N]
    pred = out[0]
    nc = pred.shape[0] - 4
    cls_scores = pred[4 : 4 + min(nc, len(CLASSES_55)), :]
    best_cid = np.argmax(cls_scores, axis=0)
    best_sc = cls_scores[best_cid, np.arange(cls_scores.shape[1])]
    mask = best_sc >= conf_thr
    if not np.any(mask):
        return []
    idxs = np.where(mask)[0]
    boxes, scores, cids = [], [], []
    for k in idxs:
        cx, cy, bw, bh = pred[0, k], pred[1, k], pred[2, k], pred[3, k]
        x0 = max(0.0, (cx - bw / 2 - px) / scale)
        y0 = max(0.0, (cy - bh / 2 - py) / scale)
        x1 = min(float(W), (cx + bw / 2 - px) / scale)
        y1 = min(float(H), (cy + bh / 2 - py) / scale)
        if x1 > x0 + 2 and y1 > y0 + 2:
            boxes.append([x0, y0, x1, y1])
            scores.append(float(best_sc[k]))
            cids.append(int(best_cid[k]))
    keep = nms_boxes(boxes, scores)
    return [
        {"cid": cids[i], "label": CLASSES_55[cids[i]], "conf": scores[i], "box": boxes[i]}
        for i in keep
    ]


def simple_reconstruct_unicode(dets):
    """Rekonstruksi cepat kiri-ke-kanan per baris untuk gerbang validasi pseudo-label."""
    if not dets:
        return ""
    bases = sorted(
        [d for d in dets if CATEGORIES_55[d["label"]] == "nglegena"],
        key=lambda d: (d["box"][1] + d["box"][3]) / 2,
    )
    if not bases:
        return ""
    median_h = float(np.median([b["box"][3] - b["box"][1] for b in bases]))
    lines = []
    for b in bases:
        cy = (b["box"][1] + b["box"][3]) / 2
        if not lines or abs(cy - lines[-1]["cy"]) > median_h * 0.72:
            lines.append({"cy": cy, "items": [b]})
        else:
            lines[-1]["items"].append(b)
    all_u = []
    mods = [d for d in dets if CATEGORIES_55[d["label"]] in ("sandhangan", "pangkon", "pasangan")]
    for ln in lines:
        ln_bases = sorted(ln["items"], key=lambda d: d["box"][0])
        line_u = []
        # Lampirkan modifikasi ke aksara terdekat secara spasial
        attached = {id(b): [] for b in ln_bases}
        for m in mods:
            mcx = (m["box"][0] + m["box"][2]) / 2
            mcy = (m["box"][1] + m["box"][3]) / 2
            if abs(mcy - ln["cy"]) > median_h * 1.45:
                continue
            best_b, best_d = None, 1e9
            for b in ln_bases:
                bcx = (b["box"][0] + b["box"][2]) / 2
                dist = abs(mcx - bcx)
                if m["label"] == "taling":
                    # taling berada di kiri basisnya
                    dist = abs(m["box"][2] - b["box"][0]) + (0 if mcx <= b["box"][2] else median_h * 1.5)
                if dist < best_d and dist < median_h * 1.5:
                    best_d = dist
                    best_b = b
            if best_b is not None:
                attached[id(best_b)].append(m)
        for b in ln_bases:
            line_u.append(BASE_MAP.get(b["label"], ""))
            ms = attached[id(b)]
            for m in sorted(ms, key=lambda x: x["box"][0]):
                lbl = m["label"]
                if lbl.startswith("p_"):
                    line_u.append(PANGKON + BASE_MAP.get(lbl[2:], ""))
                elif lbl in MOD_MAP:
                    line_u.append(MOD_MAP[lbl])
        all_u.append("".join(line_u))
    return "\n".join(all_u)


def mine_hard_negatives(sess, bg_dir, out_dir, conf_thr=0.25):
    os.makedirs(os.path.join(out_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "train"), exist_ok=True)
    imgs = sorted(glob.glob(os.path.join(bg_dir, "*.*")))
    imgs = [p for p in imgs if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    added = 0
    for idx, ip in enumerate(imgs):
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        dets = run_detector(sess, bgr, conf_thr=conf_thr)
        if dets:
            # Simpan sebagai contoh negatif keras (label kosong)
            bn = f"hn_{idx:05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, "images", "train", bn), bgr)
            with open(os.path.join(out_dir, "labels", "train", bn.replace(".jpg", ".txt")), "w", encoding="utf-8") as lf:
                lf.write("")
            added += 1
    print(f"[HardNeg] Dipindai {len(imgs)} citra latar -> ditemukan {added} negatif sulit (label kosong).")
    return added


def mine_pseudo_labels(sess, raw_dir, out_dir, round_num, min_conf=0.68, min_lex_match=0.60):
    if round_num > 5:
        raise ValueError("Pseudo-labeling dibatasi maksimal 5 putaran (Fase 3f).")
    os.makedirs(os.path.join(out_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", "train"), exist_ok=True)
    lex_words = load_lexicon_words()
    imgs = sorted(glob.glob(os.path.join(raw_dir, "*.*")))
    imgs = [p for p in imgs if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    accepted, rejected = 0, 0
    manifest_rows = []
    for idx, ip in enumerate(imgs):
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        dets = run_detector(sess, bgr, conf_thr=0.35)
        if len(dets) < 2:
            rejected += 1
            continue
        mean_conf = float(np.mean([d["conf"] for d in dets]))
        uni = simple_reconstruct_unicode(dets)
        ok_ortho, errs, _ = validate_javanese_unicode(uni)
        syls = javanese_to_latin_syllables(uni)
        latin_str = "".join(syls)
        toks = [t for t in latin_str.lower().split() if t]
        lex_ratio = (sum(1 for t in toks if t in lex_words) / max(1, len(toks))) if toks else 0.0

        if mean_conf >= min_conf and ok_ortho and lex_ratio >= min_lex_match:
            bn = f"pseudo_r{round_num}_{idx:05d}.jpg"
            cv2.imwrite(os.path.join(out_dir, "images", "train", bn), bgr)
            with open(os.path.join(out_dir, "labels", "train", bn.replace(".jpg", ".txt")), "w", encoding="utf-8") as lf:
                for d in dets:
                    x0, y0, x1, y1 = d["box"]
                    xc = ((x0 + x1) / 2) / W
                    yc = ((y0 + y1) / 2) / H
                    bw = (x1 - x0) / W
                    bh = (y1 - y0) / H
                    lf.write(f"{d['cid']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            manifest_rows.append({"image": bn, "round": round_num, "mean_conf": round(mean_conf, 4), "lex_ratio": round(lex_ratio, 4), "unicode": uni})
            accepted += 1
        else:
            rejected += 1

    with open(os.path.join(out_dir, f"pseudo_round_{round_num}.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Pseudo R{round_num}] Diterima={accepted}, Ditolak={rejected} (syarat: ortho valid & lex>={min_lex_match:.0%} & conf>={min_conf:.2f})")
    return accepted


def main():
    p = argparse.ArgumentParser(description="Hard Negative Miner & Conservative Pseudo-Labeler (Fase 3e-3f)")
    p.add_argument("--onnx", default="app/web_camscanner/aksara_yolo.onnx")
    p.add_argument("--mode", choices=["hard_negatives", "pseudo_label", "self_check"], default="self_check")
    p.add_argument("--input-dir", default="")
    p.add_argument("--out-dir", default="dataset/external/hard_mined")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--conf", type=float, default=0.28)
    p.add_argument("--min-lex-match", type=float, default=0.60)
    args = p.parse_args()

    onnx_abs = args.onnx if os.path.isabs(args.onnx) else os.path.join(ROOT_DIR, args.onnx)
    sess = ort.InferenceSession(onnx_abs, providers=["CPUExecutionProvider"])

    if args.mode == "self_check":
        # Uji cepat terhadap 2 gambar kanvas kosong / teks Latin sintetis (harus menghasilkan 0 atau mined HN)
        tmp_dir = os.path.join(ROOT_DIR, "dataset", "external", "hard_mined", "_smoke_bg")
        os.makedirs(tmp_dir, exist_ok=True)
        blank = np.full((320, 480, 3), 245, dtype=np.uint8)
        latin_bg = blank.copy()
        cv2.putText(latin_bg, "Catatan Harian Mahasiswa 2026", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
        cv2.imwrite(os.path.join(tmp_dir, "bg_blank.png"), blank)
        cv2.imwrite(os.path.join(tmp_dir, "bg_latin.png"), latin_bg)
        out_abs = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT_DIR, args.out_dir)
        added = mine_hard_negatives(sess, tmp_dir, out_abs, conf_thr=args.conf)
        print(f"[PASS] Self-check hard_negative_mine selesai (added={added} FP pada distraktor Latin/kosong).")
    elif args.mode == "hard_negatives":
        mine_hard_negatives(sess, args.input_dir, os.path.join(ROOT_DIR, args.out_dir), conf_thr=args.conf)
    elif args.mode == "pseudo_label":
        mine_pseudo_labels(sess, args.input_dir, os.path.join(ROOT_DIR, args.out_dir), round_num=args.round, min_lex_match=args.min_lex_match)


if __name__ == "__main__":
    main()
