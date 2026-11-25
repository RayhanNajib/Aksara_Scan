"""
Harness Evaluasi Baseline & Paritas Python-JS AksaraScan (ONNX Runtime CPU)
Menjalankan model ONNX yang sama dengan browser (`app/web_camscanner/aksara_yolo.onnx`)
untuk:
  1. Uji Paritas Python vs JS (`--parity eval/parity_20.jsonl`) pada 20 gambar emas
     (membandingkan tensor mentah `onnxruntime` CPU vs `onnxruntime-web` WASM dan hasil browser).
  2. Evaluasi Baseline Deteksi pada `dataset_yolo/val` (`--yolo-val <dir>`):
     - mAP50 dan mAP50-95 per kelompok (nglegena, pasangan, sandhangan, pada)
     - Recall per kelas untuk objek kecil (< 16 px pada skala 416px & asli)
     - Pasangan kelas paling sering tertukar & matriks konfusi
     - Grapheme F1 & CER cluster (GT box vs Pred box)
     - Galeri 50 kesalahan terburuk (HTML + visualisasi kotak GT hijau vs Pred merah)
  3. Evaluasi End-to-End Set Diagnostik (`--failures eval/failures/manifest.jsonl`):
     - Grapheme F1, Unicode CER, Akurasi Kata, dan Cakupan Leksikon/Terjemahan
     - Taksonomi kesalahan (deteksi, klasifikasi, perakitan, pemisahan kata, terjemahan)

Penggunaan:
  python eval/run_eval.py --all --yolo-val ../Projek_Akhir/dataset_yolo --out-dir eval/reports
"""
import argparse
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT_DIR, "app", "web_camscanner")
META_PATH = os.path.join(WEB_DIR, "yolo_meta.json")
ONNX_PATH = os.path.join(WEB_DIR, "aksara_yolo.onnx")


def load_meta():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utilitas Metrik Teks: Levenshtein CER, Grapheme Tokenizer, Word Accuracy
# ---------------------------------------------------------------------------

def levenshtein(seq_a, seq_b):
    la, lb = len(seq_a), len(seq_b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = seq_a[i - 1]
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if ca == seq_b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def strip_ws_for_cer(s):
    if not s or s == "—":
        return ""
    return re.sub(r"\s+", "", s)


def compute_cer(gt_str, pred_str):
    g = strip_ws_for_cer(gt_str)
    p = strip_ws_for_cer(pred_str)
    if not g:
        return 0.0 if not p else 1.0
    return levenshtein(g, p) / len(g)


def split_javanese_graphemes(uni_str):
    """Pecah string Unicode Blok Jawa menjadi daftar grafem/kluster suku kata atau pada."""
    if not uni_str or uni_str == "—":
        return []
    s = re.sub(r"\s+", "", uni_str)
    PADA_CHARS = set("\uA9C1\uA9C2\uA9C3\uA9C4\uA9C5\uA9C6\uA9C7\uA9C8\uA9C9\uA9CA\uA9CB\uA9CC\uA9CD\uA9CF\uA9DE\uA9DF")
    PANGKON = "\uA9C0"
    COMBINING = set([
        "\uA980", "\uA981", "\uA982", "\uA983",
        "\uA9B3", "\uA9B4", "\uA9B5", "\uA9B6", "\uA9B7", "\uA9B8",
        "\uA9B9", "\uA9BA", "\uA9BB", "\uA9BC", "\uA9BD", "\uA9BE", "\uA9BF",
    ])
    BASE_MIN, BASE_MAX = 0xA984, 0xA9B2

    clusters = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch in PADA_CHARS:
            clusters.append(ch)
            i += 1
            continue
        if BASE_MIN <= ord(ch) <= BASE_MAX:
            cl = [ch]
            i += 1
            # Consume pasangan chains: pangkon + base (unless end of word -> final pangkon stays in same cluster)
            while i < n:
                nxt = s[i]
                if nxt == PANGKON:
                    cl.append(nxt)
                    i += 1
                    if i < n and BASE_MIN <= ord(s[i]) <= BASE_MAX:
                        cl.append(s[i])
                        i += 1
                        continue
                elif nxt in COMBINING:
                    cl.append(nxt)
                    i += 1
                else:
                    break
            clusters.append("".join(cl))
        else:
            clusters.append(ch)
            i += 1
    return clusters


def compute_grapheme_f1(gt_uni, pred_uni):
    gt_g = split_javanese_graphemes(gt_uni)
    pr_g = split_javanese_graphemes(pred_uni)
    if not gt_g and not pr_g:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0, "gt_count": 0, "pred_count": 0, "matches": 0}
    if not gt_g or not pr_g:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "gt_count": len(gt_g), "pred_count": len(pr_g), "matches": 0}
    # Order-preserving LCS or multiset? Standard OCR cluster F1 via aligned LCS matches!
    la, lb = len(gt_g), len(pr_g)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            if gt_g[i - 1] == pr_g[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    matches = dp[la][lb]
    prec = matches / lb
    rec = matches / la
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "gt_count": la, "pred_count": lb, "matches": matches}


def normalize_latin_words(s):
    if not s or s == "—":
        return []
    # Lowercase, normalize é/è/ê optionally? Keep strict AND phonetic variant check!
    s_clean = re.sub(r"[,.;:!?\"'()]+", " ", s.lower())
    return [w for w in s_clean.split() if w]


def compute_word_accuracy(gt_lat, pred_lat):
    gw = normalize_latin_words(gt_lat)
    pw = normalize_latin_words(pred_lat)
    if not gw:
        return {"word_acc": 1.0 if not pw else 0.0, "gt_words": 0, "pred_words": len(pw), "wer": 0.0}
    dist = levenshtein(gw, pw)
    wer = dist / len(gw)
    wacc = max(0.0, 1.0 - wer)
    return {"word_acc": wacc, "wer": wer, "gt_words": len(gw), "pred_words": len(pw), "edit_dist": dist}


# ---------------------------------------------------------------------------
# Mesin ONNX Python (Single-pass 416x416 & Replika Tiling Browser app.js)
# ---------------------------------------------------------------------------

def box_iou(a, b):
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0.0 else 0.0


def yolo_nms(candidates, same_cat_iou=0.42, cross_cat_iou=0.72):
    cands = sorted(candidates, key=lambda d: d["confidence"], reverse=True)
    kept = []
    for cand in cands:
        suppress = False
        for k in kept:
            same_excl = (cand["category"] != "sandhangan" and k["category"] != "sandhangan")
            thr = same_cat_iou if (cand["category"] == k["category"] or same_excl) else cross_cat_iou
            if box_iou(cand["box"], k["box"]) > thr:
                suppress = True
                break
        if not suppress:
            kept.append(cand)
    return kept


def nms_per_class(candidates, iou_thr=0.70, max_det=300):
    by_c = defaultdict(list)
    for d in candidates:
        by_c[d["classIdx"]].append(d)
    out = []
    for cid, cands in by_c.items():
        cands.sort(key=lambda x: x["confidence"], reverse=True)
        kept = []
        for cand in cands:
            if any(box_iou(cand["box"], k["box"]) > iou_thr for k in kept):
                continue
            kept.append(cand)
        out.extend(kept)
    out.sort(key=lambda x: x["confidence"], reverse=True)
    return out[:max_det]


def run_onnx_letterbox(session, meta, img_rgb, imgsz=416, conf_thr=None, dump_path_prefix=None, nms_mode="browser"):
    orig_h, orig_w = img_rgb.shape[:2]
    scale = min(imgsz / orig_w, imgsz / orig_h)
    nw = int(round(orig_w * scale))
    nh = int(round(orig_h * scale))
    pad_x = int(math.floor((imgsz - nw) / 2))
    pad_y = int(math.floor((imgsz - nh) / 2))

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    chw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    inp_name = session.get_inputs()[0].name
    raw_out = session.run(None, {inp_name: chw})[0]  # (1, 4+nc, num_anchors)

    if dump_path_prefix:
        chw.astype(np.float32).tofile(dump_path_prefix + "_input.bin")
        raw_out.astype(np.float32).tofile(dump_path_prefix + "_output.bin")

    arr = raw_out[0]
    nc = len(meta["classes"])
    num_anchors = arr.shape[1]
    thr = meta.get("conf_thresh", 0.25) if conf_thr is None else conf_thr
    iou_thr = meta.get("iou_thresh", 0.42)

    cls_scores = arr[4:4 + nc, :]
    if nms_mode == "per_class_ap":
        # Multi-label / class-wise extraction above thr like Ultralytics val
        ys_c, xs_a = np.where(cls_scores > thr)
        cands = []
        for c_np, a_np in zip(ys_c, xs_a):
            c = int(c_np)
            a = int(a_np)
            sc = float(cls_scores[c, a])
            cx, cy, bw, bh = float(arr[0, a]), float(arr[1, a]), float(arr[2, a]), float(arr[3, a])
            x0 = max(0.0, (cx - bw / 2.0 - pad_x) / scale)
            y0 = max(0.0, (cy - bh / 2.0 - pad_y) / scale)
            x1 = min(float(orig_w), (cx + bw / 2.0 - pad_x) / scale)
            y1 = min(float(orig_h), (cy + bh / 2.0 - pad_y) / scale)
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            lbl = meta["classes"][c]
            cands.append({
                "classIdx": c, "label": lbl, "category": meta["categories"][c],
                "unicode": meta["unicode"].get(lbl, ""), "confidence": sc,
                "box": [x0, y0, x1, y1],
            })
        return nms_per_class(cands, iou_thr=0.70, max_det=300), raw_out

    best_cls = np.argmax(cls_scores, axis=0)
    best_sc = cls_scores[best_cls, np.arange(num_anchors)]
    keep_idx = np.where(best_sc > thr)[0]

    cands = []
    for a in keep_idx:
        c = int(best_cls[a])
        sc = float(best_sc[a])
        cx, cy, bw, bh = float(arr[0, a]), float(arr[1, a]), float(arr[2, a]), float(arr[3, a])
        x0 = max(0.0, (cx - bw / 2.0 - pad_x) / scale)
        y0 = max(0.0, (cy - bh / 2.0 - pad_y) / scale)
        x1 = min(float(orig_w), (cx + bw / 2.0 - pad_x) / scale)
        y1 = min(float(orig_h), (cy + bh / 2.0 - pad_y) / scale)
        if x1 - x0 < 3.0 or y1 - y0 < 3.0:
            continue
        lbl = meta["classes"][c]
        cands.append({
            "classIdx": c,
            "label": lbl,
            "category": meta["categories"][c],
            "unicode": meta["unicode"].get(lbl, ""),
            "confidence": sc,
            "box": [x0, y0, x1, y1],
        })
    return yolo_nms(cands, iou_thr, 0.72), raw_out


# Replika Tiling & Binerisasi app.js untuk uji paritas ujung-ke-ujung
def calculate_otsu_py(gray):
    h, w = gray.shape
    hist = np.bincount(gray.ravel(), minlength=256)
    ys, ye = int(h * 0.12), int(h * 0.88)
    xs, xe = int(w * 0.12), int(w * 0.88)
    sub = gray[ys:ye, xs:xe]
    if sub.size > 500:
        thist = np.bincount(sub.ravel(), minlength=256).astype(np.float64)
        total = float(sub.size)
    else:
        thist = hist.astype(np.float64)
        total = float(gray.size)
    total_sum = float(np.dot(np.arange(256, dtype=np.float64), thist))
    sum_b = 0.0
    w_b = 0.0
    best_var = -1.0
    thr = 127
    for t in range(256):
        w_b += thist[t]
        if w_b == 0.0:
            continue
        w_f = total - w_b
        if w_f == 0.0:
            break
        sum_b += t * thist[t]
        m_b = sum_b / w_b
        m_f = (total_sum - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            thr = t
    return thr


def adaptive_binary_py(gray, global_t):
    h, w = gray.shape
    s = max(15, round(min(w, h) / 12))
    k = 2 * s + 1
    mean_local = cv2.boxFilter(gray.astype(np.float64), ddepth=-1, ksize=(k, k), borderType=cv2.BORDER_REPLICATE)
    # Match integral box near borders exactly via integral image
    ii = np.zeros((h + 1, w + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(gray.astype(np.float64), axis=0), axis=1)
    ys = np.arange(h)
    xs = np.arange(w)
    y0 = np.maximum(0, ys - s)
    y1 = np.minimum(h - 1, ys + s)
    x0 = np.maximum(0, xs - s)
    x1 = np.minimum(w - 1, xs + s)
    area = np.outer(y1 - y0 + 1, x1 - x0 + 1)
    s_box = (
        ii[y1[:, None] + 1, x1[None, :] + 1]
        - ii[y0[:, None], x1[None, :] + 1]
        - ii[y1[:, None] + 1, x0[None, :]]
        + ii[y0[:, None], x0[None, :]]
    )
    th = np.minimum(float(global_t), (s_box / area) * 0.88)
    return (gray.astype(np.float64) < th).astype(np.uint8)


def clear_border_noise_py(bw):
    h, w = bw.shape
    num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = bw.copy()
    for i in range(1, num):
        x, y, bw_, bh_, area = stats[i]
        touches = (x <= 1 or y <= 1 or (x + bw_) >= w - 1 or (y + bh_) >= h - 1)
        if touches and (area > 1500 or area > w * h * 0.02):
            out[lab == i] = 0
    return out


def apply_median3x3_py(bw):
    ones = cv2.boxFilter(bw.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    ones = np.round(ones).astype(np.int32)
    out = np.where(bw > 0, (ones >= 1).astype(np.uint8), (ones >= 7).astype(np.uint8))
    out[0, :] = 0
    out[-1, :] = 0
    out[:, 0] = 0
    out[:, -1] = 0
    return out


def apply_morphology_py(bw):
    h, w = bw.shape
    k_len = max(13, min(31, round(w * 0.025))) | 1
    ker_h = np.ones((1, k_len), np.uint8)
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker_h, borderType=cv2.BORDER_CONSTANT)
    clean = np.where((bw > 0) & (lines == 0), np.uint8(1), np.uint8(0))
    closed = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT)
    closed[0, :] = 0
    closed[-1, :] = 0
    closed[:, 0] = 0
    closed[:, -1] = 0
    return closed


# Perakit ortografi Python identik orthography.js untuk membangun cluster & Unicode dari daftar deteksi box
JAVANESE_UNI = {
    "ha": "\uA9B2", "na": "\uA9A4", "ca": "\uA995", "ra": "\uA9AB", "ka": "\uA98F",
    "da": "\uA9A2", "ta": "\uA9A0", "sa": "\uA9B1", "wa": "\uA9AE", "la": "\uA9AD",
    "pa": "\uA9A5", "dha": "\uA99D", "ja": "\uA997", "ya": "\uA9AA", "nya": "\uA99A",
    "ma": "\uA9A9", "ga": "\uA992", "ba": "\uA9A7", "tha": "\uA99B", "nga": "\uA994",
    "wulu": "\uA9B6", "suku": "\uA9B8", "pepet": "\uA9BC", "taling": "\uA9BA", "tarung": "\uA9B4",
    "layar": "\uA982", "cecak": "\uA981", "wignyan": "\uA983", "pangkon": "\uA9C0",
    "cakra": "\uA9BF", "keret": "\uA9BD", "pengkal": "\uA9BE",
    "pada_adeg": "\uA9CB", "pada_lingsa": "\uA9C8", "pada_lungsi": "\uA9C9",
}


def box_metrics_py(box):
    x0, y0, x1, y1 = box
    return {
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "w": max(1.0, x1 - x0), "h": max(1.0, y1 - y0),
        "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0,
    }


def reconstruct_unicode_py(detections):
    """Porting exact logika orthography.js ke Python untuk evaluasi cluster/CER."""
    if not detections:
        return "—", ""
    anchors, modifiers = [], []
    for idx, d in enumerate(detections):
        item = dict(d)
        item["m"] = box_metrics_py(d["box"])
        if d["category"] in ("nglegena", "pada"):
            anchors.append({
                "isPada": d["category"] == "pada",
                "base": item,
                "pasangan": None,
                "sandhangans": [],
            })
        else:
            modifiers.append(item)

    if not anchors:
        srt = sorted(modifiers, key=lambda s: (s["m"]["cy"], s["m"]["x0"]))
        return " ".join(JAVANESE_UNI.get(s["label"], s.get("unicode", "")) for s in srt), " ".join(s["label"] for s in srt)

    base_only = [a for a in anchors if not a["isPada"]]
    ref_list = base_only if base_only else anchors
    avg_base_h = sum(b["base"]["m"]["h"] for b in ref_list) / len(ref_list)
    anchors.sort(key=lambda a: a["base"]["m"]["cy"])

    lines = []
    for cl in anchors:
        placed = False
        for line in lines:
            pool = [it for it in line if not it["isPada"]] or line
            avg_y = sum(it["base"]["m"]["cy"] for it in pool) / len(pool)
            if abs(cl["base"]["m"]["cy"] - avg_y) < max(24.0, avg_base_h * 1.05):
                line.append(cl)
                placed = True
                break
        if not placed:
            lines.append([cl])

    lines.sort(key=lambda l: sum(it["base"]["m"]["cy"] for it in l) / len(l))
    for line in lines:
        line.sort(key=lambda a: a["base"]["m"]["x0"])

    syl_clusters = [c for line in lines for c in line if not c["isPada"]]
    rem_mods = []
    for mod in modifiers:
        if mod["category"] == "pasangan":
            best_host, best_score = None, float("inf")
            is_right = mod["label"] in ("p_ha", "p_sa", "p_pa")
            for cl in syl_clusters:
                bm = cl["base"]["m"]
                dx = mod["m"]["cx"] - bm["cx"]
                dy = mod["m"]["cy"] - bm["cy"]
                if is_right:
                    if mod["m"]["x0"] >= bm["x0"] and mod["m"]["x0"] - bm["x1"] < bm["w"] * 1.4 and abs(dy) < bm["h"] * 0.9:
                        dist = math.hypot(mod["m"]["x0"] - bm["x1"], dy)
                        if dist < best_score:
                            best_score, best_host = dist, cl
                else:
                    h_ov = max(0.0, min(mod["m"]["x1"], bm["x1"]) - max(mod["m"]["x0"], bm["x0"]))
                    if 0 < dy < bm["h"] * 2.8 and (h_ov > 0 or abs(dx) < bm["w"] * 0.9):
                        dist = math.hypot(dx * 1.5, dy)
                        if dist < best_score:
                            best_score, best_host = dist, cl
            if best_host is not None and best_host["pasangan"] is None:
                best_host["pasangan"] = mod
            else:
                rem_mods.append(mod)
        else:
            rem_mods.append(mod)

    for mod in rem_mods:
        if mod["category"] != "sandhangan":
            continue
        best_host, best_score = None, float("inf")
        lbl = mod["label"]
        for cl in syl_clusters:
            bm = cl["base"]["m"]
            fa = box_metrics_py([
                min(bm["x0"], cl["pasangan"]["m"]["x0"]),
                min(bm["y0"], cl["pasangan"]["m"]["y0"]),
                max(bm["x1"], cl["pasangan"]["m"]["x1"]),
                max(bm["y1"], cl["pasangan"]["m"]["y1"]),
            ]) if cl["pasangan"] else bm
            v_ov = max(0.0, min(mod["m"]["y1"], fa["y1"]) - max(mod["m"]["y0"], fa["y0"]))
            if lbl == "taling":
                gap_l = bm["x0"] - mod["m"]["x1"]
                if mod["m"]["cx"] < bm["cx"] and -bm["w"] * 0.5 < gap_l < bm["w"] * 1.3 and v_ov > 0:
                    sc = abs(gap_l) + abs(mod["m"]["cy"] - bm["cy"]) * 0.4
                    if sc < best_score:
                        best_score, best_host = sc, cl
            elif lbl in ("tarung", "wignyan", "pangkon", "pengkal"):
                gap_r = mod["m"]["x0"] - fa["x1"]
                raw_vov = min(mod["m"]["y1"], fa["y1"]) - max(mod["m"]["y0"], fa["y0"])
                if mod["m"]["cx"] > fa["x0"] and gap_r < bm["w"] * 1.4 and raw_vov >= -bm["h"] * 0.35:
                    sc = max(0.0, gap_r) + abs(mod["m"]["cy"] - fa["cy"]) * 0.5
                    if sc < best_score:
                        best_score, best_host = sc, cl
            elif lbl in ("wulu", "pepet", "layar", "cecak"):
                dy = bm["cy"] - mod["m"]["cy"]
                h_ov = max(0.0, min(mod["m"]["x1"], bm["x1"]) - max(mod["m"]["x0"], bm["x0"]))
                if 0 < dy < bm["h"] * 2.6 and (h_ov > 0 or abs(mod["m"]["cx"] - bm["cx"]) < bm["w"] * 0.95):
                    sc = math.hypot(mod["m"]["cx"] - bm["cx"], dy * 0.7)
                    if sc < best_score:
                        best_score, best_host = sc, cl
            else:
                dy = mod["m"]["cy"] - bm["y0"]
                h_ov = max(0.0, min(mod["m"]["x1"], fa["x1"]) - max(mod["m"]["x0"], fa["x0"]))
                if 0 < dy < fa["h"] * 2.8 and (h_ov > 0 or abs(mod["m"]["cx"] - fa["cx"]) < fa["w"] * 1.15):
                    sc = math.hypot(mod["m"]["cx"] - fa["cx"], abs(mod["m"]["y0"] - fa["y1"]) * 0.8)
                    if sc < best_score:
                        best_score, best_host = sc, cl
        if best_host is not None:
            best_host["sandhangans"].append(mod)

    line_uni, line_lat = [], []
    for line in lines:
        u_words, l_words = [], []
        for cl in line:
            if cl["isPada"]:
                pn = cl["base"]["label"]
                u_words.append(JAVANESE_UNI.get(pn, ""))
                if pn == "pada_lingsa":
                    l_words.append(",")
                elif pn == "pada_lungsi":
                    l_words.append(".")
                continue
            bn = cl["base"]["label"]
            uni = JAVANESE_UNI.get(bn, "")
            lat = bn
            if cl["pasangan"]:
                pr = re.sub(r"^p_", "", cl["pasangan"]["label"])
                uni += "\uA9C0" + JAVANESE_UNI.get(pr, "")
                lat = (lat[:-1] if lat.endswith("a") else lat) + pr
            mnames = {s["label"] for s in cl["sandhangans"]}
            strip_a = lambda s: s[:-1] if s.endswith("a") else s
            if "cakra" in mnames:
                uni += JAVANESE_UNI["cakra"]
                lat = strip_a(lat) + "ra"
            elif "keret" in mnames:
                uni += JAVANESE_UNI["keret"]
                lat = strip_a(lat) + "re"
            elif "pengkal" in mnames:
                uni += JAVANESE_UNI["pengkal"]
                lat = strip_a(lat) + "ya"
            if "pangkon" in mnames:
                uni += JAVANESE_UNI["pangkon"]
                lat = strip_a(lat)
            elif "taling" in mnames and "tarung" in mnames:
                uni += JAVANESE_UNI["taling"] + JAVANESE_UNI["tarung"]
                lat = strip_a(lat) + "o"
            elif "taling" in mnames:
                uni += JAVANESE_UNI["taling"]
                lat = strip_a(lat) + "é"
            elif "wulu" in mnames:
                uni += JAVANESE_UNI["wulu"]
                lat = strip_a(lat) + "i"
            elif "suku" in mnames:
                uni += JAVANESE_UNI["suku"]
                lat = strip_a(lat) + "u"
            elif "pepet" in mnames:
                uni += JAVANESE_UNI["pepet"]
                lat = strip_a(lat) + "ê"
            if "layar" in mnames:
                uni += JAVANESE_UNI["layar"]
                lat += "r"
            if "cecak" in mnames:
                uni += JAVANESE_UNI["cecak"]
                lat += "ng"
            if "wignyan" in mnames:
                uni += JAVANESE_UNI["wignyan"]
                lat += "h"
            u_words.append(uni)
            l_words.append(lat)
        line_uni.append("".join(u_words))
        line_lat.append(" ".join(l_words))
    return "\n".join(line_uni), "\n".join(line_lat)


def run_browser_replica_py(session, meta, img_bgr):
    """Replika pipeline penuh browser (maxDim 1500 -> biner -> lineBands -> YOLO ONNX -> NMS -> orthography)."""
    ih, iw = img_bgr.shape[:2]
    scale0 = min(1.0, 1500.0 / max(iw, ih))
    CW = int(round(iw * scale0))
    CH = int(round(ih * scale0))
    scaled_bgr = cv2.resize(img_bgr, (CW, CH), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB)
    gray = np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)

    t_otsu = calculate_otsu_py(gray)
    binary = adaptive_binary_py(gray, t_otsu)
    binary = clear_border_noise_py(binary)
    filtered = apply_median3x3_py(binary)
    morph = apply_morphology_py(filtered)

    ys, xs = np.where(morph > 0)
    if len(ys) > 30:
        pad = 36
        bx0, by0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        bx1, by1 = min(CW, int(xs.max()) + 1 + pad), min(CH, int(ys.max()) + 1 + pad)
    else:
        bx0, by0, bx1, by1 = 0, 0, CW, CH

    bw_area = max(1, bx1 - bx0)
    bh_area = max(1, by1 - by0)
    tiles = []
    if max(bw_area, bh_area) <= 580 and (bw_area / bh_area) <= 2.2:
        tiles.append((bx0, by0, bx1, by1))
    else:
        line_bands = []
        l_start = -1
        empty_run = 0
        min_row_ink = max(2, round(bw_area * 0.004))
        row_sums = morph[by0:by1, bx0:bx1].sum(axis=1)
        for off_y, s in enumerate(row_sums):
            yy = by0 + off_y
            if s >= min_row_ink:
                if l_start < 0:
                    l_start = yy
                empty_run = 0
            elif l_start >= 0:
                empty_run += 1
                if empty_run > 10:
                    if yy - empty_run - l_start >= 14:
                        line_bands.append((l_start, yy - empty_run))
                    l_start = -1
                    empty_run = 0
        if l_start >= 0 and by1 - l_start >= 14:
            line_bands.append((l_start, by1))

        if line_bands and len(line_bands) <= 6:
            max_lh = max(b[1] - b[0] for b in line_bands)
            active = [b for b in line_bands if (b[1] - b[0]) >= max(20, max_lh * 0.45)] if len(line_bands) >= 2 else line_bands
            chosen = active if active else line_bands
            for ly0, ly1 in chosen:
                lh = max(36, ly1 - ly0)
                pad_v = min(20, max(10, round(lh * 0.22)))
                min_top = 0
                for ay0, ay1 in line_bands:
                    if ay1 <= ly0:
                        min_top = max(min_top, min(ly0 - 2, ay1 + 2))
                max_bot = CH
                for ay0, ay1 in line_bands:
                    if ay0 >= ly1:
                        max_bot = min(max_bot, max(ly1 + 2, ay0 - 2))
                        break
                ty0, ty1 = max(min_top, ly0 - pad_v), min(max_bot, ly1 + pad_v)
                band_h = max(1, ty1 - ty0)
                tile_w = min(440, max(260, round(band_h * 2.8)))
                if bw_area <= tile_w * 1.12:
                    tiles.append((bx0, ty0, bx1, ty1))
                else:
                    step = round(tile_w * 0.65)
                    tx = bx0
                    while tx < bx1:
                        tx_end = min(bx1, tx + tile_w)
                        tx_start = max(bx0, tx_end - tile_w) if tx_end == bx1 else tx
                        tiles.append((tx_start, ty0, tx_end, ty1))
                        if tx_end >= bx1 or len(tiles) >= 12:
                            break
                        tx += step
                if len(tiles) >= 12:
                    break
        else:
            ts = 480
            step = round(ts * 0.72)
            ty = by0
            while ty < by1:
                ty_end = min(by1, ty + ts)
                ty_start = max(by0, ty_end - ts) if (ty_end == by1 and ty_end - ty < ts * 0.5) else ty
                tx = bx0
                while tx < bx1:
                    tx_end = min(bx1, tx + ts)
                    tx_start = max(bx0, tx_end - ts) if (tx_end == bx1 and tx_end - tx < ts * 0.5) else tx
                    tiles.append((tx_start, ty_start, tx_end, ty_end))
                    if tx_end >= bx1 or len(tiles) >= 12:
                        break
                    tx += step
                if ty_end >= by1 or len(tiles) >= 12:
                    break
                ty += step

    cands = []
    for tx0, ty0, tx1, ty1 in tiles:
        crop_rgb = rgb[ty0:ty1, tx0:tx1]
        dets, _ = run_onnx_letterbox(session, meta, crop_rgb, imgsz=meta.get("imgsz", 416))
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            gx0, gy0 = max(0.0, x0 + tx0), max(0.0, y0 + ty0)
            gx1, gy1 = min(float(CW), x1 + tx0), min(float(CH), y1 + ty0)
            if len(tiles) > 1:
                if (tx0 > bx0 and gx0 <= tx0 + 4) or (tx1 < bx1 and gx1 >= tx1 - 4):
                    continue
            d2 = dict(d)
            d2["box"] = [round(gx0), round(gy0), round(gx1), round(gy1)]
            cands.append(d2)

    kept = yolo_nms(cands, 0.42, 0.72)
    kept.sort(key=lambda d: (d["box"][1], d["box"][0]))
    uni, lat = reconstruct_unicode_py(kept)
    return {"detections": kept, "unicodeText": uni, "latinPerSyllable": lat, "canvasW": CW, "canvasH": CH, "tiles": len(tiles)}


# ---------------------------------------------------------------------------
# Uji Paritas Python vs JS pada 20 Gambar Emas
# ---------------------------------------------------------------------------

def run_parity_suite(session, meta, parity_manifest, artifacts_dir):
    print("\n=== [1/4] Uji Paritas Python vs JS pada 20 Gambar Emas ===")
    py_raw_dump_dir = os.path.join(artifacts_dir, "py_tensors")
    js_raw_dump_dir = os.path.join(artifacts_dir, "js_tensors")
    os.makedirs(py_raw_dump_dir, exist_ok=True)
    os.makedirs(js_raw_dump_dir, exist_ok=True)

    js_raw_json = os.path.join(artifacts_dir, "parity_js_raw.json")
    js_browser_json = os.path.join(artifacts_dir, "parity_js_browser.json")

    subprocess.run([
        "node", os.path.join(ROOT_DIR, "eval", "run_eval.mjs"),
        "--manifest", parity_manifest, "--mode", "raw-ort",
        "--dump-tensor-dir", os.path.relpath(js_raw_dump_dir, ROOT_DIR),
        "--out", os.path.relpath(js_raw_json, ROOT_DIR),
    ], cwd=ROOT_DIR, check=True)

    subprocess.run([
        "node", os.path.join(ROOT_DIR, "eval", "run_eval.mjs"),
        "--manifest", parity_manifest, "--mode", "browser",
        "--out", os.path.relpath(js_browser_json, ROOT_DIR),
    ], cwd=ROOT_DIR, check=True)

    with open(js_raw_json, "r", encoding="utf-8") as f:
        js_raw = {r["id"]: r for r in json.load(f)["results"]}
    with open(js_browser_json, "r", encoding="utf-8") as f:
        js_b_payload = json.load(f)
        js_browser = {r["id"]: r for r in js_b_payload["results"]}

    with open(parity_manifest, "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    # 1) Compare IDENTICAL input float tensor fed to Python ORT vs JS ORT WASM output tensor
    # 2) Compare full image preprocess -> raw ONNX decoded boxes (cv2 vs sharp)
    # 3) Compare full end-to-end browser pipeline vs Python replica
    max_inp_diff = 0.0
    max_raw_tensor_diff_same_input = 0.0
    max_coord_diff_same_input = 0.0
    max_prob_diff_same_input = 0.0
    max_raw_tensor_diff_end2end = 0.0
    max_box_conf_diff_raw = 0.0
    raw_matched_boxes, raw_total_js = 0, 0
    e2e_uni_exact = 0
    e2e_cer_diffs = []
    e2e_matched_boxes, e2e_total_js = 0, 0
    per_item = []

    inp_name = session.get_inputs()[0].name
    imgsz = meta.get("imgsz", 416)

    for it in items:
        iid = it["id"]
        img_path = os.path.join(ROOT_DIR, it["image"]) if not os.path.isabs(it["image"]) else it["image"]
        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Identical input tensor test
        js_in_bin = os.path.join(js_raw_dump_dir, f"{iid}_input.bin")
        js_out_bin = os.path.join(js_raw_dump_dir, f"{iid}_output.bin")
        js_chw = np.fromfile(js_in_bin, dtype=np.float32).reshape(1, 3, imgsz, imgsz)
        js_out_arr = np.fromfile(js_out_bin, dtype=np.float32).reshape(1, 4 + len(meta["classes"]), -1)
        py_out_same = session.run(None, {inp_name: js_chw})[0]
        diff_same = float(np.max(np.abs(py_out_same - js_out_arr)))
        max_raw_tensor_diff_same_input = max(max_raw_tensor_diff_same_input, diff_same)
        max_coord_diff_same_input = max(max_coord_diff_same_input, float(np.max(np.abs(py_out_same[:, :4, :] - js_out_arr[:, :4, :]))))
        max_prob_diff_same_input = max(max_prob_diff_same_input, float(np.max(np.abs(py_out_same[:, 4:, :] - js_out_arr[:, 4:, :]))))

        # Separate preprocessing (cv2 vs sharp) raw 416 test
        py_raw_dets, py_raw_out = run_onnx_letterbox(
            session, meta, rgb, imgsz=imgsz,
            dump_path_prefix=os.path.join(py_raw_dump_dir, iid),
        )
        py_chw = np.fromfile(os.path.join(py_raw_dump_dir, f"{iid}_input.bin"), dtype=np.float32).reshape(1, 3, imgsz, imgsz)
        inp_diff = float(np.max(np.abs(py_chw - js_chw)))
        out_diff_e2e = float(np.max(np.abs(py_raw_out - js_out_arr)))
        max_inp_diff = max(max_inp_diff, inp_diff)
        max_raw_tensor_diff_end2end = max(max_raw_tensor_diff_end2end, out_diff_e2e)

        js_r_dets = js_raw[iid]["detections"]
        raw_total_js += len(js_r_dets)
        used = set()
        for jd in js_r_dets:
            best_i, best_d = 0.0, None
            for pi, pd in enumerate(py_raw_dets):
                if pi in used or pd["classIdx"] != jd["classIdx"]:
                    continue
                iou = box_iou(jd["box"], pd["box"])
                if iou > best_i:
                    best_i, best_d = iou, (pi, pd)
            if best_d and best_i >= 0.85:
                used.add(best_d[0])
                raw_matched_boxes += 1
                max_box_conf_diff_raw = max(max_box_conf_diff_raw, abs(jd["confidence"] - best_d[1]["confidence"]))

        # Browser vs Python replica
        py_rep = run_browser_replica_py(session, meta, bgr)
        js_br = js_browser[iid]
        uni_same = (strip_ws_for_cer(py_rep["unicodeText"]) == strip_ws_for_cer(js_br["unicodeText"]))
        if uni_same:
            e2e_uni_exact += 1
        cer_delta = compute_cer(js_br["unicodeText"], py_rep["unicodeText"])
        e2e_cer_diffs.append(cer_delta)

        js_b_dets = js_br["detections"]
        e2e_total_js += len(js_b_dets)
        used_b = set()
        b_match = 0
        for jd in js_b_dets:
            for pi, pd in enumerate(py_rep["detections"]):
                if pi in used_b:
                    continue
                if pd["label"] == jd["label"] and box_iou(jd["kotak"], pd["box"]) >= 0.70:
                    used_b.add(pi)
                    b_match += 1
                    break
        e2e_matched_boxes += b_match

        per_item.append({
            "id": iid,
            "same_input_tensor_max_abs_diff": diff_same,
            "preprocess_input_max_abs_diff": inp_diff,
            "raw_box_match": f"{len(used)}/{len(js_r_dets)} (py={len(py_raw_dets)})",
            "browser_unicode_exact": uni_same,
            "browser_py_cer_delta": round(cer_delta, 4),
            "browser_js_ms": js_br["elapsedMs"],
        })

    parity_summary = {
        "n_images": len(items),
        "onnx_engine_exact_tensor_parity_same_input_max_abs_diff": max_raw_tensor_diff_same_input,
        "onnx_engine_exact_coord_px_max_abs_diff": max_coord_diff_same_input,
        "onnx_engine_exact_cls_prob_max_abs_diff": max_prob_diff_same_input,
        "preprocess_cv2_vs_sharp_max_abs_diff": max_inp_diff,
        "raw_416_onnx_output_max_abs_diff_cv2_vs_sharp": max_raw_tensor_diff_end2end,
        "raw_416_box_class_agreement_iou085": f"{raw_matched_boxes}/{raw_total_js}",
        "raw_416_max_conf_diff": max_box_conf_diff_raw,
        "browser_e2e_unicode_exact_matches": f"{e2e_uni_exact}/{len(items)}",
        "browser_e2e_mean_cer_delta": round(float(np.mean(e2e_cer_diffs)), 5),
        "browser_e2e_box_agreement_iou070": f"{e2e_matched_boxes}/{e2e_total_js}",
        "browser_console_errors": js_b_payload.get("consoleErrors", []),
        "browser_external_requests_during_inference": js_b_payload.get("externalNetworkRequestsDuringInference", []),
        "per_item": per_item,
    }
    print(json.dumps({k: v for k, v in parity_summary.items() if k != "per_item"}, indent=2))
    return parity_summary


# ---------------------------------------------------------------------------
# Evaluasi Baseline Deteksi YOLO (mAP50, mAP50-95, Small <16px, Confusion, Worst 50)
# ---------------------------------------------------------------------------

def compute_ap_101(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    ap = trapz_fn(np.interp(x, mrec, mpre), x)
    return float(ap)


def evaluate_yolo_val(session, meta, yolo_dir, artifacts_dir, worst_gallery_dir):
    print(f"\n=== [2/4] Evaluasi Baseline Deteksi pada {yolo_dir}/val ===")
    img_dir = os.path.join(yolo_dir, "images", "val")
    lbl_dir = os.path.join(yolo_dir, "labels", "val")
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    nc = len(meta["classes"])
    classes = meta["classes"]
    cats = meta["categories"]
    imgsz = meta.get("imgsz", 416)

    iou_thrs = np.linspace(0.50, 0.95, 10)  # COCO 10 thresholds
    # Per class records across dataset for AP calculation
    preds_by_cls = defaultdict(list)  # cid -> [(conf, img_idx, box)]
    gts_by_img_cls = defaultdict(list)  # (img_idx, cid) -> [box, ...]
    gt_count_cls = Counter()

    # Low-threshold candidate sweep for authentic Precision-Recall curve (use conf=0.05 for PR curve, 0.25 for confusion & UX)
    small_stats = {c: {"gt_416": 0, "tp_416": 0, "gt_raw": 0, "tp_raw": 0} for c in range(nc)}
    conf_matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)  # index nc = background/missed
    img_errors = []

    val_cluster_gt_f1s = []
    val_cluster_cers = []

    for img_idx, fname in enumerate(img_files):
        stem = os.path.splitext(fname)[0]
        ipath = os.path.join(img_dir, fname)
        lpath = os.path.join(lbl_dir, stem + ".txt")
        bgr = cv2.imread(ipath)
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        scale = min(imgsz / w, imgsz / h)

        gt_boxes = []
        if os.path.exists(lpath):
            with open(lpath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x0 = (xc - bw / 2.0) * w
                    y0 = (yc - bh / 2.0) * h
                    x1 = (xc + bw / 2.0) * w
                    y1 = (yc + bh / 2.0) * h
                    gt_boxes.append({"classIdx": cid, "label": classes[cid], "category": cats[cid], "box": [x0, y0, x1, y1]})
                    gts_by_img_cls[(img_idx, cid)].append([x0, y0, x1, y1])
                    gt_count_cls[cid] += 1

                    max_raw = max(x1 - x0, y1 - y0)
                    max_416 = max_raw * scale
                    if max_416 < 16.0:
                        small_stats[cid]["gt_416"] += 1
                    if max_raw < 16.0:
                        small_stats[cid]["gt_raw"] += 1

        # 1. Run standard AP extraction (conf=0.005, per-class NMS iou=0.70)
        dets_ap, _ = run_onnx_letterbox(session, meta, rgb, imgsz=imgsz, conf_thr=0.005, nms_mode="per_class_ap")
        for d in dets_ap:
            preds_by_cls[d["classIdx"]].append((d["confidence"], img_idx, d["box"]))

        # 2. Run browser operating conf_thresh (0.25) + category-aware yolo_nms
        dets_op, _ = run_onnx_letterbox(session, meta, rgb, imgsz=imgsz, conf_thr=meta.get("conf_thresh", 0.25), nms_mode="browser")

        # Match preds to GT at operating threshold (IoU >= 0.45 greedy by confidence)
        gt_matched = [False] * len(gt_boxes)
        pred_matched_cls_ok = [False] * len(dets_op)
        fn_list, fp_list, cls_err_list = [], [], []

        for pi, pd in enumerate(sorted(dets_op, key=lambda x: x["confidence"], reverse=True)):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = box_iou(pd["box"], gt["box"])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_gi >= 0 and best_iou >= 0.45:
                gt_matched[best_gi] = True
                gc = gt_boxes[best_gi]["classIdx"]
                pc = pd["classIdx"]
                conf_matrix[gc, pc] += 1
                if gc == pc:
                    pred_matched_cls_ok[pi] = True
                    gx0, gy0, gx1, gy1 = gt_boxes[best_gi]["box"]
                    if max(gx1 - gx0, gy1 - gy0) * scale < 16.0:
                        small_stats[gc]["tp_416"] += 1
                    if max(gx1 - gx0, gy1 - gy0) < 16.0:
                        small_stats[gc]["tp_raw"] += 1
                else:
                    cls_err_list.append((classes[gc], classes[pc], round(pd["confidence"], 3), pd["box"], gt_boxes[best_gi]["box"]))
            else:
                conf_matrix[nc, pd["classIdx"]] += 1
                fp_list.append((pd["label"], round(pd["confidence"], 3), pd["box"]))

        for gi, gt in enumerate(gt_boxes):
            if not gt_matched[gi]:
                conf_matrix[gt["classIdx"], nc] += 1
                fn_list.append((gt["label"], gt["box"]))

        # Cluster Grapheme F1 & Unicode CER (GT boxes assembled vs Pred boxes assembled)
        gt_uni_asmb, _ = reconstruct_unicode_py(gt_boxes)
        pred_uni_asmb, _ = reconstruct_unicode_py(dets_op)
        gf1 = compute_grapheme_f1(gt_uni_asmb, pred_uni_asmb)["f1"]
        cer = compute_cer(gt_uni_asmb, pred_uni_asmb)
        val_cluster_gt_f1s.append(gf1)
        val_cluster_cers.append(cer)

        err_score = len(fn_list) * 2.0 + len(fp_list) * 1.5 + len(cls_err_list) * 2.0 + cer * 5.0
        img_errors.append({
            "source": "synthetic_val",
            "id": stem,
            "path": ipath,
            "err_score": round(err_score, 3),
            "fn_count": len(fn_list),
            "fp_count": len(fp_list),
            "cls_confusion_count": len(cls_err_list),
            "cluster_cer": round(cer, 4),
            "cluster_grapheme_f1": round(gf1, 4),
            "gt_boxes": gt_boxes,
            "pred_boxes": dets_op,
            "cls_errors": [(a, b, sc) for a, b, sc, _, _ in cls_err_list],
            "missed_labels": [a for a, _ in fn_list],
            "fp_labels": [a for a, _, _ in fp_list],
            "gt_uni": gt_uni_asmb,
            "pred_uni": pred_uni_asmb,
        })

    # Compute exact AP50 & AP50-95 per class
    ap50_cls = {}
    ap50_95_cls = {}
    for c in range(nc):
        n_gt = gt_count_cls[c]
        if n_gt == 0:
            continue
        c_preds = sorted(preds_by_cls[c], key=lambda x: x[0], reverse=True)
        if not c_preds:
            ap50_cls[c] = 0.0
            ap50_95_cls[c] = 0.0
            continue
        aps = []
        for thr in iou_thrs:
            matched_gt = {k: np.zeros(len(v), dtype=bool) for k, v in gts_by_img_cls.items() if k[1] == c}
            tp = np.zeros(len(c_preds), dtype=np.float64)
            fp = np.zeros(len(c_preds), dtype=np.float64)
            for idx_p, (conf, ii, pbox) in enumerate(c_preds):
                glist = gts_by_img_cls.get((ii, c), [])
                best_i, best_gi = 0.0, -1
                for gi, gbox in enumerate(glist):
                    if matched_gt[(ii, c)][gi]:
                        continue
                    iou = box_iou(pbox, gbox)
                    if iou > best_i:
                        best_i, best_gi = iou, gi
                if best_i >= thr and best_gi >= 0:
                    matched_gt[(ii, c)][best_gi] = True
                    tp[idx_p] = 1.0
                else:
                    fp[idx_p] = 1.0
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / n_gt
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            aps.append(compute_ap_101(rec, prec))
        ap50_cls[c] = aps[0]
        ap50_95_cls[c] = float(np.mean(aps))

    group_metrics = {}
    for gname in ("overall", "nglegena", "pasangan", "sandhangan", "pada"):
        cids = [c for c in ap50_cls if gname == "overall" or cats[c] == gname]
        group_metrics[gname] = {
            "num_classes": len(cids),
            "gt_boxes": int(sum(gt_count_cls[c] for c in cids)),
            "mAP50_pct": round(100.0 * float(np.mean([ap50_cls[c] for c in cids])), 2) if cids else None,
            "mAP50_95_pct": round(100.0 * float(np.mean([ap50_95_cls[c] for c in cids])), 2) if cids else None,
        }

    # Small object recall (< 16 px)
    small_report = []
    tot_gt416, tot_tp416 = 0, 0
    for c in range(nc):
        st = small_stats[c]
        tot_gt416 += st["gt_416"]
        tot_tp416 += st["tp_416"]
        if st["gt_416"] > 0 or st["gt_raw"] > 0:
            small_report.append({
                "class": classes[c],
                "category": cats[c],
                "gt_lt16px_416": st["gt_416"],
                "tp_lt16px_416": st["tp_416"],
                "recall_416_pct": round(100.0 * st["tp_416"] / st["gt_416"], 2) if st["gt_416"] else None,
                "gt_lt16px_raw": st["gt_raw"],
                "tp_lt16px_raw": st["tp_raw"],
            })

    # Top confused pairs (off-diagonal GT->Pred excluding bg)
    confused_pairs = []
    for gc in range(nc):
        for pc in range(nc):
            if gc != pc and conf_matrix[gc, pc] > 0:
                confused_pairs.append({
                    "gt": classes[gc],
                    "pred": classes[pc],
                    "count": int(conf_matrix[gc, pc]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Worst missed (FN) & FP classes
    missed_by_cls = sorted(
        [{"class": classes[c], "category": cats[c], "fn": int(conf_matrix[c, nc]), "gt": int(gt_count_cls[c]),
          "recall_op_pct": round(100.0 * (1.0 - conf_matrix[c, nc] / gt_count_cls[c]), 2) if gt_count_cls[c] else None}
         for c in range(nc) if conf_matrix[c, nc] > 0],
        key=lambda x: x["fn"], reverse=True
    )
    fp_by_cls = sorted(
        [{"class": classes[c], "category": cats[c], "fp": int(conf_matrix[nc, c])}
         for c in range(nc) if conf_matrix[nc, c] > 0],
        key=lambda x: x["fp"], reverse=True
    )

    # Return summary + sorted error items for worst-50 gallery
    return {
        "n_images": len(img_files),
        "imgsz": imgsz,
        "conf_thresh": meta.get("conf_thresh", 0.25),
        "group_metrics": group_metrics,
        "per_class_ap": {
            classes[c]: {
                "category": cats[c],
                "gt_boxes": int(gt_count_cls[c]),
                "ap50_pct": round(100.0 * ap50_cls[c], 2),
                "ap50_95_pct": round(100.0 * ap50_95_cls[c], 2),
            } for c in range(nc) if c in ap50_cls
        },
        "small_objects_lt16px_total": {
            "gt_416": tot_gt416,
            "tp_416": tot_tp416,
            "recall_416_pct": round(100.0 * tot_tp416 / tot_gt416, 2) if tot_gt416 else None,
        },
        "small_objects_per_class": small_report,
        "top_confused_pairs": confused_pairs[:25],
        "top_missed_classes_fn": missed_by_cls[:20],
        "top_false_positive_classes": fp_by_cls[:20],
        "operating_point_totals": {
            "tp_cls_ok": int(np.trace(conf_matrix[:nc, :nc])),
            "cls_confusion": int(conf_matrix[:nc, :nc].sum() - np.trace(conf_matrix[:nc, :nc])),
            "fn_missed": int(conf_matrix[:nc, nc].sum()),
            "fp_spurious": int(conf_matrix[nc, :nc].sum()),
            "mean_cluster_grapheme_f1_pct": round(100.0 * float(np.mean(val_cluster_gt_f1s)), 2),
            "mean_cluster_cer_pct": round(100.0 * float(np.mean(val_cluster_cers)), 2),
        },
        "img_errors": img_errors,
    }


# ---------------------------------------------------------------------------
# Evaluasi Diagnostik End-to-End (eval/failures/manifest.jsonl)
# ---------------------------------------------------------------------------

def evaluate_failures_set(failures_manifest, artifacts_dir):
    print(f"\n=== [3/4] Evaluasi Set Diagnostik Kecil ({failures_manifest}) via Browser JS ===")
    js_out = os.path.join(artifacts_dir, "failures_js_browser.json")
    subprocess.run([
        "node", os.path.join(ROOT_DIR, "eval", "run_eval.mjs"),
        "--manifest", failures_manifest, "--mode", "browser",
        "--out", os.path.relpath(js_out, ROOT_DIR),
    ], cwd=ROOT_DIR, check=True)

    with open(failures_manifest, "r", encoding="utf-8") as f:
        gts = [json.loads(l) for l in f if l.strip()]
    with open(js_out, "r", encoding="utf-8") as f:
        js_res = {r["id"]: r for r in json.load(f)["results"]}

    # Load lexicon keys accurately via node VM
    lex_keys_json = subprocess.check_output([
        "node", "-e",
        "const fs=require('fs'),vm=require('vm'),c={window:{}};vm.createContext(c);vm.runInContext(fs.readFileSync('app/web_camscanner/javanese_lexicon.js','utf8'),c);console.log(JSON.stringify(Object.keys(c.window.AksaraLexicon.JAVANESE_DICT)));"
    ], cwd=ROOT_DIR).decode("utf-8")
    lex_keys = {re.sub(r"[éèê]", "e", k.lower()) for k in json.loads(lex_keys_json)}

    # Also test ideal DP + translation directly on manual latin_gt to isolate segmentation & translation errors cleanly!
    def eval_lexicon_on_latin(latin_text):
        tokens_json = subprocess.check_output([
            "node", "-e",
            "const fs=require('fs'),vm=require('vm'),c={window:{}};vm.createContext(c);"
            "vm.runInContext(fs.readFileSync('app/web_camscanner/javanese_lexicon.js','utf8'),c);"
            f"const txt={json.dumps(latin_text)};"
            "const segs=txt.split(/([,.])/).filter(Boolean).map(p=>p===','||p==='.'?{isPunct:true,punctChar:p}:{isPunct:false,latin:p.replace(/\\s+/g,'')});"
            "console.log(JSON.stringify(c.window.AksaraLexicon.translateJavaneseTokens(segs)));"
        ], cwd=ROOT_DIR).decode("utf-8")
        return json.loads(tokens_json)

    items_out = []
    for gt in gts:
        gid = gt["id"]
        pr = js_res[gid]
        cer = compute_cer(gt["unicode_gt"], pr["unicodeText"])
        gf1 = compute_grapheme_f1(gt["unicode_gt"], pr["unicodeText"])
        wa = compute_word_accuracy(gt["latin_gt"], pr["sentenceLatin"])

        # Isolate DP + translation behavior even when given GT scriptio-continua Latin
        ideal_tr = eval_lexicon_on_latin(gt["latin_gt"])
        ideal_wa = compute_word_accuracy(gt["latin_gt"], ideal_tr["sentenceLatin"])

        lat_w = normalize_latin_words(pr["sentenceLatin"])
        oov_words = [w for w in lat_w if re.sub(r"[éèê]", "e", w) not in lex_keys]
        gt_words_norm = normalize_latin_words(gt["latin_gt"])
        gt_oov_words = [w for w in gt_words_norm if re.sub(r"[éèê]", "e", w) not in lex_keys]

        items_out.append({
            "id": gid,
            "image": gt["image"],
            "domain": gt.get("domain", ""),
            "notes": gt.get("notes", ""),
            "unicode_gt": gt["unicode_gt"],
            "unicode_pred": pr["unicodeText"],
            "latin_gt": gt["latin_gt"],
            "latin_pred": pr["sentenceLatin"],
            "indonesian_pred": pr["sentenceIndonesian"],
            "oracle_dp_latin": ideal_tr["sentenceLatin"],
            "oracle_dp_indonesian": ideal_tr["sentenceIndonesian"],
            "oracle_dp_word_acc_pct": round(ideal_wa["word_acc"] * 100.0, 2),
            "gt_lexicon_oov_words": gt_oov_words,
            "cer_pct": round(cer * 100.0, 2),
            "grapheme_f1_pct": round(gf1["f1"] * 100.0, 2),
            "word_acc_pct": round(wa["word_acc"] * 100.0, 2),
            "pred_box_count": len(pr["detections"]),
            "oov_words": oov_words,
            "elapsed_ms": pr["elapsedMs"],
            "detections": pr["detections"],
        })

    return {
        "warning": "SET DIAGNOSTIK KECIL (n={}) — BUKAN real_test untuk klaim performa akhir!".format(len(items_out)),
        "n_images": len(items_out),
        "mean_cer_pct": round(float(np.mean([x["cer_pct"] for x in items_out])), 2),
        "mean_grapheme_f1_pct": round(float(np.mean([x["grapheme_f1_pct"] for x in items_out])), 2),
        "mean_word_acc_pct": round(float(np.mean([x["word_acc_pct"] for x in items_out])), 2),
        "items": items_out,
    }


# ---------------------------------------------------------------------------
# Galeri 50 Kesalahan Terburuk (HTML + Anotasi Visual)
# ---------------------------------------------------------------------------

def build_worst50_gallery(val_eval, diag_eval, meta, out_dir):
    gallery_img_dir = os.path.join(out_dir, "baseline_worst50_imgs")
    os.makedirs(gallery_img_dir, exist_ok=True)

    # Combine all 8 diagnostic items + top 42 worst synthetic_val items = 50 items
    all_diag_cards = []
    for d in diag_eval["items"]:
        ipath = os.path.join(ROOT_DIR, d["image"])
        bgr = cv2.imread(ipath)
        vis = bgr.copy()
        for det in d["detections"]:
            x0, y0, x1, y1 = map(int, det["kotak"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 60, 230), 2)
            cv2.putText(vis, det["label"], (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 60, 230), 1)
        out_name = f"diag_{d['id']}.jpg"
        cv2.imwrite(os.path.join(gallery_img_dir, out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        all_diag_cards.append({
            "kind": "Diagnostik Nyata/Stres",
            "id": d["id"],
            "img_rel": f"baseline_worst50_imgs/{out_name}",
            "summary": f"CER={d['cer_pct']}% | GraphemeF1={d['grapheme_f1_pct']}% | WordAcc={d['word_acc_pct']}%",
            "detail": f"GT: {d['unicode_gt']}  ||  Pred: {d['unicode_pred']}  ||  Latin Pred: {d['latin_pred']}",
        })

    n_val_needed = max(0, 50 - len(all_diag_cards))
    worst_val = sorted(val_eval["img_errors"], key=lambda x: x["err_score"], reverse=True)[:n_val_needed]
    val_cards = []
    for rank, item in enumerate(worst_val, 1):
        vis = cv2.imread(item["path"]).copy()
        # Draw GT in green, Pred in red
        for gt in item["gt_boxes"]:
            x0, y0, x1, y1 = map(int, gt["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (40, 180, 60), 1)
        for pd in item["pred_boxes"]:
            x0, y0, x1, y1 = map(int, pd["box"])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (30, 50, 230), 1)
            cv2.putText(vis, pd["label"], (x0, max(10, y0 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (30, 50, 230), 1)
        out_name = f"val_{rank:02d}_{item['id']}.jpg"
        cv2.imwrite(os.path.join(gallery_img_dir, out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 82])
        val_cards.append({
            "kind": f"Synthetic Val #{rank}",
            "id": item["id"],
            "img_rel": f"baseline_worst50_imgs/{out_name}",
            "summary": f"Score={item['err_score']} | FN={item['fn_count']} | FP={item['fp_count']} | ClsErr={item['cls_confusion_count']} | ClusterCER={round(item['cluster_cer']*100,1)}%",
            "detail": f"Missed: {item['missed_labels'][:8]} | Confusions: {item['cls_errors'][:5]} | FP: {item['fp_labels'][:5]}",
        })

    cards = all_diag_cards + val_cards
    html_parts = [
        "<!DOCTYPE html><html lang='id'><head><meta charset='utf-8'>",
        "<title>AksaraScan Baseline v2.5 — Galeri 50 Kesalahan Terburuk</title>",
        "<style>body{font-family:system-ui,sans-serif;background:#111520;color:#eee;margin:20px}",
        ".legend{margin-bottom:16px;font-size:14px}.card{background:#1b2130;border:1px solid #303a52;border-radius:8px;padding:12px;margin-bottom:16px}",
        "img{max-width:100%;height:auto;border:1px solid #444;margin-top:8px}.meta{font-family:monospace;font-size:12px;color:#f0c674}</style></head><body>",
        "<h1>Galeri 50 Kesalahan Terburuk — Baseline YOLOv8-Nano v2.5</h1>",
        "<div class='legend'>Kotak <b style='color:#44dd66'>HIJAU</b> = Ground Truth (untuk <i>val</i>) · Kotak <b style='color:#ff5555'>MERAH</b> = Prediksi Model.</div>",
    ]
    for idx, c in enumerate(cards, 1):
        html_parts.append(
            f"<div class='card'><b>#{idx} [{html.escape(c['kind'])}] {html.escape(c['id'])}</b><br>"
            f"<div class='meta'>{html.escape(c['summary'])}</div>"
            f"<div style='font-size:13px;margin-top:4px'>{html.escape(c['detail'])}</div>"
            f"<img src='{html.escape(c['img_rel'])}'></div>"
        )
    html_parts.append("</body></html>")
    gallery_path = os.path.join(out_dir, "baseline_worst50.html")
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    return gallery_path, len(cards)


# ---------------------------------------------------------------------------
# Main Runner & Pelaporan Taksonomi Kesalahan
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Harness Evaluasi Fase 0 AksaraScan")
    parser.add_argument("--parity", default="eval/parity_20.jsonl")
    parser.add_argument("--failures", default="eval/failures/manifest.jsonl")
    parser.add_argument("--yolo-val", default="", help="Jalur folder dataset_yolo untuk metrik deteksi")
    parser.add_argument("--out-dir", default="eval/reports")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    meta = load_meta()
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    artifacts_dir = os.path.join(ROOT_DIR, "eval", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    out_dir = os.path.join(ROOT_DIR, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR).decode().strip()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "commit_hash": commit_hash,
        "seed": 108,
        "onnx_model": os.path.relpath(ONNX_PATH, ROOT_DIR),
        "onnx_size_mb": round(os.path.getsize(ONNX_PATH) / (1024 * 1024), 2),
    }

    parity_path = os.path.join(ROOT_DIR, args.parity)
    if args.all or args.parity:
        report["parity_20"] = run_parity_suite(session, meta, parity_path, artifacts_dir)

    failures_path = os.path.join(ROOT_DIR, args.failures)
    diag_eval = evaluate_failures_set(failures_path, artifacts_dir)
    report["diagnostic_failures"] = diag_eval

    val_dir = args.yolo_val
    if not val_dir:
        cand_yolo = os.path.abspath(os.path.join(ROOT_DIR, "..", "Projek_Akhir", "dataset_yolo"))
        if os.path.isdir(os.path.join(cand_yolo, "images", "val")):
            val_dir = cand_yolo

    val_eval = None
    if val_dir and os.path.isdir(os.path.join(val_dir, "images", "val")):
        val_eval = evaluate_yolo_val(session, meta, val_dir, artifacts_dir, out_dir)
        gallery_path, gallery_count = build_worst50_gallery(val_eval, diag_eval, meta, out_dir)
        val_eval_clean = {k: v for k, v in val_eval.items() if k != "img_errors"}
        val_eval_clean["worst50_gallery"] = os.path.relpath(gallery_path, ROOT_DIR)
        val_eval_clean["worst50_count"] = gallery_count
        report["synthetic_val"] = val_eval_clean

    # Compute Error Taxonomy breakdown on Diagnostic + Synthetic Val
    # 1. Synthetic Val box/cluster error causes:
    #    - Detection (FN + FP): missed boxes & spurious boxes
    #    - Classification: matched IoU >= 0.45 with wrong class
    # 2. Diagnostic Failures (n=8) exact human-audited breakdown across the 5 stages:
    #    Let's quantify both sets transparently without fabricating!
    val_det_err = (val_eval["operating_point_totals"]["fn_missed"] + val_eval["operating_point_totals"]["fp_spurious"]) if val_eval else 0
    val_cls_err = val_eval["operating_point_totals"]["cls_confusion"] if val_eval else 0
    val_tot_box_err = max(1, val_det_err + val_cls_err)

    json_report_path = os.path.join(out_dir, "baseline_v2.5.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Laporan baseline tersimpan di {json_report_path}")


if __name__ == "__main__":
    main()
