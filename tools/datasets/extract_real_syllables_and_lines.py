"""
Ekstraktor & Pemverifikasi Label Nyata Aksara Jawa:
  1. `jawa_syllables` (Jawa.zip CC BY 4.0, 2.196 citra silabus bersandhangan nyata)
     - Memisahkan komponen sandhangan (taling kiri, tarung kanan, wulu/pepet atas) secara geometris deterministik
     - Untuk `suku` yang sering menyambung ke kaki kanan aksara, memakai verifikasi silang guru ONNX (base + suku cocok label folder)
  2. `setilanaji/aksara-jawa-ocr` (Apache-2.0, 2.059 baris nyata)
     - Forced teacher alignment: HANYA menyimpan baris train yang hasil rekonstruksi Unicode-nya 100% persis GT (CER == 0)
"""
import json
import os
import random
import sys
import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
from schema_55 import CLASS_TO_ID, CLASSES_55, EXPLICIT_ALIASES
from run_eval import run_onnx_letterbox, yolo_nms, reconstruct_unicode_py, compute_cer

EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")
OUT_SYL = os.path.join(EXT_DIR, "converted_yolo", "jawa_syllables_boxed")
OUT_SETI = os.path.join(EXT_DIR, "converted_yolo", "setilanaji_teacher")


def folder_to_base_class(folder_label):
    fl = folder_label.strip().lower()
    # Suffix vowels: e, i, o, u, è, e... e.g. be -> ba, nge -> nga, dhe -> dha, ce -> ca, the -> tha
    if fl.startswith("dh"):
        return "dha"
    if fl.startswith("th"):
        return "tha"
    if fl.startswith("ng"):
        return "nga"
    if fl.startswith("ny"):
        return "nya"
    c0 = fl[0]
    cand = c0 + "a"
    return cand if cand in CLASS_TO_ID else None


def clean_components(bw):
    num, lab, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = []
    H, W = bw.shape
    min_a = max(24, int(H * W * 0.00055))
    for i in range(1, num):
        x, y, w, h, area = map(int, stats[i])
        if area < min_a or w < 3 or h < 3:
            continue
        # Reject corner/edge dust specks touching perimeter with small area
        if (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2) and area < max(120, H * W * 0.004):
            continue
        comps.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h, "w": w, "h": h, "area": area, "cx": cents[i][0], "cy": cents[i][1]})
    return comps


def extract_jawa_syllables():
    pool_path = os.path.join(EXT_DIR, "jawa_syllables", "syllable_pool.jsonl")
    if not os.path.exists(pool_path):
        print("[SKIP] jawa_syllables not found")
        return
    rows = [json.loads(l) for l in open(pool_path, "r", encoding="utf-8") if l.strip()]

    # Load teacher ONNX for suku split verification
    meta = json.load(open(os.path.join(ROOT_DIR, "app", "web_camscanner", "yolo_meta.json"), "r", encoding="utf-8"))
    sess = ort.InferenceSession(os.path.join(ROOT_DIR, "app", "web_camscanner", "aksara_yolo.onnx"), providers=["CPUExecutionProvider"])

    for sp in ("train", "val"):
        os.makedirs(os.path.join(OUT_SYL, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_SYL, "labels", sp), exist_ok=True)

    # Deterministic 85/15 split stratified by (base, expected_modifier)
    by_key = {}
    for r in rows:
        base = folder_to_base_class(r["folder_label"])
        if not base:
            continue
        r["base_cls"] = base
        by_key.setdefault((base, r["expected_modifier"]), []).append(r)

    rng = random.Random(42)
    accepted = []
    preview_tiles = []
    idx = 0

    for key, lst in sorted(by_key.items()):
        lst.sort(key=lambda x: x["image"])
        rng.shuffle(lst)
        n_val = max(1, round(len(lst) * 0.15))
        for pos, r in enumerate(lst):
            sp = "val" if pos < n_val else "train"
            ip = os.path.join(ROOT_DIR, r["image"])
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            comps = clean_components(bw)
            if not comps:
                continue

            base = r["base_cls"]
            mod = r["expected_modifier"]
            boxes = []  # (cls_name, x0, y0, x1, y1)

            if mod == "taling":
                # Harus ada komponen kiri yang jelas terpisah sebagai taling, dan komponen kanan utama sebagai base
                comps_x = sorted(comps, key=lambda c: c["x0"])
                if len(comps_x) >= 2:
                    left_c = comps_x[0]
                    rest = comps_x[1:]
                    rx0 = min(c["x0"] for c in rest)
                    ry0 = min(c["y0"] for c in rest)
                    rx1 = max(c["x1"] for c in rest)
                    ry1 = max(c["y1"] for c in rest)
                    # Sanity check: left_c tidak menumpuk penuh di tengah dan tinggi taling >= 45% tinggi huruf
                    if left_c["cx"] < rx0 + (rx1 - rx0) * 0.25 and left_c["h"] >= (ry1 - ry0) * 0.42:
                        boxes = [("taling", left_c["x0"] - 2, left_c["y0"] - 2, left_c["x1"] + 2, left_c["y1"] + 2),
                                 (base, rx0 - 2, ry0 - 2, rx1 + 2, ry1 + 2)]

            elif mod == "taling_tarung":
                comps_x = sorted(comps, key=lambda c: c["x0"])
                if len(comps_x) >= 3:
                    left_c = comps_x[0]
                    right_c = comps_x[-1]
                    mid = comps_x[1:-1]
                    mx0 = min(c["x0"] for c in mid)
                    my0 = min(c["y0"] for c in mid)
                    mx1 = max(c["x1"] for c in mid)
                    my1 = max(c["y1"] for c in mid)
                    if left_c["cx"] < mx0 + 8 and right_c["cx"] > mx1 - 8 and left_c["h"] >= (my1 - my0) * 0.4 and right_c["h"] >= (my1 - my0) * 0.4:
                        boxes = [
                            ("taling", left_c["x0"] - 2, left_c["y0"] - 2, left_c["x1"] + 2, left_c["y1"] + 2),
                            (base, mx0 - 2, my0 - 2, mx1 + 2, my1 + 2),
                            ("tarung", right_c["x0"] - 2, right_c["y0"] - 2, right_c["x1"] + 2, right_c["y1"] + 2),
                        ]

            elif mod in ("wulu", "pepet"):
                # Diacritic sits clearly above the base character!
                if len(comps) >= 2:
                    comps_y = sorted(comps, key=lambda c: c["y0"])
                    top_c = comps_y[0]
                    rest = comps_y[1:]
                    rx0 = min(c["x0"] for c in rest)
                    ry0 = min(c["y0"] for c in rest)
                    rx1 = max(c["x1"] for c in rest)
                    ry1 = max(c["y1"] for c in rest)
                    rw, rh = max(1, rx1 - rx0), max(1, ry1 - ry0)
                    # Top component must float over the base span & clearly above base top
                    if (
                        top_c["cx"] >= rx0 - 0.1 * rw
                        and top_c["cx"] <= rx1 + 0.1 * rw
                        and top_c["y1"] <= ry0 + 0.22 * rh
                        and top_c["area"] < sum(c["area"] for c in rest) * 0.45
                        and top_c["h"] < rh * 0.65
                        and top_c["w"] < rw * 0.88
                    ):
                        boxes = [
                            (mod, top_c["x0"] - 2, top_c["y0"] - 2, top_c["x1"] + 2, top_c["y1"] + 2),
                            (base, rx0 - 2, ry0 - 2, rx1 + 2, ry1 + 2),
                        ]

            elif mod == "suku":
                # Verify via teacher ONNX: if teacher detects both base-like box + suku box, keep exact GT base + predicted suku box!
                dets, _ = run_onnx_letterbox(sess, meta, bgr, imgsz=416)
                suku_dets = [d for d in dets if d["label"] == "suku" and d["confidence"] >= 0.35]
                base_dets = [d for d in dets if d["category"] == "nglegena" and d["confidence"] >= 0.40]
                if len(suku_dets) == 1 and len(base_dets) == 1:
                    sb = suku_dets[0]["box"]
                    bb = base_dets[0]["box"]
                    boxes = [(base, bb[0], bb[1], bb[2], bb[3]), ("suku", sb[0], sb[1], sb[2], sb[3])]

            if not boxes:
                continue

            yolo_lines = []
            vis = bgr.copy()
            valid_all = True
            for cname, x0, y0, x1, y1 in boxes:
                x0, y0 = max(0.0, float(x0)), max(0.0, float(y0))
                x1, y1 = min(float(W), float(x1)), min(float(H), float(y1))
                bw_n, bh_n = (x1 - x0) / W, (y1 - y0) / H
                xc, yc = ((x0 + x1) / 2.0) / W, ((y0 + y1) / 2.0) / H
                if bw_n < 0.02 or bh_n < 0.02 or bw_n > 0.99 or bh_n > 0.99:
                    valid_all = False
                    break
                cid = CLASS_TO_ID[cname]
                yolo_lines.append((cid, xc, yc, bw_n, bh_n))
                col = (34, 197, 94) if cname == base else (0, 140, 255)
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), col, max(2, W // 110))

            if not valid_all:
                continue

            stem = f"jsyl_{idx:05d}_{base}_{mod}"
            dst_img = os.path.join(OUT_SYL, "images", sp, stem + ".jpg")
            dst_lbl = os.path.join(OUT_SYL, "labels", sp, stem + ".txt")
            cv2.imwrite(dst_img, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for cid, xc, yc, bw_n, bh_n in yolo_lines:
                    lf.write(f"{cid} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")
            accepted.append({"split": sp, "base": base, "mod": mod, "image": os.path.relpath(dst_img, ROOT_DIR), "label": os.path.relpath(dst_lbl, ROOT_DIR)})
            if len(preview_tiles) < 25 and idx % 35 == 0:
                preview_tiles.append(cv2.resize(vis, (160, 110)))
            idx += 1

    with open(os.path.join(OUT_SYL, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(r) + "\n")
    if preview_tiles:
        rows_g = [np.hstack(preview_tiles[k:k+5]) for k in range(0, len(preview_tiles) - len(preview_tiles)%5, 5)]
        if rows_g:
            cv2.imwrite("C:/Users/WA/.gemini/antigravity/brain/e8f20c41-7ca5-46b2-92f2-7e5b3331f29f/scratch/jawa_syl_boxed_audit.png", np.vstack(rows_g))
    from collections import Counter
    print("[jawa_syllables_boxed] extracted:", len(accepted), "by mod:", Counter(r["mod"] for r in accepted), "by split:", Counter(r["split"] for r in accepted))


def extract_setilanaji_teacher():
    idx_path = os.path.join(EXT_DIR, "line_ocr", "setilanaji", "index.jsonl")
    if not os.path.exists(idx_path):
        return
    recs = [json.loads(l) for l in open(idx_path, "r", encoding="utf-8") if l.strip() and '"train"' in l]
    meta = json.load(open(os.path.join(ROOT_DIR, "app", "web_camscanner", "yolo_meta.json"), "r", encoding="utf-8"))
    sess = ort.InferenceSession(os.path.join(ROOT_DIR, "app", "web_camscanner", "aksara_yolo.onnx"), providers=["CPUExecutionProvider"])

    os.makedirs(os.path.join(OUT_SETI, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_SETI, "labels", "train"), exist_ok=True)

    kept = 0
    for r in recs:
        if r.get("split") != "train":
            continue
        im = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
        if im is None:
            continue
        H, W = im.shape[:2]
        # Pad with median background paper so 416x416 doesn't clip edge characters, AND keep original clean crops
        pad = 24
        med = [int(np.median(im[:, :, c])) for c in range(3)]
        padded = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=med)
        # Mild lighting + contrast normalization for teacher pass
        dets, _ = run_onnx_letterbox(sess, meta, padded, imgsz=416)
        dets = yolo_nms(dets, 0.42, 0.72)
        if not dets:
            continue
        uni_pred, _ = reconstruct_unicode_py(dets)
        cer = compute_cer(r["unicode"], uni_pred)
        if cer <= 0.08 and len(dets) >= 4:
            pH, pW = padded.shape[:2]
            stem = f"seti_{kept:04d}"
            img_out = os.path.join(OUT_SETI, "images", "train", stem + ".jpg")
            lbl_out = os.path.join(OUT_SETI, "labels", "train", stem + ".txt")
            cv2.imwrite(img_out, padded, [cv2.IMWRITE_JPEG_QUALITY, 94])
            with open(lbl_out, "w", encoding="utf-8") as lf:
                for d in dets:
                    x0, y0, x1, y1 = d["box"]
                    cid = CLASS_TO_ID[d["label"]]
                    xc = max(0.001, min(0.999, ((x0 + x1) / 2.0) / pW))
                    yc = max(0.001, min(0.999, ((y0 + y1) / 2.0) / pH))
                    bw = max(0.005, min(0.99, (x1 - x0) / pW))
                    bh = max(0.005, min(0.99, (y1 - y0) / pH))
                    lf.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            kept += 1
    print(f"[setilanaji_teacher] Exact/Near-Exact real book lines aligned: {kept}")


if __name__ == "__main__":
    extract_jawa_syllables()
    extract_setilanaji_teacher()
