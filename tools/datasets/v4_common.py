"""
Utilitas bersama pipeline ingest v4:
  - Penulisan label YOLO & manifes JSONL v4 terpadu
  - Generator contact sheet audit visual
  - Laporan QA per sumber
"""
import json
import os
import random
import cv2
import numpy as np

from schema_v4 import CLASSES_55, qa_validate_sample

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_DIR = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4")
REPORT_DIR = os.path.join(ROOT_DIR, "dataset", "external", "reports")
V4_SOURCES_DIR = os.path.join(ROOT_DIR, "dataset", "external", "v4_sources")


def write_yolo_label_file(lbl_path, boxes):
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    with open(lbl_path, "w", encoding="utf-8") as lf:
        for b in boxes:
            xc, yc, bw, bh = b["bbox"]
            lf.write(f"{int(b['cid'])} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def save_v4_manifest(name, records):
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    out_path = os.path.join(MANIFEST_DIR, f"{name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


def build_contact_sheet(items, title, out_path, max_n=60, cols=10, cell_w=132, cell_h=104, seed=42):
    if not items:
        return None
    rng = random.Random(seed)
    sample = items if len(items) <= max_n else rng.sample(items, max_n)
    rows = (len(sample) + cols - 1) // cols
    canvas = np.full((rows * cell_h + 38, cols * cell_w, 3), 246, dtype=np.uint8)
    cv2.putText(canvas, f"{title} (n={len(sample)} / {len(items)})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (20, 26, 38), 2, cv2.LINE_AA)

    for idx, rec in enumerate(sample):
        r, c = idx // cols, idx % cols
        x0, y0 = c * cell_w, 38 + r * cell_h
        ipath = rec["image"] if os.path.isabs(rec["image"]) else os.path.join(ROOT_DIR, rec["image"])
        bgr = cv2.imread(ipath)
        if bgr is None:
            continue
        ih, iw = bgr.shape[:2]
        sc = min((cell_w - 8) / max(1, iw), (cell_h - 24) / max(1, ih))
        nw, nh = max(1, int(iw * sc)), max(1, int(ih * sc))
        thumb = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        ox = x0 + (cell_w - nw) // 2
        oy = y0 + (cell_h - 20 - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = thumb

        for b in rec.get("boxes", []):
            xc, yc, bw, bh = b["bbox"]
            bx0 = int(ox + (xc - bw / 2.0) * nw)
            by0 = int(oy + (yc - bh / 2.0) * nh)
            bx1 = int(ox + (xc + bw / 2.0) * nw)
            by1 = int(oy + (yc + bh / 2.0) * nh)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (22, 130, 230), 1)

        preview = rec.get("label_preview")
        if not preview:
            if rec.get("boxes"):
                preview = f"{rec['boxes'][0]['class']} [{rec.get('split','')[:2]}]"
            else:
                preview = f"NEG [{rec.get('split','')[:2]}]"
        cv2.putText(canvas, str(preview)[:17], (x0 + 4, y0 + cell_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (35, 35, 42), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path
