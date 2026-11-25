"""
Konverter dataset `setilanaji/aksara-jawa-ocr` (Apache-2.0) dari format PPOCRLabel
(`points` poligon + `transcription` Unicode Aksara Jawa) menjadi pasangan gambar baris
bersih (`dataset/external/line_ocr/setilanaji/{train,val}/`) untuk:
  - Track B Line-OCR (CRNN-CTC / TrOCR)
  - Evaluasi pembacaan baris kalimat bersandhangan & berpasangan.

Juga mengaudit cakupan karakter Unicode terhadap blok Aksara Jawa (U+A980..U+A9DF)
dan mencatat karakter di luar 55 kelas atomik (seperti dha/tha mahaprana atau swara).
"""
import json
import os
import re
import cv2
import numpy as np

from schema_55 import compute_dhash64

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(ROOT_DIR, "dataset", "external", "setilanaji_aksara_jawa_ocr")
OUT_DIR = os.path.join(ROOT_DIR, "dataset", "external", "line_ocr", "setilanaji")

# Set kode Unicode yang ter-cover oleh 55 kelas atomik AksaraScan
COVERED_UNICODE = set(
    "\uA9B2\uA9A4\uA995\uA9AB\uA98F\uA9A2\uA9A0\uA9B1\uA9AE\uA9AD"
    "\uA9A5\uA99D\uA997\uA9AA\uA99A\uA9A9\uA992\uA9A7\uA99B\uA994"
    "\uA9B6\uA9B8\uA9BC\uA9BA\uA9B4\uA982\uA981\uA983\uA9C0\uA9BF\uA9BD\uA9BE"
    "\uA9CB\uA9C8\uA9C9"
)


def parse_ppocr_label_file(lbl_path, img_base_dir, split_name):
    records = []
    oov_chars = {}
    if not os.path.exists(lbl_path):
        return records, oov_chars

    os.makedirs(os.path.join(OUT_DIR, "images", split_name), exist_ok=True)
    with open(lbl_path, "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line or "\t" not in line:
                continue
            fname, json_part = line.split("\t", 1)
            img_path = os.path.join(img_base_dir, fname)
            if not os.path.exists(img_path):
                continue
            try:
                anns = json.loads(json_part)
            except Exception:
                continue
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]

            for ann_idx, ann in enumerate(anns):
                if ann.get("illegibility"):
                    continue
                txt = (ann.get("transcription") or "").strip()
                pts = ann.get("points") or []
                if not txt or len(pts) < 4:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x0 = max(0, int(min(xs)) - 2)
                y0 = max(0, int(min(ys)) - 2)
                x1 = min(w, int(max(xs)) + 2)
                y1 = min(h, int(max(ys)) + 2)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                crop = bgr[y0:y1, x0:x1]
                out_name = f"seti_{split_name}_{os.path.splitext(os.path.basename(fname))[0]}_L{ann_idx}.jpg"
                out_path = os.path.join(OUT_DIR, "images", split_name, out_name)
                cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 94])

                for ch in txt:
                    if not ch.isspace() and ch not in COVERED_UNICODE:
                        oov_chars[ch] = oov_chars.get(ch, 0) + 1

                records.append({
                    "source": "hf_setilanaji_aksara_jawa_ocr",
                    "split": split_name,
                    "source_page": fname,
                    "image": os.path.relpath(out_path, ROOT_DIR),
                    "unicode": txt,
                    "dhash": compute_dhash64(crop),
                    "width": x1 - x0,
                    "height": y1 - y0,
                })
    return records, oov_chars


def main():
    if not os.path.isdir(SRC_DIR):
        print(f"[SKIP] {SRC_DIR} belum tersedia.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    tr_recs, oov_tr = parse_ppocr_label_file(
        os.path.join(SRC_DIR, "train", "Label.txt"),
        os.path.join(SRC_DIR, "train"),
        "train",
    )
    va_recs, oov_va = parse_ppocr_label_file(
        os.path.join(SRC_DIR, "Label.txt"),
        SRC_DIR,
        "val",
    )

    all_recs = tr_recs + va_recs
    with open(os.path.join(OUT_DIR, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    oov_merged = {}
    for d in (oov_tr, oov_va):
        for k, v in d.items():
            oov_merged[f"{k} (U+{ord(k):04X})"] = oov_merged.get(f"{k} (U+{ord(k):04X})", 0) + v

    report = {
        "source": "hf_setilanaji_aksara_jawa_ocr",
        "license": "Apache-2.0",
        "train_lines": len(tr_recs),
        "val_lines": len(va_recs),
        "total_lines": len(all_recs),
        "oov_unicode_chars_outside_55": oov_merged,
    }
    with open(os.path.join(OUT_DIR, "conversion_log.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[convert_setilanaji] train={len(tr_recs)} baris, val={len(va_recs)} baris, OOV chars={oov_merged}")


if __name__ == "__main__":
    main()
