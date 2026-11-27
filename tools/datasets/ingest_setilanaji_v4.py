"""
Ingest v4 untuk `setilanaji/aksara-jawa-ocr` (Apache-2.0):
  1. Memisahkan `source_page` secara tegas (`seti_train/<halaman>` vs `seti_val/<halaman>`)
     agar tabrakan nama berkas `aksara_0001.jpg` di train vs val tidak menipu pemisah split.
  2. Menghitung pHash64 + dHash64 dan menyingkirkan duplikat dekat antar-split.
  3. Mempertahankan kotak guru terverifikasi (`setilanaji_teacher` di split `train` SAJA) dan
     mengekspos seluruh baris `setilanaji` untuk Track B & evaluasi cetak nyata (`val` diisolasi).
"""
import argparse
import json
import os
from collections import Counter, defaultdict
import cv2

from schema_v4 import CLASSES_55, cluster_near_duplicates, compute_dhash64, compute_phash64, qa_validate_sample
from v4_common import ROOT_DIR, REPORT_DIR, build_contact_sheet, save_v4_manifest

IDX_LINE = os.path.join(ROOT_DIR, "dataset", "external", "line_ocr", "setilanaji", "index.jsonl")
TEACHER_DIR = os.path.join(ROOT_DIR, "dataset", "external", "converted_yolo", "setilanaji_teacher")


def main():
    if not os.path.exists(IDX_LINE):
        print("[SKIP] setilanaji line_ocr belum tersedia.")
        return

    raw = [json.loads(l) for l in open(IDX_LINE, "r", encoding="utf-8") if l.strip()]
    enriched = []
    for r in raw:
        ip = os.path.join(ROOT_DIR, r["image"])
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        sp = "val" if r["split"] == "val" else "train"
        page_key = f"seti_{sp}/{os.path.basename(r['source_page'])}"
        enriched.append({
            "image_id": os.path.splitext(os.path.basename(ip))[0],
            "source": "hf_setilanaji_aksara_jawa_ocr",
            "license": "Apache-2.0",
            "writer_id": page_key,
            "page_key": page_key,
            "split": sp,
            "unicode": r["unicode"],
            "phash": compute_phash64(gray),
            "dhash": r.get("dhash") or compute_dhash64(gray),
            "image": r["image"].replace("\\", "/"),
            "label": None,
            "boxes": [],
            "label_preview": f"line({len(r['unicode'])}ch) [{sp[:2]}]",
            "qa_status": "pass",
        })

    # Pastikan cluster dekat lintas train/val dipisahkan secara aman (jika ada duplikat dekat di val yg muncul di train, buang dari val)
    cluster_near_duplicates(enriched, dhash_thresh=4, phash_thresh=6)
    tr_clusters = {r["cluster_id_num"] for r in enriched if r["split"] == "train"}

    clean_lines = []
    removed_cross = 0
    val_candidates = []
    for r in enriched:
        cid_num = r["cluster_id_num"]
        if r["split"] == "val" and cid_num in tr_clusters:
            removed_cross += 1
            continue
        r["cluster_id"] = f"seti_cl_{cid_num:05d}"
        del r["cluster_id_num"]
        del r["page_key"]
        if r["split"] == "val":
            val_candidates.append(r)
        else:
            clean_lines.append(r)

    # Pisahkan 16 baris dari non-train menjadi `test_heldout` terkunci (eksklusif untuk eval/heldout_real)
    # dan sisanya ke `val` agar nol tumpang tindih val vs heldout_real!
    val_candidates.sort(key=lambda x: x["image_id"])
    step_ho = max(1, len(val_candidates) // 16)
    heldout_idx = set(range(0, len(val_candidates), step_ho)[:16])
    for vi, r in enumerate(val_candidates):
        if vi in heldout_idx:
            r["split"] = "test_heldout"
            r["label_preview"] = r["label_preview"].replace("[va]", "[te]")
        clean_lines.append(r)

    # Muat juga setilanaji_teacher kotak (khusus train)
    teacher_recs = []
    t_img_dir = os.path.join(TEACHER_DIR, "images", "train")
    t_lbl_dir = os.path.join(TEACHER_DIR, "labels", "train")
    if os.path.isdir(t_img_dir):
        for fn in sorted(os.listdir(t_img_dir)):
            if not fn.lower().endswith((".jpg", ".png")):
                continue
            stem = os.path.splitext(fn)[0]
            ip = os.path.join(t_img_dir, fn)
            lp = os.path.join(t_lbl_dir, stem + ".txt")
            bgr = cv2.imread(ip)
            if bgr is None or not os.path.exists(lp):
                continue
            boxes = []
            for ln in open(lp, "r", encoding="utf-8"):
                ps = ln.strip().split()
                if len(ps) < 5:
                    continue
                cid = int(float(ps[0]))
                xc, yc, bw, bh = map(float, ps[1:5])
                if 0 <= cid < len(CLASSES_55):
                    boxes.append({"class": CLASSES_55[cid], "cid": cid, "bbox": [xc, yc, bw, bh]})
            ok, _ = qa_validate_sample(bgr.shape, boxes)
            if not ok or not boxes:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            teacher_recs.append({
                "image_id": stem,
                "source": "hf_setilanaji_teacher_boxed",
                "license": "Apache-2.0",
                "writer_id": f"seti_teacher_{stem}",
                "cluster_id": f"seti_teacher_{stem}",
                "split": "train",
                "phash": compute_phash64(gray),
                "dhash": compute_dhash64(gray),
                "image": os.path.relpath(ip, ROOT_DIR).replace("\\", "/"),
                "label": os.path.relpath(lp, ROOT_DIR).replace("\\", "/"),
                "boxes": boxes,
                "qa_status": "pass",
            })

    m_lines = save_v4_manifest("setilanaji_lines_v4", clean_lines)
    m_teach = save_v4_manifest("setilanaji_teacher_v4", teacher_recs)
    build_contact_sheet(
        clean_lines, "setilanaji_lines_v4 (Real Book OCR)", os.path.join(REPORT_DIR, "v4_contact_setilanaji.jpg")
    )
    print(json.dumps({
        "lines_manifest": os.path.relpath(m_lines, ROOT_DIR),
        "lines_total": len(clean_lines),
        "lines_splits": dict(Counter(r["split"] for r in clean_lines)),
        "cross_split_dupes_removed": removed_cross,
        "teacher_boxed_manifest": os.path.relpath(m_teach, ROOT_DIR),
        "teacher_boxed_total": len(teacher_recs),
    }, indent=2))


if __name__ == "__main__":
    main()
