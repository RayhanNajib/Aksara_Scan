"""
Ingest v4 modular untuk dataset Kaggle Aksara Jawa:
  - Mendukung format folder per-kelas (`kg_phiard`) ataupun format YOLO (`kg_hermansugiharto`).
  - Memvalidasi lisensi eksplisit sebelum memproses & mengekspor manifes v4.
"""
import argparse
import glob
import json
import os
import random
import shutil
from collections import Counter, defaultdict

import cv2

from schema_v4 import (
    ALIASES_55,
    CLASS_TO_ID,
    cluster_near_duplicates,
    compute_dhash64,
    compute_phash64,
    compute_tight_foreground_yolo_box,
    qa_validate_sample,
)
from v4_common import ROOT_DIR, REPORT_DIR, V4_SOURCES_DIR, build_contact_sheet, save_v4_manifest, write_yolo_label_file
from ingest_roboflow import ingest_roboflow_dir


def ingest_kaggle_folder(src_dir, name, license_str, seed=42):
    if not license_str or license_str.strip().lower() in ("unknown", "none", ""):
        raise ValueError("Aturan Besi #1: --license wajib diisi lisensi terbuka terverifikasi.")
    if os.path.exists(os.path.join(src_dir, "data.yaml")):
        return ingest_roboflow_dir(src_dir, name, license_str)
    if not os.path.isdir(src_dir):
        print(f"[SKIP] {src_dir} belum ditemukan (unduh manual via Kaggle CLI sesuai MANUAL.md).")
        return []

    raw = []
    qa_rejected = Counter()
    for root, _, files in os.walk(src_dir):
        cls_raw = os.path.basename(root).strip().lower()
        mapped = ALIASES_55.get(cls_raw)
        if not mapped or mapped not in CLASS_TO_ID:
            continue
        cid = CLASS_TO_ID[mapped]
        for fn in sorted(files):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            ip = os.path.join(root, fn)
            bgr = cv2.imread(ip)
            if bgr is None:
                qa_rejected["unreadable"] += 1
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            xc, yc, bw, bh = compute_tight_foreground_yolo_box(gray, pad=5)
            boxes = [{"class": mapped, "cid": cid, "bbox": [xc, yc, bw, bh]}]
            ok, reason = qa_validate_sample(bgr.shape, boxes)
            if not ok:
                qa_rejected[reason] += 1
                continue
            raw.append({
                "src": ip,
                "stem": f"{name}_{mapped}_{os.path.splitext(fn)[0]}",
                "ext": os.path.splitext(fn)[1],
                "dhash": compute_dhash64(gray),
                "phash": compute_phash64(gray),
                "boxes": boxes,
            })

    cluster_near_duplicates(raw, dhash_thresh=4, phash_thresh=6)
    clusters = defaultdict(list)
    for idx, r in enumerate(raw):
        clusters[r["cluster_id_num"]].append(idx)

    rng = random.Random(seed)
    cids = sorted(clusters.keys())
    rng.shuffle(cids)
    val_c = set(cids[:max(1, round(len(cids) * 0.15))])

    out_dir = os.path.join(V4_SOURCES_DIR, f"{name}_v4")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)

    manifest = []
    for c_key in cids:
        sp = "val" if c_key in val_c else "train"
        for i in clusters[c_key]:
            r = raw[i]
            dst_img = os.path.join(out_dir, "images", sp, r["stem"] + r["ext"])
            dst_lbl = os.path.join(out_dir, "labels", sp, r["stem"] + ".txt")
            shutil.copy2(r["src"], dst_img)
            write_yolo_label_file(dst_lbl, r["boxes"])
            manifest.append({
                "image_id": r["stem"],
                "source": name,
                "license": license_str,
                "writer_id": f"{name}_cl_{c_key:05d}",
                "cluster_id": f"{name}_cl_{c_key:05d}",
                "split": sp,
                "phash": r["phash"],
                "dhash": r["dhash"],
                "image": os.path.relpath(dst_img, ROOT_DIR).replace("\\", "/"),
                "label": os.path.relpath(dst_lbl, ROOT_DIR).replace("\\", "/"),
                "boxes": r["boxes"],
                "qa_status": "pass",
            })

    save_v4_manifest(f"{name}_v4", manifest)
    build_contact_sheet(manifest, f"{name}_v4", os.path.join(REPORT_DIR, f"v4_contact_{name}.jpg"))
    print(json.dumps({"name": name, "kept": len(manifest), "qa_rejected": dict(qa_rejected)}, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--license", required=True)
    args = ap.parse_args()
    ingest_kaggle_folder(args.src, args.name, args.license)
