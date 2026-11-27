"""
Ingest v4 modular untuk ekspor Roboflow Universe format YOLOv8.
Wajib menyertakan bukti lisensi terverifikasi (mis. --license "CC BY 4.0").
Menjamin QA kotak, pHash64/dHash64, klastering near-duplicate sebelum split,
pemetaan alias `classes_v4.yaml`, serta pembuatan contact sheet audit visual.

Contoh:
  python tools/datasets/ingest_roboflow.py --src dataset/external/rf_fasel_sandhangan --name rf_fasel_sandhangan --license "CC BY 4.0"
"""
import argparse
import json
import os
import shutil
from collections import Counter

import cv2
import yaml

from schema_v4 import (
    ALIASES_55,
    CLASS_TO_ID,
    cluster_near_duplicates,
    compute_dhash64,
    compute_phash64,
    qa_validate_sample,
)
from v4_common import ROOT_DIR, REPORT_DIR, V4_SOURCES_DIR, build_contact_sheet, save_v4_manifest, write_yolo_label_file


def ingest_roboflow_dir(src_dir, name, license_str):
    if not license_str or license_str.strip().lower() in ("unknown", "none", ""):
        raise ValueError("Aturan Besi #1: --license wajib diisi lisensi terbuka terverifikasi (mis. 'CC BY 4.0').")
    yaml_path = os.path.join(src_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[SKIP] {yaml_path} belum ditemukan (unduh manual sesuai dataset/LICENSES.md & tools/datasets/MANUAL.md).")
        return []

    src_cfg = yaml.safe_load(open(yaml_path, "r", encoding="utf-8"))
    raw_names = src_cfg.get("names", {})
    src_names = {i: n for i, n in enumerate(raw_names)} if isinstance(raw_names, list) else {int(k): v for k, v in raw_names.items()}

    mapping = {}
    for scid, sname in src_names.items():
        mapped = ALIASES_55.get(str(sname).strip().lower())
        if mapped and mapped in CLASS_TO_ID:
            mapping[scid] = (mapped, CLASS_TO_ID[mapped])

    out_dir = os.path.join(V4_SOURCES_DIR, f"{name}_v4")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    raw_recs = []
    qa_rejected = Counter()
    for src_sp, tgt_sp in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "val")):
        img_dir = os.path.join(src_dir, src_sp, "images")
        lbl_dir = os.path.join(src_dir, src_sp, "labels")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(src_dir, "images", src_sp)
            lbl_dir = os.path.join(src_dir, "labels", src_sp)
        if not os.path.isdir(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = os.path.splitext(fname)[0]
            ip = os.path.join(img_dir, fname)
            lp = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lp):
                continue
            bgr = cv2.imread(ip)
            if bgr is None:
                qa_rejected["unreadable"] += 1
                continue
            boxes = []
            for ln in open(lp, "r", encoding="utf-8"):
                ps = ln.strip().split()
                if len(ps) < 5:
                    continue
                scid = int(float(ps[0]))
                if scid not in mapping:
                    continue
                tname, tcid = mapping[scid]
                xc, yc, bw, bh = map(float, ps[1:5])
                boxes.append({"class": tname, "cid": tcid, "bbox": [xc, yc, bw, bh]})
            ok, reason = qa_validate_sample(bgr.shape, boxes)
            if not ok or not boxes:
                qa_rejected[reason] += 1
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            raw_recs.append({
                "src_img": ip,
                "stem": f"{name}_{stem}",
                "ext": os.path.splitext(fname)[1],
                "init_split": tgt_sp,
                "phash": compute_phash64(gray),
                "dhash": compute_dhash64(gray),
                "boxes": boxes,
            })

    cluster_near_duplicates(raw_recs, dhash_thresh=4, phash_thresh=6)
    tr_clusters = {r["cluster_id_num"] for r in raw_recs if r["init_split"] == "train"}

    manifest = []
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)

    for r in raw_recs:
        sp = "train" if r["cluster_id_num"] in tr_clusters else r["init_split"]
        dst_img = os.path.join(out_dir, "images", sp, r["stem"] + r["ext"])
        dst_lbl = os.path.join(out_dir, "labels", sp, r["stem"] + ".txt")
        shutil.copy2(r["src_img"], dst_img)
        write_yolo_label_file(dst_lbl, r["boxes"])
        manifest.append({
            "image_id": r["stem"],
            "source": name,
            "license": license_str,
            "writer_id": f"{name}_cl_{r['cluster_id_num']:05d}",
            "cluster_id": f"{name}_cl_{r['cluster_id_num']:05d}",
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
    ingest_roboflow_dir(args.src, args.name, args.license)
