"""
Ingest v4 untuk `hanacaraka_vzrenggamani` (CC0-1.0, dataset/hanacaraka/):
Memperbaiki kebocoran split penulis v1/v3 (`79` penulis tersebar ke train/val/test)
dengan pembagian ketat *writer-disjoint* + *cluster-disjoint*:
  - 55 penulis -> train
  - 12 penulis -> val
  - 12 penulis -> test_heldout (terkunci untuk eval/heldout_real)
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
    CLASS_TO_ID,
    EXPLICIT_ALIASES,
    cluster_near_duplicates,
    compute_dhash64,
    compute_phash64,
    compute_tight_foreground_yolo_box,
    qa_validate_sample,
)
from v4_common import ROOT_DIR, REPORT_DIR, V4_SOURCES_DIR, build_contact_sheet, save_v4_manifest, write_yolo_label_file

SRC_DIR = os.path.join(ROOT_DIR, "dataset", "hanacaraka")
OUT_DIR = os.path.join(V4_SOURCES_DIR, "hanacaraka_v4")


def main(seed=42):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    for sp in ("train", "val", "test_heldout"):
        os.makedirs(os.path.join(OUT_DIR, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", sp), exist_ok=True)

    raw_recs = []
    qa_rejected = Counter()
    for cls_dir in sorted(os.listdir(SRC_DIR)):
        cpath = os.path.join(SRC_DIR, cls_dir)
        if not os.path.isdir(cpath):
            continue
        mapped = EXPLICIT_ALIASES.get(cls_dir.strip().lower())
        if mapped is None or mapped not in CLASS_TO_ID:
            qa_rejected[f"unmapped_dir:{cls_dir}"] += 1
            continue
        cid = CLASS_TO_ID[mapped]
        for img_path in sorted(glob.glob(os.path.join(cpath, "*.*"))):
            if not img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            bgr = cv2.imread(img_path)
            if bgr is None:
                qa_rejected["unreadable"] += 1
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            xc, yc, bw_n, bh_n = compute_tight_foreground_yolo_box(gray, pad=6)
            boxes = [{"class": mapped, "cid": cid, "bbox": [xc, yc, bw_n, bh_n]}]
            ok, reason = qa_validate_sample(bgr.shape, boxes)
            if not ok:
                qa_rejected[reason] += 1
                continue

            bn = os.path.basename(img_path)
            writer_raw = bn.split("_")[0] if "_" in bn else bn
            raw_recs.append({
                "src_path": img_path,
                "bn": bn,
                "cls_name": mapped,
                "cid": cid,
                "writer_id": f"hnc_w_{writer_raw}",
                "dhash": compute_dhash64(gray),
                "phash": compute_phash64(gray),
                "boxes": boxes,
                "shape": bgr.shape[:2],
            })

    # Klaster near-duplicates sebelum split (dhash<=2 & phash<=3 agar tdk menyatukan penulis berbeda)
    cluster_near_duplicates(raw_recs, dhash_thresh=2, phash_thresh=3)

    # Gabungkan writer_id dan near-duplicate cluster via Union-Find agar nol kebocoran penulis maupun gambar dekat
    n = len(raw_recs)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_w, by_c = defaultdict(list), defaultdict(list)
    for i, r in enumerate(raw_recs):
        by_w[r["writer_id"]].append(i)
        by_c[r["cluster_id_num"]].append(i)
    for grp in list(by_w.values()) + list(by_c.values()):
        for idx in grp[1:]:
            union(grp[0], idx)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Urutkan grup secara deterministik dengan jumlah sampel seimbang lalu acak ber-seed
    group_ids = sorted(groups.keys(), key=lambda g: (len(groups[g]), raw_recs[groups[g][0]]["writer_id"]))
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    # Alokasi penulis/grup ketat: ~70% train, ~15% val, ~15% test_heldout (min 12 grup -> ~240 citra test_heldout)
    n_groups = len(group_ids)
    n_test = max(12, round(n_groups * 0.15))
    n_val = max(12, round(n_groups * 0.15))
    test_g = set(group_ids[:n_test])
    val_g = set(group_ids[n_test:n_test + n_val])

    manifest = []
    split_writers = defaultdict(set)
    split_counts = Counter()

    for g_idx, g_key in enumerate(group_ids):
        sp = "test_heldout" if g_key in test_g else ("val" if g_key in val_g else "train")
        super_cluster = f"hnc_grp_{g_idx:04d}"
        for i in groups[g_key]:
            r = raw_recs[i]
            stem = f"hnc_{r['cls_name']}_{os.path.splitext(r['bn'])[0]}"
            dst_img = os.path.join(OUT_DIR, "images", sp, stem + ".png")
            dst_lbl = os.path.join(OUT_DIR, "labels", sp, stem + ".txt")
            shutil.copy2(r["src_path"], dst_img)
            write_yolo_label_file(dst_lbl, r["boxes"])

            rel_img = os.path.relpath(dst_img, ROOT_DIR).replace("\\", "/")
            rel_lbl = os.path.relpath(dst_lbl, ROOT_DIR).replace("\\", "/")
            manifest.append({
                "image_id": stem,
                "source": "hanacaraka_vzrenggamani",
                "license": "CC0-1.0",
                "writer_id": r["writer_id"],
                "cluster_id": super_cluster,
                "split": sp,
                "phash": r["phash"],
                "dhash": r["dhash"],
                "image": rel_img,
                "label": rel_lbl,
                "boxes": r["boxes"],
                "qa_status": "pass",
            })
            split_writers[sp].add(r["writer_id"])
            split_counts[sp] += 1

    mpath = save_v4_manifest("hanacaraka_v4", manifest)
    cs_path = build_contact_sheet(
        manifest, "hanacaraka_v4 (Writer-Disjoint)", os.path.join(REPORT_DIR, "v4_contact_hanacaraka.jpg")
    )
    print(json.dumps({
        "manifest": os.path.relpath(mpath, ROOT_DIR),
        "total": len(manifest),
        "splits": dict(split_counts),
        "unique_writers_per_split": {k: len(v) for k, v in split_writers.items()},
        "qa_rejected": dict(qa_rejected),
        "contact_sheet": os.path.relpath(cs_path, ROOT_DIR) if cs_path else None,
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(seed=args.seed)
