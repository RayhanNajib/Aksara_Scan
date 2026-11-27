"""
Ingest v4 untuk Mendeley `Jawa.zip` (CC BY 4.0, DOI 10.17632/vfj32bpjsf.1):
  1. Memuat SELURUH `Vokal A` (9.994) + `variations` (1.066) -> skema 55 kelas.
  2. Menghitung dHash64 + pHash64 & mengelompokkan near-duplicate cluster SEBELUM split 85/15
     agar cluster identik/mirip tidak bocor antar split.
  3. Menggabungkan silabus bersandhangan nyata `jawa_syllables_boxed` (Vokal E/I/O/U/È) ke manifes
     terpadu dengan cluster_id pHash/dHash yang disinkronkan.
"""
import argparse
import json
import os
import random
import shutil
import zipfile
from collections import Counter, defaultdict

import cv2
import numpy as np

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

ZIP_PATH = os.path.join(ROOT_DIR, "dataset", "external", "jawa_mendeley", "Jawa.zip")
SYL_BOXED_IDX = os.path.join(ROOT_DIR, "dataset", "external", "converted_yolo", "jawa_syllables_boxed", "index.jsonl")
OUT_JW = os.path.join(V4_SOURCES_DIR, "jawa_zip_v4")
OUT_SYL = os.path.join(V4_SOURCES_DIR, "jawa_syllables_v4")


def ingest_jawa_zip(seed=42):
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(f"Jawa.zip tidak ditemukan di {ZIP_PATH}")
    if os.path.exists(OUT_JW):
        shutil.rmtree(OUT_JW)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(OUT_JW, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_JW, "labels", sp), exist_ok=True)

    raw_items = []
    qa_rejected = Counter()
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = sorted(n for n in z.namelist() if not n.endswith("/") and n.lower().endswith((".png", ".jpg", ".jpeg")))
        for zp in all_names:
            parts = zp.split("/")
            src_cls, sub = None, None
            if len(parts) >= 5 and parts[1] == "all_class" and parts[2] == "Vokal A":
                src_cls, sub = parts[3].strip().lower(), "vokal_a"
            elif len(parts) >= 4 and parts[1] == "variations":
                src_cls, sub = parts[2].strip().lower(), "variations"
            if src_cls is None:
                continue

            mapped = EXPLICIT_ALIASES.get(src_cls)
            if mapped is None or mapped not in CLASS_TO_ID:
                qa_rejected[f"unmapped:{src_cls}"] += 1
                continue
            cid = CLASS_TO_ID[mapped]

            arr = np.frombuffer(z.read(zp), np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
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

            raw_items.append({
                "zp": zp,
                "sub": sub,
                "cls_name": mapped,
                "cid": cid,
                "bgr": bgr,
                "dhash": compute_dhash64(gray),
                "phash": compute_phash64(gray),
                "boxes": boxes,
            })

    # Klastering near-duplicate SEBELUM pembagian 85/15
    cluster_near_duplicates(raw_items, dhash_thresh=4, phash_thresh=6)
    clusters = defaultdict(list)
    for idx, item in enumerate(raw_items):
        clusters[item["cluster_id_num"]].append(idx)

    # Deduplikasi identik eksak (dHash+pHash persis) & pisahkan cluster secara utuh per split
    rng = random.Random(seed)
    cids = sorted(clusters.keys())
    rng.shuffle(cids)
    n_val_c = max(1, round(len(cids) * 0.15))
    val_clusters = set(cids[:n_val_c])

    manifest = []
    seen_exact = set()
    dupes_dropped = 0
    split_counts = Counter()

    for c_key in cids:
        sp = "val" if c_key in val_clusters else "train"
        cluster_tag = f"jw_cl_{c_key:05d}"
        for i in clusters[c_key]:
            it = raw_items[i]
            exact_sig = (it["dhash"], it["phash"], it["cid"])
            if exact_sig in seen_exact:
                dupes_dropped += 1
                continue
            seen_exact.add(exact_sig)

            idx_num = len(manifest) + 1
            stem = f"jw_{it['sub']}_{it['cls_name']}_{idx_num:05d}"
            dst_img = os.path.join(OUT_JW, "images", sp, stem + ".jpg")
            dst_lbl = os.path.join(OUT_JW, "labels", sp, stem + ".txt")
            cv2.imwrite(dst_img, it["bgr"], [cv2.IMWRITE_JPEG_QUALITY, 94])
            write_yolo_label_file(dst_lbl, it["boxes"])

            manifest.append({
                "image_id": stem,
                "source": f"mendeley_jawa_zip:{it['sub']}",
                "license": "CC BY 4.0",
                "writer_id": cluster_tag,
                "cluster_id": cluster_tag,
                "split": sp,
                "phash": it["phash"],
                "dhash": it["dhash"],
                "image": os.path.relpath(dst_img, ROOT_DIR).replace("\\", "/"),
                "label": os.path.relpath(dst_lbl, ROOT_DIR).replace("\\", "/"),
                "boxes": it["boxes"],
                "qa_status": "pass",
            })
            split_counts[sp] += 1

    mpath = save_v4_manifest("jawa_zip_v4", manifest)
    cs_path = build_contact_sheet(
        manifest, "jawa_zip_v4 (Cluster-Disjoint Vokal A + Variations)", os.path.join(REPORT_DIR, "v4_contact_jawa_zip.jpg")
    )
    print("[jawa_zip_v4]", json.dumps({
        "manifest": os.path.relpath(mpath, ROOT_DIR),
        "raw": len(raw_items),
        "kept": len(manifest),
        "dupes_dropped": dupes_dropped,
        "splits": dict(split_counts),
        "qa_rejected": dict(qa_rejected),
    }, indent=2))
    return manifest


def ingest_jawa_syllables(seed=42):
    if not os.path.exists(SYL_BOXED_IDX):
        print("[SKIP] jawa_syllables_boxed belum ditemukan.")
        return []
    if os.path.exists(OUT_SYL):
        shutil.rmtree(OUT_SYL)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(OUT_SYL, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_SYL, "labels", sp), exist_ok=True)

    rows = [json.loads(l) for l in open(SYL_BOXED_IDX, "r", encoding="utf-8") if l.strip()]
    items = []
    qa_rejected = Counter()
    for r in rows:
        ip = os.path.join(ROOT_DIR, r["image"])
        lp = os.path.join(ROOT_DIR, r["label"])
        bgr = cv2.imread(ip)
        if bgr is None or not os.path.exists(lp):
            qa_rejected["missing_img_or_lbl"] += 1
            continue
        boxes = []
        for ln in open(lp, "r", encoding="utf-8"):
            ps = ln.strip().split()
            if len(ps) < 5:
                continue
            cid = int(float(ps[0]))
            xc, yc, bw, bh = map(float, ps[1:5])
            from schema_v4 import CLASSES_55
            if not (0 <= cid < len(CLASSES_55)):
                continue
            boxes.append({"class": CLASSES_55[cid], "cid": cid, "bbox": [xc, yc, bw, bh]})
        ok, reason = qa_validate_sample(bgr.shape, boxes)
        if not ok or not boxes:
            qa_rejected[reason] += 1
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        items.append({
            "src_img": ip,
            "src_lbl": lp,
            "base": r.get("base", ""),
            "mod": r.get("mod", ""),
            "dhash": compute_dhash64(gray),
            "phash": compute_phash64(gray),
            "boxes": boxes,
        })

    cluster_near_duplicates(items, dhash_thresh=4, phash_thresh=6)
    clusters = defaultdict(list)
    for idx, it in enumerate(items):
        clusters[it["cluster_id_num"]].append(idx)

    rng = random.Random(seed)
    cids = sorted(clusters.keys())
    rng.shuffle(cids)
    n_val_c = max(1, round(len(cids) * 0.15))
    val_clusters = set(cids[:n_val_c])

    manifest = []
    split_counts = Counter()
    for c_key in cids:
        sp = "val" if c_key in val_clusters else "train"
        cl_tag = f"jsyl_cl_{c_key:05d}"
        for idx in clusters[c_key]:
            it = items[idx]
            stem = os.path.splitext(os.path.basename(it["src_img"]))[0]
            dst_img = os.path.join(OUT_SYL, "images", sp, stem + ".jpg")
            dst_lbl = os.path.join(OUT_SYL, "labels", sp, stem + ".txt")
            shutil.copy2(it["src_img"], dst_img)
            write_yolo_label_file(dst_lbl, it["boxes"])
            manifest.append({
                "image_id": stem,
                "source": "mendeley_jawa_zip:syllables_boxed",
                "license": "CC BY 4.0",
                "writer_id": cl_tag,
                "cluster_id": cl_tag,
                "split": sp,
                "phash": it["phash"],
                "dhash": it["dhash"],
                "image": os.path.relpath(dst_img, ROOT_DIR).replace("\\", "/"),
                "label": os.path.relpath(dst_lbl, ROOT_DIR).replace("\\", "/"),
                "boxes": it["boxes"],
                "qa_status": "pass",
            })
            split_counts[sp] += 1

    mpath = save_v4_manifest("jawa_syllables_v4", manifest)
    build_contact_sheet(
        manifest, "jawa_syllables_v4 (Sandhangan Tulisan Tangan)", os.path.join(REPORT_DIR, "v4_contact_syllables.jpg")
    )
    print("[jawa_syllables_v4]", json.dumps({
        "manifest": os.path.relpath(mpath, ROOT_DIR),
        "kept": len(manifest),
        "splits": dict(split_counts),
        "qa_rejected": dict(qa_rejected),
    }, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ingest_jawa_zip(seed=args.seed)
    ingest_jawa_syllables(seed=args.seed)
