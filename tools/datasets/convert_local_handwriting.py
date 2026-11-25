"""
Konverter dataset lokal:
  1. `hanacaraka_vzrenggamani` (dataset/split/ atau dataset/hanacaraka/) -> skema YOLO 55 kelas
     Dengan tetap mempertahankan pemisahan asli train / validation / test (tidak mencampur split!).
  2. `mendeley_indonesian_scripts_jawa` (Jawa.zip) ->
       - `Vokal A` (9.994) + `variations` (1.066) -> skema YOLO 55 kelas (0..19 nglegena)
       - `Vokal E/I/O/U/È` (2.196) -> diekspor ke manifes `syllable_pool.jsonl` (bukan kotak tunggal YOLO karena berisi gabungan base + sandhangan)

Keluaran di `dataset/external/converted_yolo/{hanacaraka,jawa_zip}/`:
  - `images/{train,val}/...` + `labels/{train,val}/...` + `index.jsonl` + `class_map_log.json`
"""
import csv
import json
import os
import random
import shutil
import zipfile

import cv2
import numpy as np

from schema_55 import (
    CLASSES_55,
    CLASS_TO_ID,
    EXPLICIT_ALIASES,
    compute_dhash64,
    compute_tight_foreground_yolo_box,
    validate_yolo_box,
    write_yolo_data_yaml,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_BASE = os.path.join(ROOT_DIR, "dataset", "external", "converted_yolo")


def convert_hanacaraka():
    out_dir = os.path.join(OUT_BASE, "hanacaraka")
    for s in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", s), exist_ok=True)

    manifest_csv = os.path.join(ROOT_DIR, "dataset", "manifest.csv")
    split_root = os.path.join(ROOT_DIR, "dataset", "split")
    records = []
    dropped = {}
    split_map = {"train": "train", "validation": "val", "test": "test"}

    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row["path"]
            src_cls = row["class"].strip().lower()
            src_split = row["split"].strip().lower()
            tgt_split = split_map.get(src_split, "train")

            mapped = EXPLICIT_ALIASES.get(src_cls)
            if mapped is None or mapped not in CLASS_TO_ID:
                dropped[src_cls] = dropped.get(src_cls, 0) + 1
                continue

            cid = CLASS_TO_ID[mapped]
            src_img = os.path.join(split_root, rel_path)
            bgr = cv2.imread(src_img)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            xc, yc, bw_n, bh_n = compute_tight_foreground_yolo_box(gray, pad=6)
            ok, err = validate_yolo_box(cid, xc, yc, bw_n, bh_n)
            if not ok:
                raise ValueError(f"Validasi gagal pada {src_img}: {err}")

            stem = f"hnc_{mapped}_{os.path.splitext(os.path.basename(rel_path))[0]}"
            dst_img = os.path.join(out_dir, "images", tgt_split, stem + ".png")
            dst_lbl = os.path.join(out_dir, "labels", tgt_split, stem + ".txt")
            shutil.copy2(src_img, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                lf.write(f"{cid} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")

            records.append({
                "source": "hanacaraka_vzrenggamani",
                "split": tgt_split,
                "image": os.path.relpath(dst_img, ROOT_DIR),
                "label": os.path.relpath(dst_lbl, ROOT_DIR),
                "dhash": compute_dhash64(gray),
                "boxes": [{"class": mapped, "cid": cid, "bbox": [xc, yc, bw_n, bh_n]}],
            })

    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "class_map_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": "hanacaraka_vzrenggamani",
            "mapped_classes": sorted({b["class"] for r in records for b in r["boxes"]}),
            "dropped_classes": dropped,
            "total_images": len(records),
        }, f, indent=2)
    write_yolo_data_yaml(out_dir)
    print(f"[convert_hanacaraka] {len(records)} citra dikonversi, dropped={dropped}")


def convert_jawa_zip():
    zip_path = os.path.join(ROOT_DIR, "dataset", "external", "jawa_mendeley", "Jawa.zip")
    if not os.path.exists(zip_path):
        print(f"[SKIP] Jawa.zip belum ada di {zip_path}")
        return

    out_dir = os.path.join(OUT_BASE, "jawa_zip")
    for s in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", s), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", s), exist_ok=True)

    syllable_dir = os.path.join(ROOT_DIR, "dataset", "external", "jawa_syllables")
    os.makedirs(syllable_dir, exist_ok=True)

    records = []
    syllable_records = []
    dropped = {}
    rng = random.Random(42)

    with zipfile.ZipFile(zip_path, "r") as z:
        all_names = [n for n in z.namelist() if not n.endswith("/") and n.lower().endswith((".png", ".jpg", ".jpeg"))]
        # Kelompokkan per sumber kelas & subdir
        nglegena_items = []
        for n in all_names:
            parts = n.split("/")
            if len(parts) >= 5 and parts[1] == "all_class" and parts[2] == "Vokal A":
                nglegena_items.append((parts[3].strip().lower(), n, "vokal_a"))
            elif len(parts) >= 4 and parts[1] == "variations":
                nglegena_items.append((parts[2].strip().lower(), n, "variations"))
            elif len(parts) >= 5 and parts[1] == "all_class" and parts[2] in ("Vokal E", "Vokal I", "Vokal O", "Vokal U", "Vokal È"):
                syllable_records.append((parts[2], parts[3], n))

        # Split deterministik 85/15 per kelas untuk Jawa.zip (train vs val terpisah tanpa kebocoran!)
        by_cls = {}
        for src_cls, zp, sub in nglegena_items:
            by_cls.setdefault(src_cls, []).append((zp, sub))

        idx_counter = 0
        for src_cls, entries in sorted(by_cls.items()):
            mapped = EXPLICIT_ALIASES.get(src_cls)
            if mapped is None or mapped not in CLASS_TO_ID:
                dropped[src_cls] = len(entries)
                continue
            cid = CLASS_TO_ID[mapped]
            entries.sort()
            rng.shuffle(entries)
            n_val = max(1, round(len(entries) * 0.15))
            val_set = set(x[0] for x in entries[:n_val])

            for zp, sub in entries:
                raw_bytes = z.read(zp)
                arr = np.frombuffer(raw_bytes, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                xc, yc, bw_n, bh_n = compute_tight_foreground_yolo_box(gray, pad=6)
                ok, err = validate_yolo_box(cid, xc, yc, bw_n, bh_n)
                if not ok:
                    raise ValueError(f"Validasi gagal pada {zp}: {err}")

                tgt_split = "val" if zp in val_set else "train"
                idx_counter += 1
                stem = f"jw_{sub}_{mapped}_{idx_counter:05d}"
                dst_img = os.path.join(out_dir, "images", tgt_split, stem + ".jpg")
                dst_lbl = os.path.join(out_dir, "labels", tgt_split, stem + ".txt")
                cv2.imwrite(dst_img, bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
                with open(dst_lbl, "w", encoding="utf-8") as lf:
                    lf.write(f"{cid} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")

                records.append({
                    "source": f"mendeley_jawa_zip:{sub}",
                    "split": tgt_split,
                    "image": os.path.relpath(dst_img, ROOT_DIR),
                    "label": os.path.relpath(dst_lbl, ROOT_DIR),
                    "dhash": compute_dhash64(gray),
                    "boxes": [{"class": mapped, "cid": cid, "bbox": [xc, yc, bw_n, bh_n]}],
                })

        # Ekspor suku kata ber-sandhangan (Vokal E/I/O/U/È) untuk Track B / referensi tulisan tangan
        syl_manifest = []
        vokal_to_mod = {
            "Vokal E": "taling",
            "Vokal I": "wulu",
            "Vokal O": "taling_tarung",
            "Vokal U": "suku",
            "Vokal È": "pepet",
        }
        for s_idx, (v_dir, folder_cls, zp) in enumerate(syllable_records):
            out_s = os.path.join(syllable_dir, f"syl_{s_idx:04d}_{v_dir.replace(' ', '')}.jpg")
            if not os.path.exists(out_s):
                arr = np.frombuffer(z.read(zp), np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    cv2.imwrite(out_s, bgr, [cv2.IMWRITE_JPEG_QUALITY, 94])
            syl_manifest.append({
                "source": "mendeley_jawa_zip:sandhangan_syllable",
                "vokal_group": v_dir,
                "folder_label": folder_cls,
                "expected_modifier": vokal_to_mod[v_dir],
                "image": os.path.relpath(out_s, ROOT_DIR),
            })
        with open(os.path.join(syllable_dir, "syllable_pool.jsonl"), "w", encoding="utf-8") as f:
            for r in syl_manifest:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "class_map_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source": "mendeley_indonesian_scripts_jawa",
            "mapped_atomic_nglegena_images": len(records),
            "exported_sandhangan_syllable_images": len(syl_manifest),
            "dropped_classes": dropped,
            "note": "Vokal E/I/O/U/È (2.196 citra) adalah kluster gabungan induk+sandhangan tanpa kotak terpisah, disimpan terpisah di dataset/external/jawa_syllables/ agar tidak mengotori kotak atomik.",
        }, f, ensure_ascii=False, indent=2)
    write_yolo_data_yaml(out_dir)
    print(f"[convert_jawa_zip] {len(records)} nglegena atomik (85/15 split) + {len(syl_manifest)} citra suku kata ber-sandhangan diekspor.")


if __name__ == "__main__":
    convert_hanacaraka()
    convert_jawa_zip()
