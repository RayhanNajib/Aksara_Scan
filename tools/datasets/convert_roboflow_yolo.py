"""
Konverter generik dataset deteksi objek format YOLO (Roboflow Universe / Kaggle)
menuju skema kanonik 55 kelas AksaraScan.
Menjalankan pemetaan kelas eksplisit (EXPLICIT_ALIASES), membuang kelas sumber yang tak
dapat dipetakan secara transparan ke `class_map_log.json`, dan memvalidasi setiap kotak.

Penggunaan:
  python tools/datasets/convert_roboflow_yolo.py --src dataset/external/rf_fasel_sandhangan --name rf_fasel_sandhangan
"""
import argparse
import json
import os
import shutil
import cv2
import yaml

from schema_55 import CLASS_TO_ID, EXPLICIT_ALIASES, compute_dhash64, validate_yolo_box, write_yolo_data_yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_BASE = os.path.join(ROOT_DIR, "dataset", "external", "converted_yolo")


def convert_dataset(src_dir, name):
    yaml_path = os.path.join(src_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[SKIP] {yaml_path} tidak ditemukan. Unduh manual sesuai tools/datasets/MANUAL.md terlebih dahulu.")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        src_cfg = yaml.safe_load(f)
    src_names = src_cfg.get("names", {})
    if isinstance(src_names, list):
        src_names = {i: n for i, n in enumerate(src_names)}
    else:
        src_names = {int(k): v for k, v in src_names.items()}

    mapping = {}
    dropped = {}
    for scid, sname in src_names.items():
        norm_name = str(sname).strip().lower()
        target = EXPLICIT_ALIASES.get(norm_name)
        if target and target in CLASS_TO_ID:
            mapping[scid] = (target, CLASS_TO_ID[target])
        else:
            dropped[str(sname)] = 0

    out_dir = os.path.join(OUT_BASE, name)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)

    records = []
    for src_sp, tgt_sp in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
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
            ipath = os.path.join(img_dir, fname)
            lpath = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lpath):
                continue

            bgr = cv2.imread(ipath)
            if bgr is None:
                continue
            kept_boxes = []
            with open(lpath, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    scid = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    if scid not in mapping:
                        sname = str(src_names.get(scid, scid))
                        dropped[sname] = dropped.get(sname, 0) + 1
                        continue
                    tname, tcid = mapping[scid]
                    ok, err = validate_yolo_box(tcid, xc, yc, bw, bh)
                    if not ok:
                        raise ValueError(f"Kotak tidak sah pada {ipath}: {err}")
                    kept_boxes.append({"class": tname, "cid": tcid, "bbox": [xc, yc, bw, bh]})

            if not kept_boxes:
                continue
            dst_img = os.path.join(out_dir, "images", tgt_sp, f"{name}_{fname}")
            dst_lbl = os.path.join(out_dir, "labels", tgt_sp, f"{name}_{stem}.txt")
            shutil.copy2(ipath, dst_img)
            with open(dst_lbl, "w", encoding="utf-8") as lf:
                for b in kept_boxes:
                    xc, yc, bw, bh = b["bbox"]
                    lf.write(f"{b['cid']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            records.append({
                "source": name,
                "split": tgt_sp,
                "image": os.path.relpath(dst_img, ROOT_DIR),
                "label": os.path.relpath(dst_lbl, ROOT_DIR),
                "dhash": compute_dhash64(bgr),
                "boxes": kept_boxes,
            })

    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "class_map_log.json"), "w", encoding="utf-8") as f:
        json.dump({"source": name, "mapping": {src_names[k]: v[0] for k, v in mapping.items()}, "dropped_boxes": dropped, "total_images": len(records)}, f, indent=2)
    write_yolo_data_yaml(out_dir)
    print(f"[convert_roboflow_yolo] {name}: {len(records)} citra terkonversi, dropped={dropped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    src = args.src if os.path.isabs(args.src) else os.path.join(ROOT_DIR, args.src)
    convert_dataset(src, args.name)


if __name__ == "__main__":
    main()
