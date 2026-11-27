"""
Ingest HANYA-EVALUASI untuk subset Aksara Jawa (`script == 'jawa'`) dari
`NusaAksara/NusaAksara` (ACL 2025, CC BY-SA 4.0).

ATURAN KERAS LISENSI:
  - Diberi tanda eksplisit `split: eval_only_external`, `usage: HANYA-EVALUASI`.
  - TIDAK BOLEH dipakai untuk melatih bobot model MIT AksaraScan.
  - Dimanfaatkan sebagai tolok ukur eksternal standar & bagian eval/heldout_real.
"""
import argparse
import json
import os
import urllib.request
from collections import Counter

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from schema_v4 import compute_dhash64, compute_phash64, compute_sha256_file
from v4_common import ROOT_DIR, REPORT_DIR, build_contact_sheet, save_v4_manifest

OUT_DIR = os.path.join(ROOT_DIR, "dataset", "external", "eval_only_nusaaksara")


def main(max_samples=48):
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    parquet_path = hf_hub_download(
        "NusaAksara/NusaAksara",
        "Image Transcription (OCR)/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    rows = [r for r in pq.read_table(parquet_path).to_pylist() if r.get("script") == "jawa"]

    manifest = []
    skipped = 0
    for idx, row in enumerate(rows):
        if len(manifest) >= max_samples:
            break
        url = row.get("image")
        uni = (row.get("transcription") or "").strip()
        if not url or not uni:
            skipped += 1
            continue
        dst_img = os.path.join(OUT_DIR, "images", f"nusa_jawa_{idx:04d}.png")
        if not os.path.exists(dst_img) or os.path.getsize(dst_img) < 100:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AksaraScan/4.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                arr = np.frombuffer(data, np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is None or min(bgr.shape[:2]) < 12:
                    skipped += 1
                    continue
                cv2.imwrite(dst_img, bgr)
            except Exception:
                skipped += 1
                continue
        else:
            bgr = cv2.imread(dst_img)
            if bgr is None:
                skipped += 1
                continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        manifest.append({
            "image_id": f"nusa_jawa_{idx:04d}",
            "source": "hf_nusaaksara_acl2025",
            "license": "CC BY-SA 4.0",
            "usage": "HANYA-EVALUASI",
            "writer_id": f"nusa_doc_{os.path.basename(url).split('_')[1] if '_' in url else idx}",
            "cluster_id": f"nusa_cl_{idx:04d}",
            "split": "eval_only_external",
            "unicode": uni,
            "source_url": url,
            "sha256": compute_sha256_file(dst_img),
            "phash": compute_phash64(gray),
            "dhash": compute_dhash64(gray),
            "image": os.path.relpath(dst_img, ROOT_DIR).replace("\\", "/"),
            "boxes": [],
            "label_preview": f"nusa({len(uni)}ch)",
            "qa_status": "pass",
        })

    mpath = save_v4_manifest("nusaaksara_eval_only_v4", manifest)
    cs_path = build_contact_sheet(
        manifest, "NusaAksara Jawa (HANYA-EVALUASI)", os.path.join(REPORT_DIR, "v4_contact_nusaaksara_eval.jpg")
    )
    print(json.dumps({
        "manifest": os.path.relpath(mpath, ROOT_DIR),
        "total_jawa_in_hf": len(rows),
        "downloaded_for_eval": len(manifest),
        "skipped": skipped,
        "contact_sheet": os.path.relpath(cs_path, ROOT_DIR) if cs_path else None,
    }, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=48)
    args = ap.parse_args()
    main(max_samples=args.max_samples)
