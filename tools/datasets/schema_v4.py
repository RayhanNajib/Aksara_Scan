"""
Skema Kelas Kanonik v4, Perceptual Hashing ganda (pHash64 DCT + dHash64),
Validasi Kotak QA Ketat, dan Klastering Duplikat Dekat (Disjoint Clusters).
"""
import hashlib
import os
import sys
import yaml
import cv2
import numpy as np

from schema_55 import (
    CLASSES_55,
    CLASS_TO_ID,
    EXPLICIT_ALIASES,
    EXPLICIT_ALIASES as ALIASES_55,
    compute_dhash64,
    compute_tight_foreground_yolo_box,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_YAML_PATH = os.path.join(ROOT_DIR, "dataset", "schema", "classes_v4.yaml")


def load_schema_v4():
    with open(SCHEMA_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_phash64(img_bgr_or_gray):
    """DCT-based 64-bit perceptual hash untuk deteksi near-duplicate lintas penskalaan/blur ringan."""
    if img_bgr_or_gray is None:
        return None
    gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY) if img_bgr_or_gray.ndim == 3 else img_bgr_or_gray
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].flatten()
    # Gunakan median tanpa komponen DC (index 0)
    med = float(np.median(low[1:]))
    bits = low > med
    val = 0
    for b in bits:
        val = (val << 1) | (1 if b else 0)
    return f"{val:016x}"


def compute_sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hamming_hex(h1, h2):
    if not h1 or not h2:
        return 64
    return (int(h1, 16) ^ int(h2, 16)).bit_count()


def qa_validate_sample(img_shape, boxes, min_box_px=3, max_aspect=14.0):
    """
    Audit QA anotasi v4:
      - Menolak kotak keluar batas gambar
      - Menolak kotak sangat kecil (< 3px), kosong, atau rasio aspek ekstrem (> 14:1)
      - Menolak class_id tak dikenal
    """
    H, W = img_shape[:2]
    if H < 8 or W < 8:
        return False, f"dimensi gambar terlalu kecil ({W}x{H})"
    for idx, b in enumerate(boxes):
        cid = int(b["cid"])
        if not (0 <= cid < len(CLASSES_55)):
            return False, f"box[{idx}] class_id {cid} tidak dikenal"
        xc, yc, bw, bh = map(float, b["bbox"])
        if bw <= 0 or bh <= 0:
            return False, f"box[{idx}] ukuran nol/negatif"
        x0, y0 = xc - bw / 2.0, yc - bh / 2.0
        x1, y1 = xc + bw / 2.0, yc + bh / 2.0
        if x0 < -0.002 or y0 < -0.002 or x1 > 1.002 or y1 > 1.002:
            return False, f"box[{idx}] keluar gambar [{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}]"
        pw, ph = bw * W, bh * H
        if pw < min_box_px or ph < min_box_px:
            return False, f"box[{idx}] terlalu kecil {pw:.1f}x{ph:.1f}px"
        ar = max(pw / max(1e-3, ph), ph / max(1e-3, pw))
        if ar > max_aspect:
            return False, f"box[{idx}] rasio aspek ekstrem ({ar:.1f}:1)"
    return True, "ok"


def cluster_near_duplicates(records, dhash_thresh=3, phash_thresh=4):
    """
    Mengelompokkan sampel near-duplicate lewat Union-Find menggunakan kombinasi KETAT
    (dHash <= dhash_thresh DAN pHash <= phash_thresh) agar tidak terjadi transitive chaining
    palsu antar penulis berbeda yang menggambar di latar putih polos.
    """
    n = len(records)
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

    exact_pair = {}
    prefix_buckets_d = {}

    for i, r in enumerate(records):
        dh = r.get("dhash")
        ph = r.get("phash")
        if not dh or not ph:
            continue
        sig = (dh, ph)
        if sig in exact_pair:
            union(i, exact_pair[sig])
        else:
            exact_pair[sig] = i
        p16 = dh[:4]
        for j in prefix_buckets_d.get(p16, ()):
            rj = records[j]
            if hamming_hex(dh, rj["dhash"]) <= dhash_thresh and hamming_hex(ph, rj["phash"]) <= phash_thresh:
                union(i, j)
        prefix_buckets_d.setdefault(p16, []).append(i)

    root_to_cid = {}
    for i, r in enumerate(records):
        rt = find(i)
        if rt not in root_to_cid:
            root_to_cid[rt] = len(root_to_cid)
        r["cluster_id_num"] = root_to_cid[rt]
    return records
