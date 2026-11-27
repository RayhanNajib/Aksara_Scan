"""
Pembangun & Pengunci Set Uji Nyata Held-Out (`eval/heldout_real/`) — Gerbang 1 Mandatory.

Menghimpun >= 42 citra evaluasi bebas-kebocoran dari sumber yang TIDAK pernah terlihat di train/val:
  1. `nusaaksara_eval` (18 baris nyata Jawa ACL 2025 CC BY-SA 4.0; HANYA-EVALUASI)
  2. `setilanaji_val` (12 baris buku nyata split val yg lolos deduplikasi lintas-split)
  3. `hanacaraka_locked_12writers` (12 lembar multi-aksara dari 12 penulis manusia `test_heldout`
     dengan ragam kondisi uji: tajam, motion blur, defocus blur, pulpen biru/merah, garis tembok/buku,
     serta header Latin distractor).

Menghasilkan `eval/heldout_real/manifest.jsonl` + `eval/heldout_real/SHA256SUMS.lock`.
"""
import json
import os
import random
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools"))

from schema_v4 import CLASS_TO_ID, compute_dhash64, compute_phash64, compute_sha256_file
from v4_common import ROOT_DIR as _R, build_contact_sheet
from generate_v4_elastic_hw import (
    discover_latin_fonts,
    draw_latin_block,
    make_complex_surface_bg,
    paste_bw_patch,
    read_gray_alpha_white,
)

OUT_DIR = os.path.join(ROOT_DIR, "eval", "heldout_real")
MANIFEST_DIR = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4")

NGLEGENA_UNI = {
    "ha": "\uA9B2", "na": "\uA9A4", "ca": "\uA995", "ra": "\uA9AB", "ka": "\uA98F",
    "da": "\uA9A2", "ta": "\uA9A0", "sa": "\uA9B1", "wa": "\uA9AE", "la": "\uA9AD",
    "pa": "\uA9A5", "dha": "\uA99D", "ja": "\uA997", "ya": "\uA9AA", "nya": "\uA99A",
    "ma": "\uA9A9", "ga": "\uA992", "ba": "\uA9A7", "tha": "\uA99B", "nga": "\uA994",
}


def build_heldout(seed=777):
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    rng = random.Random(seed)
    records = []

    # 1. Subset A: 18 citra nyata dari NusaAksara (HANYA-EVALUASI, CC BY-SA 4.0)
    nusa_p = os.path.join(MANIFEST_DIR, "nusaaksara_eval_only_v4.jsonl")
    if os.path.exists(nusa_p):
        nusa_rows = [json.loads(l) for l in open(nusa_p, "r", encoding="utf-8") if l.strip()]
        chosen = nusa_rows[:18]
        for idx, r in enumerate(chosen):
            src = os.path.join(ROOT_DIR, r["image"])
            dst = os.path.join(OUT_DIR, "images", f"hr_nusa_{idx:02d}.png")
            shutil.copy2(src, dst)
            bgr = cv2.imread(dst)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            records.append({
                "id": f"hr_nusa_{idx:02d}",
                "image": os.path.relpath(dst, ROOT_DIR).replace("\\", "/"),
                "subset": "nusaaksara_acl2025_eval_only",
                "license": "CC BY-SA 4.0 (HANYA-EVALUASI)",
                "condition": "real_printed_or_manuscript",
                "writer_id": r["writer_id"],
                "cluster_id": r["cluster_id"],
                "split": "heldout_real",
                "unicode_gt": r["unicode"],
                "sha256": compute_sha256_file(dst),
                "phash": compute_phash64(gray),
                "dhash": compute_dhash64(gray),
                "label_preview": f"nusa_{idx:02d}",
            })

    # 2. Subset B: 12 citra buku nyata setilanaji split test_heldout (terisolasi dari train & val)
    seti_p = os.path.join(MANIFEST_DIR, "setilanaji_lines_v4.jsonl")
    if os.path.exists(seti_p):
        ho_rows = [json.loads(l) for l in open(seti_p, "r", encoding="utf-8") if l.strip() and '"test_heldout"' in l]
        for idx, r in enumerate(ho_rows[:12]):
            src = os.path.join(ROOT_DIR, r["image"])
            dst = os.path.join(OUT_DIR, "images", f"hr_seti_{idx:02d}.jpg")
            shutil.copy2(src, dst)
            bgr = cv2.imread(dst)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            records.append({
                "id": f"hr_seti_{idx:02d}",
                "image": os.path.relpath(dst, ROOT_DIR).replace("\\", "/"),
                "subset": "setilanaji_real_book_val",
                "license": "Apache-2.0",
                "condition": "real_scanned_book_line",
                "writer_id": r["writer_id"],
                "cluster_id": r["cluster_id"],
                "split": "heldout_real",
                "unicode_gt": r["unicode"],
                "sha256": compute_sha256_file(dst),
                "phash": compute_phash64(gray),
                "dhash": compute_dhash64(gray),
                "label_preview": f"seti_{idx:02d}",
            })

    # 3. Subset C: 12 lembar tulisan tangan multi-aksara dari 12 penulis manusia TERKUNCI (`test_heldout` hanacaraka)
    hnc_p = os.path.join(MANIFEST_DIR, "hanacaraka_v4.jsonl")
    latin_fonts = discover_latin_fonts()
    if os.path.exists(hnc_p):
        by_writer = defaultdict(list)
        for ln in open(hnc_p, "r", encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") == "test_heldout":
                by_writer[r["writer_id"]].append(r)

        conditions = [
            ("sharp_clean", "aged_paper", (20, 20, 20), "none"),
            ("sharp_clean", "plain", (25, 25, 25), "none"),
            ("motion_blur", "ruled_notebook", (175, 55, 22), "motion"),
            ("motion_blur", "aged_paper", (25, 25, 25), "motion"),
            ("defocus_blur", "grid_paper", (32, 32, 175), "defocus"),
            ("defocus_blur", "wall_lines", (40, 120, 35), "defocus"),
            ("multi_color_pen", "ruled_notebook", (185, 62, 20), "none"),
            ("multi_color_pen", "aged_paper", (35, 32, 185), "none"),
            ("wall_lines_shadow", "wall_lines", (28, 28, 28), "none"),
            ("wall_lines_shadow", "shadow_crease", (165, 52, 24), "none"),
            ("latin_distractor_mix", "ruled_notebook", (24, 24, 24), "none"),
            ("latin_distractor_blur", "wall_lines", (170, 58, 26), "defocus"),
        ]

        sorted_writers = sorted(by_writer.keys())[:12]
        for w_idx, wid in enumerate(sorted_writers):
            samples = by_writer[wid]
            rng.shuffle(samples)
            chosen_glyphs = samples[:min(10, len(samples))]
            rows_n = 2 if len(chosen_glyphs) > 5 else 1
            cols_n = 5
            cw, ch = 128, 114
            W, H = cols_n * cw + 60, rows_n * ch + 90
            cond_name, bg_kind, ink_bgr, blur_kind = conditions[w_idx % len(conditions)]
            if bg_kind == "plain":
                bg = np.full((H, W, 3), 246, dtype=np.uint8)
            else:
                bg, _, _ = make_complex_surface_bg(W, H, rng, style_force=bg_kind)
            ink = np.array(ink_bgr, dtype=np.float32)

            if "latin_distractor" in cond_name and latin_fonts:
                draw_latin_block(bg, f"Penulis #{w_idx + 1} · Aksara Jawa", 28, 12, latin_fonts[0], ink)

            uni_lines = []
            gt_boxes = []
            for gi, g_rec in enumerate(chosen_glyphs):
                rr, cc = gi // cols_n, gi % cols_n
                while len(uni_lines) <= rr:
                    uni_lines.append("")
                cls_name = g_rec["boxes"][0]["class"]
                uni_lines[rr] += NGLEGENA_UNI.get(cls_name, "")
                raw_g = read_gray_alpha_white(os.path.join(ROOT_DIR, g_rec["image"]))
                if raw_g is None:
                    continue
                _, bw = cv2.threshold(raw_g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ys, xs = np.where(bw > 0)
                if len(ys) < 12:
                    continue
                crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                sc = 64.0 / max(12, crop.shape[0])
                nw = max(10, min(96, int(round(crop.shape[1] * sc))))
                nh = max(10, min(88, int(round(crop.shape[0] * sc))))
                rs = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                x0 = 30 + cc * cw + (cw - nw) // 2
                y0 = 56 + rr * ch + (ch - nh) // 2
                if paste_bw_patch(bg, rs, ink, x0, y0):
                    gt_boxes.append({
                        "class": cls_name,
                        "cid": CLASS_TO_ID[cls_name],
                        "bbox": [((x0 + x0 + nw) / 2.0) / W, ((y0 + y0 + nh) / 2.0) / H, nw / W, nh / H],
                    })

            if blur_kind == "motion":
                ker = np.zeros((5, 5), dtype=np.float32)
                ker[2, :] = 0.2
                bg = cv2.filter2D(bg, -1, ker)
            elif blur_kind == "defocus":
                bg = cv2.GaussianBlur(bg, (5, 5), 1.35)

            dst = os.path.join(OUT_DIR, "images", f"hr_hw_{w_idx:02d}_{cond_name}.jpg")
            cv2.imwrite(dst, bg, [cv2.IMWRITE_JPEG_QUALITY, 90])
            gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
            records.append({
                "id": f"hr_hw_{w_idx:02d}",
                "image": os.path.relpath(dst, ROOT_DIR).replace("\\", "/"),
                "subset": "hanacaraka_12_unseen_human_writers",
                "license": "CC0-1.0",
                "condition": cond_name,
                "writer_id": wid,
                "cluster_id": chosen_glyphs[0]["cluster_id"],
                "split": "heldout_real",
                "unicode_gt": "\n".join(uni_lines),
                "boxes": gt_boxes,
                "sha256": compute_sha256_file(dst),
                "phash": compute_phash64(gray),
                "dhash": compute_dhash64(gray),
                "label_preview": f"{cond_name[:12]}",
            })

    m_path = os.path.join(OUT_DIR, "manifest.jsonl")
    with open(m_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Kunci checksum SHA256SUMS.lock untuk seluruh berkas heldout_real
    lock_path = os.path.join(OUT_DIR, "SHA256SUMS.lock")
    with open(lock_path, "w", encoding="utf-8") as lf:
        lf.write(f"{compute_sha256_file(m_path)}  manifest.jsonl\n")
        for r in records:
            rel_img = os.path.relpath(os.path.join(ROOT_DIR, r["image"]), OUT_DIR).replace("\\", "/")
            lf.write(f"{r['sha256']}  {rel_img}\n")

    build_contact_sheet(
        records, "eval/heldout_real (42 Citra Multi-Penulis & Nyata Terkunci)",
        os.path.join(ROOT_DIR, "dataset", "external", "reports", "v4_contact_heldout_real.jpg"),
        max_n=42, cols=7, cell_w=175, cell_h=120,
    )
    print(json.dumps({
        "heldout_manifest": os.path.relpath(m_path, ROOT_DIR),
        "lock_file": os.path.relpath(lock_path, ROOT_DIR),
        "total_images": len(records),
        "unique_writers_or_docs": len({r["writer_id"] for r in records}),
    }, indent=2))


if __name__ == "__main__":
    build_heldout()
