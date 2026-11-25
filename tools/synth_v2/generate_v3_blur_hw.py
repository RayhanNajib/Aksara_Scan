"""
AksaraScan v3.1 — Generator Data Sintetis & Komposit Tulisan Tangan Tahan Blur (Synth v3)
Perluasan kritis di atas v2:
  1. Menggunakan 11 font SIL OFL 1.1 & Apache-2.0 terverifikasi:
     - 4 font full OpenType (NotoSansJavanese Regular & Bold, TuladhaJejegOT, Sehulbari) untuk semua kluster termasuk pasangan
     - 7 font kaligrafi/cetak SIL OFL (Bharatayuddha, Chandrasa, Righma_Ciddhi, TUNJUNG_BIRU, Tunjung_putih, Warastra, YBG_Kramawirya) untuk kalimat bebas-pasangan
  2. Memuat SELURUH pool tulisan tangan nyata (`hanacaraka` CC0 + seluruh ~9.4k `jawa_zip` CC BY 4.0 + `jawa_syllables_boxed` sandhangan nyata!) tanpa batas 180/kelas.
  3. Layout Grid Tulisan Tangan Lebar (2x5 & 4x5, garis tipis 1px, bayangan gelap, bingkai tepi scanner, + label Latin di atas aksara) agar tahan skenario `sample_1/2/3`.
  4. Augmentasi Blur & Optik Kuat: Gaussian defocus blur, directional motion blur, simulasi resolusi rendah (down-up), tinta pudar pensil/pulpen biru/gel.
  5. Sampel Negatif Keras (Hard Negatives) teks Latin murni & kertas kosong (file label 0 baris) agar header Latin tidak memicu deteksi palsu.
"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))

from schema_55 import CLASSES_55, CLASS_TO_ID, NGLEGENA, validate_yolo_box, write_yolo_data_yaml
from orthography_rules import PANGKON, NGLEGENA_LIST, validate_javanese_unicode
from generate_v2 import (
    BASE_CH_TO_NAME, render_cluster_cached, create_paper_background, select_weighted_sentences
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")

FULL_OT_FONTS = {
    "NotoSansJavanese.ttf",
    "NotoSansJavanese-Bold.ttf",
    "TuladhaJejegOT-Regular.ttf",
    "TuladhaJejegOT-Regular.otf",
    "Sehulbari_Arif_Budiarto.ttf",
}

ALL_NGLEGENA_NAMES = [n for n, _ in NGLEGENA]


def discover_expanded_fonts():
    font_dir = os.path.join(EXT_DIR, "fonts")
    full_fonts, nopas_fonts = [], []
    for fn in sorted(os.listdir(font_dir)):
        if not fn.lower().endswith((".ttf", ".otf")) or fn == "NAYAKA.ttf":
            continue
        fp = os.path.join(font_dir, fn)
        is_full = fn in FULL_OT_FONTS
        for sz in (46, 56, 66):
            try:
                pf = ImageFont.truetype(fp, sz, layout_engine=ImageFont.Layout.RAQM)
                item = (fn, sz, pf)
                if is_full:
                    full_fonts.append(item)
                else:
                    nopas_fonts.append(item)
            except Exception:
                pass
    return full_fonts, nopas_fonts


def load_all_handwriting_pools():
    """Muat SELURUH pool tulisan tangan nyata per split tanpa cap 180."""
    pools = {"train": {n: [] for n in ALL_NGLEGENA_NAMES}, "val": {n: [] for n in ALL_NGLEGENA_NAMES}}
    for src in ("hanacaraka", "jawa_zip"):
        idx_p = os.path.join(EXT_DIR, "converted_yolo", src, "index_deduped.jsonl")
        if not os.path.exists(idx_p):
            idx_p = os.path.join(EXT_DIR, "converted_yolo", src, "index.jsonl")
        if not os.path.exists(idx_p):
            continue
        with open(idx_p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sp = rec["split"]
                if sp not in ("train", "val"):
                    continue
                cls_name = rec["boxes"][0]["class"]
                if cls_name not in pools[sp]:
                    continue
                ipath = os.path.join(ROOT_DIR, rec["image"])
                gray = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ys, xs = np.where(bw > 0)
                if len(ys) < 18:
                    continue
                crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                pools[sp][cls_name].append((crop, src))
    return pools


def load_syllable_pools():
    """Muat potongan silabus tulisan tangan nyata bersandhangan (`jawa_syllables_boxed`)."""
    idx_p = os.path.join(EXT_DIR, "converted_yolo", "jawa_syllables_boxed", "index.jsonl")
    pools = {"train": [], "val": []}
    if not os.path.exists(idx_p):
        return pools
    with open(idx_p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sp = rec.get("split")
            if sp not in ("train", "val"):
                continue
            ip = os.path.join(ROOT_DIR, rec["image"])
            lp = os.path.join(ROOT_DIR, rec["label"])
            gray = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if gray is None or not os.path.exists(lp):
                continue
            h, w = gray.shape[:2]
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            boxes = []
            with open(lp, "r", encoding="utf-8") as lf:
                for ln in lf:
                    ps = ln.strip().split()
                    if len(ps) < 5:
                        continue
                    cid = int(float(ps[0]))
                    xc, yc, bwn, bhn = map(float, ps[1:5])
                    boxes.append((cid, (xc - bwn / 2) * w, (yc - bhn / 2) * h, (xc + bwn / 2) * w, (yc + bhn / 2) * h))
            if not boxes:
                continue
            x0 = max(0, int(min(b[1] for b in boxes)) - 2)
            y0 = max(0, int(min(b[2] for b in boxes)) - 2)
            x1 = min(w, int(max(b[3] for b in boxes)) + 2)
            y1 = min(h, int(max(b[4] for b in boxes)) + 2)
            patch = bw[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            rel = [(cid, bx0 - x0, by0 - y0, bx1 - x0, by1 - y0) for cid, bx0, by0, bx1, by1 in boxes]
            pools[sp].append((patch, rel))
    return pools


def random_ink_color(rng):
    kind = rng.choice(["black", "black", "faded_pencil", "thin_grey", "blue_pen"])
    if kind == "black":
        v = rng.randint(8, 45)
        return np.array([v, v, v], dtype=np.float32)
    if kind == "faded_pencil":
        v = rng.randint(85, 138)
        return np.array([v, v, v], dtype=np.float32)
    if kind == "thin_grey":
        v = rng.randint(60, 110)
        return np.array([v, v, v], dtype=np.float32)
    return np.array([rng.randint(120, 178), rng.randint(32, 74), rng.randint(15, 52)], dtype=np.float32)


def paste_bw_patch(canvas, patch, ink_col, x0, y0):
    H, W = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + pw > W or y0 + ph > H:
        return False
    alpha = (patch.astype(np.float32) / 255.0)[..., None]
    roi = canvas[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    canvas[y0:y0 + ph, x0:x0 + pw] = np.clip(roi * (1.0 - alpha) + ink_col * alpha, 0, 255).astype(np.uint8)
    return True


def make_thin_or_thick_stroke(patch, rel_boxes, rng, force_thin=False):
    r = rng.random()
    aug = patch.copy()
    if force_thin or r < 0.30:
        # Simulasi goresan tipis 1-2px ala pensil/sample_1/2/3 (skelet-ish via erode ringan)
        k = rng.choice([2, 3])
        er = cv2.erode(aug, np.ones((k, k), np.uint8), iterations=1)
        if (er > 0).sum() >= max(12, int((aug > 0).sum() * 0.25)):
            aug = er
    elif r > 0.72:
        kx, ky = rng.choice([(2, 2), (3, 2), (2, 3)])
        aug = cv2.dilate(aug, np.ones((ky, kx), np.uint8), iterations=1)
    return aug, rel_boxes


def apply_strong_blur_and_optics(canvas, labels_abs, rng):
    H, W = canvas.shape[:2]
    # 1. Bayangan / gradien pencahayaan tidak merata (lebih realistis termasuk shadow gelap tajam)
    if rng.random() < 0.75:
        gx = np.linspace(rng.uniform(0.42, 1.0), rng.uniform(0.82, 1.06), W, dtype=np.float32)
        gy = np.linspace(rng.uniform(0.55, 1.0), rng.uniform(0.84, 1.06), H, dtype=np.float32)
        shad = np.outer(gy, gx)
        if rng.random() < 0.4:
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            yy, xx = np.ogrid[:H, :W]
            dist = ((xx - cx) / max(1, W * 0.38)) ** 2 + ((yy - cy) / max(1, H * 0.38)) ** 2
            shad *= np.clip(0.45 + 0.55 * np.tanh(dist - 0.25), 0.38, 1.0).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) * shad[..., None], 0, 255).astype(np.uint8)

    # 2. Rotasi ringan ±5 deg
    if rng.random() < 0.45 and labels_abs:
        angle = rng.uniform(-4.5, 4.5)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
        canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        new_lbls = []
        for cid, x0, y0, x1, y1 in labels_abs:
            pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float32)
            tpts = (M @ pts.T).T
            nx0, ny0 = max(0.0, float(tpts[:, 0].min())), max(0.0, float(tpts[:, 1].min()))
            nx1, ny1 = min(float(W), float(tpts[:, 0].max())), min(float(H), float(tpts[:, 1].max()))
            if nx1 - nx0 >= 3.0 and ny1 - ny0 >= 3.0:
                new_lbls.append((cid, nx0, ny0, nx1, ny1))
        labels_abs = new_lbls

    # 3. Blur Kuat Terarah (58% sampel!): Gaussian Defocus, Motion Blur, dan Pixelation Down-Up Blur
    r_blur = rng.random()
    if r_blur < 0.28:
        k = rng.choice([3, 5, 7])
        sig = rng.uniform(0.65, 1.75)
        canvas = cv2.GaussianBlur(canvas, (k, k), sig)
    elif r_blur < 0.46:
        k = rng.choice([3, 5, 7])
        ker = np.zeros((k, k), dtype=np.float32)
        kind_m = rng.choice(["h", "v", "d1", "d2"])
        if kind_m == "h":
            ker[k // 2, :] = 1.0 / k
        elif kind_m == "v":
            ker[:, k // 2] = 1.0 / k
        elif kind_m == "d1":
            np.fill_diagonal(ker, 1.0 / k)
        else:
            np.fill_diagonal(np.fliplr(ker), 1.0 / k)
        canvas = cv2.filter2D(canvas, -1, ker)
    elif r_blur < 0.58:
        # Kamera blur karena zoom/resolusi rendah lalu diperbesar kembali
        down = rng.uniform(0.42, 0.72)
        dw, dh = max(32, int(W * down)), max(32, int(H * down))
        small = cv2.resize(canvas, (dw, dh), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)

    # 4. Sensor noise
    if rng.random() < 0.45:
        sig = rng.uniform(2.0, 7.0)
        noise = np.random.normal(0, sig, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return canvas, labels_abs


def gen_handwriting_grid_sample(hw_pool, syl_pool, rng):
    """Bangun lembar tulisan tangan 2x5 atau 4x5 (seperti sample_1/2/3) + variasi goresan tipis + header Latin."""
    is_4x5 = rng.random() < 0.45
    rows_n = 4 if is_4x5 else 2
    cols_n = 5
    W = rng.choice([880, 1080, 1280])
    H = rng.choice([760, 920, 1060]) if is_4x5 else rng.choice([540, 680, 780])
    canvas, _ = create_paper_background(W, H, rng, enable_bg=True)
    ink_col = random_ink_color(rng)

    add_headers = rng.random() < 0.55
    add_caption = rng.random() < 0.45
    if add_caption:
        cap = rng.choice([
            "AksaraScan - tampilan web",
            "Gambar 2. Sampel nyata 20 aksara Jawa Nglegena",
            "Latihan Menulis Aksara Jawa Hanacaraka",
            "Pemuatan gambar dari file / kamera HP",
        ])
        cv2.putText(canvas, cap, (rng.randint(18, 45), rng.randint(32, 52)),
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.55, 0.85), (40, 40, 45), rng.choice([1, 2]), cv2.LINE_AA)

    top_pad = 95 if add_caption else 45
    row_step = (H - top_pad - 35) // rows_n
    col_step = (W - 90) // cols_n
    labels_abs = []
    target_h = rng.randint(52, min(132, int(row_step * 0.58)))
    force_thin = rng.random() < 0.55

    for rr in range(rows_n):
        for cc in range(cols_n):
            if not is_4x5 and rng.random() < 0.12:
                continue  # blank cell seperti sample_2
            use_syl = bool(syl_pool) and (rng.random() < 0.30)
            if use_syl:
                patch, rel_b = rng.choice(syl_pool)
                ph0, pw0 = patch.shape[:2]
                sc = min(target_h * 1.25 / max(1, ph0), (col_step * 0.78) / max(1, pw0))
                nw, nh = max(12, int(round(pw0 * sc))), max(12, int(round(ph0 * sc)))
                patch_s = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
                rel_s = [(cid, bx0 * (nw / pw0), by0 * (nh / ph0), bx1 * (nw / pw0), by1 * (nh / ph0))
                         for cid, bx0, by0, bx1, by1 in rel_b]
            else:
                cname = rng.choice(ALL_NGLEGENA_NAMES)
                crop, _src = rng.choice(hw_pool[cname])
                ph0, pw0 = crop.shape[:2]
                sc = min(target_h / max(1, ph0), (col_step * 0.72) / max(1, pw0))
                nw, nh = max(12, int(round(pw0 * sc))), max(12, int(round(ph0 * sc)))
                patch_s = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                rel_s = [(CLASS_TO_ID[cname], 0.0, 0.0, float(nw), float(nh))]

            patch_s, rel_s = make_thin_or_thick_stroke(patch_s, rel_s, rng, force_thin=force_thin)
            cx = 55 + cc * col_step + col_step // 2 + rng.randint(-12, 12)
            cy = top_pad + rr * row_step + row_step // 2 + rng.randint(-8, 8)
            x0 = max(6, min(W - patch_s.shape[1] - 6, cx - patch_s.shape[1] // 2))
            y0 = max(top_pad - 20, min(H - patch_s.shape[0] - 6, cy - patch_s.shape[0] // 2))

            if add_headers:
                lbl_txt = ALL_NGLEGENA_NAMES[rel_s[0][0]] if rel_s[0][0] < 20 else rng.choice(ALL_NGLEGENA_NAMES)
                cv2.putText(canvas, lbl_txt, (max(8, x0 + patch_s.shape[1] // 3), max(22, y0 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.55, 0.75), (25, 25, 30), 2, cv2.LINE_AA)

            if paste_bw_patch(canvas, patch_s, ink_col, x0, y0):
                for cid, bx0, by0, bx1, by1 in rel_s:
                    labels_abs.append((cid, x0 + bx0, y0 + by0, x0 + bx1, y0 + by1))

    if rng.random() < 0.30:
        # Bingkai tepi scanner / garis dinding (seperti sample_1_paper / 05_test_walllines)
        thick = rng.randint(4, 18)
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (20, 20, 24), thick)

    return canvas, labels_abs


def gen_handwriting_line_or_sentence(hw_pool, syl_pool, rng):
    """Bangun 1-3 baris kalimat/frasa tulisan tangan nyata campuran nglegena & sandhangan."""
    W = rng.choice([760, 920, 1100])
    n_lines = rng.randint(1, 3)
    H = rng.choice([240, 320]) if n_lines == 1 else rng.choice([480, 620, 720])
    canvas, _ = create_paper_background(W, H, rng, enable_bg=True)
    ink_col = random_ink_color(rng)

    y_cur = rng.randint(35, 65)
    if rng.random() < 0.40:
        cv2.putText(canvas, rng.choice(["Wedang bubuk tanpa gula.", "Catatan Aksara Jawa Harian", "ha na ca ra ka da ta sa wa la"]),
                    (rng.randint(16, 40), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 50), 1, cv2.LINE_AA)
        y_cur += 32

    labels_abs = []
    base_h = rng.randint(36, 72)
    force_thin = rng.random() < 0.35

    for _ in range(n_lines):
        if y_cur + base_h + 24 >= H:
            break
        x_cur = rng.randint(24, 52)
        row_max_h = base_h
        for _ in range(rng.randint(6, 16)):
            use_syl = bool(syl_pool) and (rng.random() < 0.45)
            if use_syl:
                patch, rel_b = rng.choice(syl_pool)
                ph0, pw0 = patch.shape[:2]
                sc = base_h * 1.15 / max(1, ph0)
                nw, nh = max(10, int(round(pw0 * sc))), max(10, int(round(ph0 * sc)))
                patch_s = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
                rel_s = [(cid, bx0 * (nw / pw0), by0 * (nh / ph0), bx1 * (nw / pw0), by1 * (nh / ph0))
                         for cid, bx0, by0, bx1, by1 in rel_b]
            else:
                cname = rng.choice(ALL_NGLEGENA_NAMES)
                crop, _ = rng.choice(hw_pool[cname])
                ph0, pw0 = crop.shape[:2]
                sc = base_h / max(1, ph0)
                nw, nh = max(10, int(round(pw0 * sc))), max(10, int(round(ph0 * sc)))
                patch_s = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                rel_s = [(CLASS_TO_ID[cname], 0.0, 0.0, float(nw), float(nh))]

            patch_s, rel_s = make_thin_or_thick_stroke(patch_s, rel_s, rng, force_thin=force_thin)
            py = y_cur + rng.randint(-4, 4)
            if x_cur + patch_s.shape[1] >= W - 20 or py + patch_s.shape[0] >= H - 10:
                break
            if paste_bw_patch(canvas, patch_s, ink_col, x_cur, py):
                for cid, bx0, by0, bx1, by1 in rel_s:
                    labels_abs.append((cid, x_cur + bx0, py + by0, x_cur + bx1, py + by1))
            x_cur += patch_s.shape[1] + rng.randint(3, 14)
            row_max_h = max(row_max_h, patch_s.shape[0])
        y_cur += row_max_h + rng.randint(24, 46)

    return canvas, labels_abs


def gen_multifont_sentence_sample(full_fonts, nopas_fonts, corpus, weights, rng):
    """Render kalimat dari 11 font OFL/Apache-2.0 pada beragam resolusi & gaya kertas."""
    use_nopas = bool(nopas_fonts) and (rng.random() < 0.42)
    font_name, font_sz, pil_font = rng.choice(nopas_fonts if use_nopas else full_fonts)

    is_strip = rng.random() < 0.50
    if is_strip:
        W = rng.choice([720, 896, 1024])
        H = rng.choice([210, 270, 330])
        num_lines = 1
    else:
        W = rng.choice([640, 800, 960])
        H = rng.choice([520, 640, 740])
        num_lines = rng.randint(2, 4)

    canvas, _ = create_paper_background(W, H, rng, enable_bg=True)
    ink_col = random_ink_color(rng)
    y_cur = rng.randint(24, 48)
    if rng.random() < 0.35:
        cap = rng.choice(["Wedang bubuk tanpa gula, aja ngamuk mundhak tuwa.", "Latihan Transliterasi Aksara Jawa", "ha na ca ra ka da ta sa wa la"])
        cv2.putText(canvas, cap, (rng.randint(14, 40), y_cur + 10), cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.45, 0.72), tuple(int(c) for c in ink_col), rng.choice([1, 2]), cv2.LINE_AA)
        y_cur += rng.randint(34, 52)

    labels_abs = []
    for _ in range(num_lines):
        if y_cur > H - 68:
            break
        for _try in range(12):
            s_item = rng.choices(corpus, weights=weights, k=1)[0]
            uni_s = s_item["unicode"]
            if use_nopas and PANGKON in uni_s:
                continue
            ok, _, clusters = validate_javanese_unicode(uni_s, strict_55=True)
            if ok and clusters:
                break
        else:
            continue

        scale = rng.uniform(0.55, 1.25)
        x_cur = rng.randint(18, 42)
        max_row_h = 0
        for cl in clusters:
            patch, rel_boxes = render_cluster_cached(cl, font_name, font_sz, pil_font)
            if patch is None or not rel_boxes:
                continue
            ph0, pw0 = patch.shape
            nw, nh = max(6, int(round(pw0 * scale))), max(6, int(round(ph0 * scale)))
            patch_s = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
            rel_s = [(cid, bx0 * (nw / pw0), by0 * (nh / ph0), bx1 * (nw / pw0), by1 * (nh / ph0))
                     for cid, bx0, by0, bx1, by1 in rel_boxes]
            ph, pw = patch_s.shape
            py = max(4, y_cur + rng.randint(-2, 2))
            if x_cur + pw >= W - 16 or py + ph >= H - 8:
                break
            paste_bw_patch(canvas, patch_s, ink_col, x_cur, py)
            for cid, bx0, by0, bx1, by1 in rel_s:
                labels_abs.append((cid, x_cur + bx0, py + by0, x_cur + bx1, py + by1))
            x_cur += pw + rng.randint(1, 7)
            max_row_h = max(max_row_h, ph)
        y_cur += max(52, max_row_h + rng.randint(16, 30))

    return canvas, labels_abs


def gen_hard_negative_latin_sample(rng):
    """Gambar murni distraktor Latin / kertas bergaris tanpa Aksara Jawa (0 kotak)."""
    W = rng.choice([640, 800, 960])
    H = rng.choice([320, 480, 640])
    canvas, _ = create_paper_background(W, H, rng, enable_bg=True)
    ink = random_ink_color(rng)
    lines = [
        "ha na ca ra ka da ta sa wa la",
        "pa dha ja ya nya ma ga ba tha nga",
        "Wedang bubuk tanpa gula, aja ngamuk mundhak tuwa.",
        "Gambar 2. Sampel nyata 20 aksara Jawa Nglegena",
        "AksaraScan - Pemindai Aksara Jawa Digital",
        "Pemuatan gambar dari file / kamera HP",
        "abcdefghijklmnopqrstuvwxyz 0123456789",
    ]
    y = rng.randint(35, 70)
    while y < H - 30:
        txt = rng.choice(lines)
        cv2.putText(canvas, txt, (rng.randint(16, 80), y), rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]),
                    rng.uniform(0.45, 0.95), tuple(int(c) for c in ink), rng.choice([1, 2]), cv2.LINE_AA)
        y += rng.randint(36, 75)
    if rng.random() < 0.4:
        cv2.rectangle(canvas, (2, 2), (W - 3, H - 3), (25, 25, 30), rng.randint(2, 12))
    canvas, _ = apply_strong_blur_and_optics(canvas, [], rng)
    return canvas, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=2600)
    parser.add_argument("--n-val", type=int, default=320)
    parser.add_argument("--out-dir", default="dataset/external/synth_v3")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT_DIR, args.out_dir)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)

    full_fonts, nopas_fonts = discover_expanded_fonts()
    hw_pools = load_all_handwriting_pools()
    syl_pools = load_syllable_pools()

    corpus_path = os.path.join(EXT_DIR, "corpus_v2", "sentences_validated.jsonl")
    corpus = [json.loads(l) for l in open(corpus_path, "r", encoding="utf-8") if l.strip()]
    weights = select_weighted_sentences(corpus)

    print(
        f"[Synth v3] full_fonts={len(full_fonts)} | nopas_fonts={len(nopas_fonts)} | "
        f"hw_train={sum(len(v) for v in hw_pools['train'].values())} | "
        f"syl_train={len(syl_pools['train'])}"
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    preview_tiles = []

    for split, count in (("train", args.n_train), ("val", args.n_val)):
        hw_p = hw_pools[split]
        syl_p = syl_pools[split]
        saved = 0
        for i in range(count):
            r_kind = rng.random()
            if r_kind < 0.08:
                canvas, labels_abs = gen_hard_negative_latin_sample(rng)
                is_neg = True
            elif r_kind < 0.28:
                canvas, labels_abs = gen_handwriting_grid_sample(hw_p, syl_p, rng)
                is_neg = False
            elif r_kind < 0.50:
                canvas, labels_abs = gen_handwriting_line_or_sentence(hw_p, syl_p, rng)
                is_neg = False
            else:
                canvas, labels_abs = gen_multifont_sentence_sample(full_fonts, nopas_fonts, corpus, weights, rng)
                is_neg = False

            if not is_neg:
                canvas, labels_abs = apply_strong_blur_and_optics(canvas, labels_abs, rng)
                if not labels_abs:
                    continue

            H, W = canvas.shape[:2]
            yolo_boxes = []
            for cid, x0, y0, x1, y1 in labels_abs:
                xc, yc = ((x0 + x1) / 2.0) / W, ((y0 + y1) / 2.0) / H
                bw, bh = (x1 - x0) / W, (y1 - y0) / H
                if bw * W < 3.0 or bh * H < 3.0:
                    continue
                ok, _ = validate_yolo_box(cid, xc, yc, bw, bh)
                if ok:
                    yolo_boxes.append((cid, xc, yc, bw, bh))

            if not is_neg and not yolo_boxes:
                continue

            stem = f"synth3_{split}_{i:05d}"
            ip = os.path.join(out_dir, "images", split, stem + ".jpg")
            lp = os.path.join(out_dir, "labels", split, stem + ".txt")
            cv2.imwrite(ip, canvas, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(46, 95)])
            with open(lp, "w", encoding="utf-8") as lf:
                for cid, xc, yc, bw, bh in yolo_boxes:
                    lf.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            saved += 1

            if split == "val" and len(preview_tiles) < 16 and i % 15 == 0:
                vis = canvas.copy()
                for cid, xc, yc, bw, bh in yolo_boxes:
                    vx0, vy0 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
                    vx1, vy1 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
                    cv2.rectangle(vis, (vx0, vy0), (vx1, vy1), (34, 197, 94), 2)
                preview_tiles.append(cv2.resize(vis, (240, 160)))

            if (i + 1) % 500 == 0 or (i + 1) == count:
                print(f"  [{split}] {i + 1}/{count} (saved={saved})")

    if len(preview_tiles) >= 12:
        grid = np.vstack([np.hstack(preview_tiles[k:k+4]) for k in range(0, 12, 4)])
        cv2.imwrite("C:/Users/WA/.gemini/antigravity/brain/e8f20c41-7ca5-46b2-92f2-7e5b3331f29f/scratch/synth_v3_audit.png", grid)
    write_yolo_data_yaml(out_dir)
    print(f"[SELESAI] Synth v3 tersimpan ke {out_dir}")


if __name__ == "__main__":
    main()
