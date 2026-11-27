"""
Generator Sintetis & Komposit Tahan Lengkungan, Blur, Multi-Tinta, Latar Garis/Tembok,
serta Pertahanan Teks Latin/Non-Jawa — AksaraScan v4.0 (`tools/generate_v4_elastic_hw.py`).

Memanfaatkan HANYA split sumber v4 bebas-kebocoran (`manifests_v4/{hanacaraka_v4,jawa_zip_v4,jawa_syllables_v4}.jsonl`),
12 font Aksara Jawa OFL/Apache-2.0, serta 3 font Latin SIL OFL 1.1.
JUGA mengekspor bank patch `non_aksara_distractors_v4` untuk pelatihan Stage-2 Reject Prototype di Fase 2.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "synth_v2"))

from schema_v4 import CLASSES_55, CLASS_TO_ID, compute_dhash64, compute_phash64, qa_validate_sample
from v4_common import REPORT_DIR, build_contact_sheet, save_v4_manifest, write_yolo_label_file
from orthography_rules import PANGKON, validate_javanese_unicode
from generate_v2 import BASE_CH_TO_NAME, create_paper_background, render_cluster_cached, select_weighted_sentences
from generate_v3_blur_hw import FULL_OT_FONTS, discover_expanded_fonts

MANIFEST_DIR = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4")
OUT_SYNTH = os.path.join(ROOT_DIR, "dataset", "external", "synth_v4")
OUT_DISTRACTORS = os.path.join(ROOT_DIR, "dataset", "external", "non_aksara_distractors_v4")
LATIN_FONT_DIR = os.path.join(ROOT_DIR, "dataset", "external", "fonts_latin_ofl")

ALL_NGLEGENA = [n for n in CLASSES_55[:20]]

LATIN_DISTRACTOR_PHRASES = [
    "Aksara Jawa", "Ha Na Ca Ra Ka", "Da Ta Sa Wa La", "Pa Dha Ja Ya Nya", "Ma Ga Ba Tha Nga",
    "Bab 3: Carakan", "Latihan Menulis", "Catatan Harian", "Pasangan & Sandhangan",
    "Halaman 14", "Fig. 2B Tulisan", "Nomor 01-20", "Wédang bubuk", "Semarang, 2026",
    "Kelompok 4 UM", "aksara krama ngoko", "Tabel Huruf Vokal", "Suku Pepet Taling",
]

# Kalimat khusus berkelas lengkap untuk menjamin seluruh 55 kelas (pasangan langka + keret + pengkal + pada)
RARE_COMPLETE_LINES = [
    "꧋ꦏꦽꦠꦼꦒ꧀ꦥꦿꦱꦗ꧈ꦱꦿꦺꦁꦔꦺꦔꦺꦔ꧀ꦪꦸꦥꦿꦪꦂ꧉",
    "꧋ꦩꦤ꧀ꦝꦾꦔ꧀ꦛꦔ꧀ꦚꦩ꧀ꦥꦸꦂꦏ꧀ꦫꦩ꧈ꦲꦤ꧀ꦠꦾꦧꦱ꧉",
    "꧋ꦱꦥ꧀ꦫꦺꦴꦔ꧀ꦪꦃꦔ꧀ꦛꦺꦴꦏ꧀ꦒ꧀ꦪꦤ꧀꧈ꦥ꦳ꦶꦠ꧀ꦔ꧀ꦝꦏ꧀꧉",
    "꧋ꦧꦽꦩ꧀ꦱꦽꦔꦺꦔꦺ꧈ꦏꦾꦲꦶꦏꦿꦩꦏꦼꦛꦸꦏ꧀꧉",
    "꧋ꦲꦏ꧀ꦪꦱꦤ꧀ꦚ꧀ꦕꦔ꧀ꦒꦸꦤꦸꦁ꧈ꦠ꧀ꦫꦸꦱ꧀ꦏ꧀ꦔꦮꦶ꧉",
    "꧋ꦱꦠ꧀ꦪꦏ꧀ꦫꦩꦤ꧀ꦚꦱꦽꦠ꧀꧈ꦏꦧꦺꦃꦔ꧀ꦒꦮꦺ꧉",
]


def discover_latin_fonts():
    fonts = []
    if not os.path.isdir(LATIN_FONT_DIR):
        return fonts
    for fn in sorted(os.listdir(LATIN_FONT_DIR)):
        if not fn.lower().endswith((".ttf", ".otf")):
            continue
        fp = os.path.join(LATIN_FONT_DIR, fn)
        kind = "handwritten" if "caveat" in fn.lower() else ("serif" if "newsreader" in fn.lower() else "sans")
        for sz in (18, 24, 30, 38):
            try:
                fonts.append((fn, kind, sz, ImageFont.truetype(fp, sz)))
            except Exception:
                pass
    return fonts


def read_gray_alpha_white(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        return im
    if im.shape[2] == 4:
        alpha = im[:, :, 3:4].astype(np.float32) / 255.0
        rgb_gray = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)[:, :, None]
        comp = rgb_gray * alpha + 255.0 * (1.0 - alpha)
        return np.clip(comp[:, :, 0], 0, 255).astype(np.uint8)
    return cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)


def load_v4_glyph_pools():
    """Muat glyph tulisan tangan TERPISAH STRICT per split ('train' vs 'val' saja; JANGAN muat test_heldout!)."""
    pools = {"train": {n: [] for n in ALL_NGLEGENA}, "val": {n: [] for n in ALL_NGLEGENA}}
    for mname in ("hanacaraka_v4.jsonl", "jawa_zip_v4.jsonl"):
        mp = os.path.join(MANIFEST_DIR, mname)
        if not os.path.exists(mp):
            continue
        for ln in open(mp, "r", encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            sp = r.get("split")
            if sp not in ("train", "val"):
                continue
            cls_name = r["boxes"][0]["class"]
            if cls_name not in pools[sp]:
                continue
            gray = read_gray_alpha_white(os.path.join(ROOT_DIR, r["image"]))
            if gray is None:
                continue
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ys, xs = np.where(bw > 0)
            if len(ys) < 16:
                continue
            pools[sp][cls_name].append((bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1], r["source"]))

    syl_pools = {"train": [], "val": []}
    syl_mp = os.path.join(MANIFEST_DIR, "jawa_syllables_v4.jsonl")
    if os.path.exists(syl_mp):
        for ln in open(syl_mp, "r", encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            sp = r.get("split")
            if sp not in ("train", "val"):
                continue
            gray = read_gray_alpha_white(os.path.join(ROOT_DIR, r["image"]))
            if gray is None:
                continue
            h, w = gray.shape[:2]
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            boxes = []
            for b in r["boxes"]:
                xc, yc, bwn, bhn = b["bbox"]
                boxes.append((b["cid"], (xc - bwn / 2) * w, (yc - bhn / 2) * h, (xc + bwn / 2) * w, (yc + bhn / 2) * h))
            x0 = max(0, int(min(x[1] for x in boxes)) - 2)
            y0 = max(0, int(min(x[2] for x in boxes)) - 2)
            x1 = min(w, int(max(x[3] for x in boxes)) + 2)
            y1 = min(h, int(max(x[4] for x in boxes)) + 2)
            patch = bw[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            rel = [(cid, bx0 - x0, by0 - y0, bx1 - x0, by1 - y0) for cid, bx0, by0, bx1, by1 in boxes]
            syl_pools[sp].append((patch, rel))
    return pools, syl_pools


def tps_warp_patch(patch, rel_boxes, rng, amp=0.14):
    """
    Thin-Plate Spline (TPS) Elastic Warp pada potongan glif/kluster:
    Mengubah lengkungan goresan secara realistis (simulasi beda gaya penulis)
    sambil menyelaraskan koordinat kotak ke tepi tinta hasil warp.
    """
    H, W = patch.shape[:2]
    if H < 14 or W < 14 or (patch > 0).sum() < 15:
        return patch, rel_boxes

    pad = max(4, int(max(H, W) * 0.16))
    padded = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    pH, pW = padded.shape[:2]

    # Grid kontrol 3x3 di ruang [0..pW-1, 0..pH-1]; titik pusat & tepi digeser elastis secara halus
    gx = np.linspace(0, pW - 1, 3)
    gy = np.linspace(0, pH - 1, 3)
    src_pts = []
    dst_pts = []
    dx_scale = W * amp
    dy_scale = H * amp
    for iy, yy in enumerate(gy):
        for ix, xx in enumerate(gx):
            src_pts.append([xx, yy])
            if ix == 1 and iy == 1:
                dst_pts.append([xx + rng.uniform(-dx_scale, dx_scale), yy + rng.uniform(-dy_scale, dy_scale)])
            else:
                dst_pts.append([xx + rng.uniform(-dx_scale * 0.38, dx_scale * 0.38), yy + rng.uniform(-dy_scale * 0.38, dy_scale * 0.38)])

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    K_n = len(src)

    def _u(r2):
        return r2 * np.log(r2 + 1e-6)

    diff = dst[:, None, :] - dst[None, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    K = _u(r2)
    P = np.hstack([np.ones((K_n, 1)), dst])
    L = np.zeros((K_n + 3, K_n + 3), dtype=np.float64)
    L[:K_n, :K_n] = K + np.eye(K_n) * 1e-2
    L[:K_n, K_n:] = P
    L[K_n:, :K_n] = P.T
    Y = np.vstack([src, np.zeros((3, 2), dtype=np.float64)])
    coeffs = np.linalg.solve(L, Y)
    W_x, a_x = coeffs[:K_n, 0], coeffs[K_n:, 0]
    W_y, a_y = coeffs[:K_n, 1], coeffs[K_n:, 1]

    yy_grid, xx_grid = np.mgrid[0:pH, 0:pW].astype(np.float64)
    qx = xx_grid.ravel()
    qy = yy_grid.ravel()
    dx = qx[:, None] - dst[None, :, 0]
    dy = qy[:, None] - dst[None, :, 1]
    u_q = _u(dx * dx + dy * dy)
    map_x = (a_x[0] + a_x[1] * qx + a_x[2] * qy + u_q @ W_x).reshape(pH, pW).astype(np.float32)
    map_y = (a_y[0] + a_y[1] * qx + a_y[2] * qy + u_q @ W_y).reshape(pH, pW).astype(np.float32)

    warped = cv2.remap(padded, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Remap box boundaries via inverse nearest displacement & ketatkan ke tinta bila mono-box
    ys, xs = np.where(warped > 32)
    if len(ys) < 12:
        return patch, rel_boxes
    nx0_all, ny0_all, nx1_all, ny1_all = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    cropped = warped[ny0_all:ny1_all, nx0_all:nx1_all]

    if len(rel_boxes) == 1:
        cid = rel_boxes[0][0]
        cH, cW = cropped.shape[:2]
        return cropped, [(cid, 0.0, 0.0, float(cW), float(cH))]

    # Multi-box (syllable/cluster): perbarui tiap kotak dengan padding aman + clamp ke ukuran baru
    cH, cW = cropped.shape[:2]
    shift_x = pad - nx0_all
    shift_y = pad - ny0_all
    new_rel = []
    margin = max(2.0, max(dx_scale, dy_scale) * 0.45)
    for cid, bx0, by0, bx1, by1 in rel_boxes:
        nbx0 = max(0.0, min(float(cW - 2), bx0 + shift_x - margin))
        nby0 = max(0.0, min(float(cH - 2), by0 + shift_y - margin))
        nbx1 = min(float(cW), max(nbx0 + 3.0, bx1 + shift_x + margin))
        nby1 = min(float(cH), max(nby0 + 3.0, by1 + shift_y + margin))
        new_rel.append((cid, nbx0, nby0, nbx1, nby1))
    return cropped, new_rel


def modulate_stroke_and_shear(patch, rel_boxes, rng, shear_max=4.5):
    """Modulasi ketebalan kuas/pulpen (1-2px erosi/dilasi) + shear gaya tulis."""
    aug = patch.copy()
    r = rng.random()
    if r < 0.34:
        k = rng.choice([2, 3])
        er = cv2.erode(aug, np.ones((k, k), np.uint8), iterations=1)
        if (er > 32).sum() >= max(10, int((aug > 32).sum() * 0.25)):
            aug = er
    elif r > 0.68:
        k = rng.choice([(2, 2), (2, 3), (3, 2)])
        aug = cv2.dilate(aug, np.ones((k[1], k[0]), np.uint8), iterations=1)

    if rng.random() < 0.55 and aug.shape[0] >= 12 and aug.shape[1] >= 12:
        sh = math.tan(math.radians(rng.uniform(-shear_max, shear_max)))
        h, w = aug.shape[:2]
        extra = int(abs(sh) * h) + 2
        M = np.array([[1.0, sh, max(0.0, -sh * h)], [0.0, 1.0, 0.0]], dtype=np.float32)
        nw = w + extra
        sh_img = cv2.warpAffine(aug, M, (nw, h), flags=cv2.INTER_LINEAR, borderValue=0)
        ys, xs = np.where(sh_img > 32)
        if len(ys) >= 10 and len(rel_boxes) == 1:
            aug = sh_img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            cH, cW = aug.shape[:2]
            rel_boxes = [(rel_boxes[0][0], 0.0, 0.0, float(cW), float(cH))]
    return aug, rel_boxes


def sample_ink_color(rng, is_dark_bg=False):
    """Tinta realistis multi-warna (hitam, pensil, pulpen biru, merah, hijau tua, emas gelap)."""
    if is_dark_bg:
        return rng.choice([
            np.array([235, 224, 200], dtype=np.float32),
            np.array([92, 188, 226], dtype=np.float32),
            np.array([245, 245, 245], dtype=np.float32),
        ])
    palette = rng.choices(
        ["black", "blue_ballpoint", "red_pen", "green_pen", "faded_pencil", "sepia"],
        weights=[0.32, 0.24, 0.14, 0.10, 0.12, 0.08],
        k=1,
    )[0]
    if palette == "black":
        v = rng.randint(10, 48)
        return np.array([v, v, v], dtype=np.float32)
    if palette == "blue_ballpoint":
        return np.array([rng.randint(145, 210), rng.randint(40, 88), rng.randint(14, 48)], dtype=np.float32)
    if palette == "red_pen":
        return np.array([rng.randint(25, 60), rng.randint(24, 58), rng.randint(155, 212)], dtype=np.float32)
    if palette == "green_pen":
        return np.array([rng.randint(38, 78), rng.randint(98, 152), rng.randint(22, 62)], dtype=np.float32)
    if palette == "faded_pencil":
        v = rng.randint(92, 146)
        return np.array([v, v + 2, v + 4], dtype=np.float32)
    return np.array([rng.randint(36, 70), rng.randint(60, 96), rng.randint(96, 138)], dtype=np.float32)


def make_complex_surface_bg(W, H, rng, style_force=None):
    """Buat latar dunia nyata: kertas dluwang, buku bergaris, grid, retakan/garis tembok, lipatan, dark mode."""
    style = style_force or rng.choices(
        ["aged_paper", "ruled_notebook", "grid_paper", "wall_lines", "shadow_crease", "dark_mode"],
        weights=[0.20, 0.22, 0.14, 0.22, 0.14, 0.08],
        k=1,
    )[0]

    if style == "dark_mode":
        base = np.full((H, W, 3), [rng.randint(28, 46), rng.randint(18, 30), rng.randint(12, 22)], dtype=np.uint8)
        return base, style, True

    bg, _ = create_paper_background(W, H, rng)
    if style == "ruled_notebook":
        step = rng.randint(34, 52)
        col_line = rng.choice([(210, 178, 152), (175, 165, 155)])
        for yy in range(step, H, step):
            cv2.line(bg, (0, yy), (W, yy), col_line, rng.choice([1, 1, 2]))
        if W > 320 and rng.random() < 0.7:
            mx = rng.randint(48, 90)
            cv2.line(bg, (mx, 0), (mx, H), (150, 145, 225), 2)
    elif style == "grid_paper":
        step = rng.randint(26, 40)
        for yy in range(step, H, step):
            cv2.line(bg, (0, yy), (W, yy), (212, 206, 198), 1)
        for xx in range(step, W, step):
            cv2.line(bg, (xx, 0), (xx, H), (212, 206, 198), 1)
    elif style == "wall_lines":
        # Garis tembok / plester / bingkai kamera tebal-tipis memotong bidang gambar
        n_lines = rng.randint(4, 10)
        for _ in range(n_lines):
            x0, y0 = rng.randint(-20, W + 20), rng.randint(-20, H + 20)
            x1, y1 = rng.randint(-20, W + 20), rng.randint(-20, H + 20)
            c = rng.randint(115, 185)
            cv2.line(bg, (x0, y0), (x1, y1), (c, c + rng.randint(-5, 6), c + rng.randint(-5, 6)), rng.choice([1, 2, 3]))
        # Tambah pinggiran tepi gelap bingkai scanner/meja di tepi
        if rng.random() < 0.7:
            bw = rng.randint(5, 18)
            side = rng.choice(["top", "left", "bottom", "right"])
            dc = rng.randint(45, 95)
            if side == "top": bg[:bw, :] = dc
            elif side == "bottom": bg[H - bw:, :] = dc
            elif side == "left": bg[:, :bw] = dc
            else: bg[:, W - bw:] = dc
    elif style == "shadow_crease":
        # Lipatan kertas diagonal + bayangan gradien tajam
        px0, py0 = rng.randint(0, W // 2), rng.randint(0, H)
        px1, py1 = rng.randint(W // 2, W), rng.randint(0, H)
        cv2.line(bg, (px0, py0), (px1, py1), (158, 152, 145), 2)
        xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        ang = rng.uniform(0, math.pi)
        proj = xx * math.cos(ang) + yy * math.sin(ang)
        mask = np.clip(0.55 + 0.45 * np.tanh((proj - 0.5) * rng.uniform(4.0, 9.0)), 0.42, 1.0)[..., None]
        bg = np.clip(bg.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    return bg, style, False


def apply_optical_degradations(canvas, labels_abs, rng, rot_max=8.0, force_blur=False):
    """PSF Defocus, Arbitrary-Angle Motion Blur, Down-Up LowRes, Sensor Noise, JPEG artifacts."""
    H, W = canvas.shape[:2]

    # 1. Pencahayaan non-uniform tambahan
    if rng.random() < 0.65:
        gx = np.linspace(rng.uniform(0.58, 1.0), rng.uniform(0.75, 1.05), W, dtype=np.float32)
        gy = np.linspace(rng.uniform(0.62, 1.0), rng.uniform(0.75, 1.05), H, dtype=np.float32)
        shad = np.outer(gy, gx)[..., None]
        canvas = np.clip(canvas.astype(np.float32) * shad, 0, 255).astype(np.uint8)

    # 2. Rotasi lebih lebar (+- rot_max) dengan pembaruan bbox aman
    if labels_abs and rng.random() < 0.55:
        ang = rng.uniform(-rot_max, rot_max)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
        canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        upd = []
        for cid, x0, y0, x1, y1 in labels_abs:
            pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float32)
            t = (M @ pts.T).T
            nx0, ny0 = max(0.0, float(t[:, 0].min())), max(0.0, float(t[:, 1].min()))
            nx1, ny1 = min(float(W), float(t[:, 0].max())), min(float(H), float(t[:, 1].max()))
            if nx1 - nx0 >= 3.0 and ny1 - ny0 >= 3.0:
                upd.append((cid, nx0, ny0, nx1, ny1))
        labels_abs = upd

    # 3. Blur PSF (Defocus / Motion arbitrary-angle / Down-Up Zoom)
    rb = 0.20 if force_blur else rng.random()
    if rb < 0.30:
        sig = rng.uniform(0.75, 2.15)
        k = int(max(3, min(11, 2 * round(sig * 2.0) + 1)))
        canvas = cv2.GaussianBlur(canvas, (k, k), sig)
    elif rb < 0.52:
        k_len = rng.choice([3, 5, 7, 9])
        ker = np.zeros((k_len, k_len), dtype=np.float32)
        ker[k_len // 2, :] = 1.0 / k_len
        ang_m = rng.uniform(0, 180)
        M_m = cv2.getRotationMatrix2D((k_len / 2.0 - 0.5, k_len / 2.0 - 0.5), ang_m, 1.0)
        ker = cv2.warpAffine(ker, M_m, (k_len, k_len))
        s = ker.sum()
        if s > 1e-6:
            ker /= s
            canvas = cv2.filter2D(canvas, -1, ker)
    elif rb < 0.64:
        sc = rng.uniform(0.42, 0.72)
        sm = cv2.resize(canvas, (max(32, int(W * sc)), max(32, int(H * sc))), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)

    # 4. Sensor noise + JPEG artifacts
    if rng.random() < 0.45:
        ns = np.random.normal(0, rng.uniform(2.0, 6.5), canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + ns, 0, 255).astype(np.uint8)
    if rng.random() < 0.55:
        q = rng.randint(42, 84)
        _, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
        canvas = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return canvas, labels_abs


def draw_latin_block(canvas, text, x, y, font_tuple, color_bgr):
    _, _, _, pil_font = font_tuple
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    c_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    draw.text((x, y), text, font=pil_font, fill=c_rgb)
    bb = draw.textbbox((x, y), text, font=pil_font)
    res_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    canvas[:] = res_bgr
    return bb


def paste_bw_patch(canvas, patch, ink_col, x0, y0):
    H, W = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + pw > W or y0 + ph > H:
        return False
    alpha = (patch.astype(np.float32) / 255.0)[..., None]
    roi = canvas[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
    canvas[y0:y0 + ph, x0:x0 + pw] = np.clip(roi * (1.0 - alpha) + ink_col * alpha, 0, 255).astype(np.uint8)
    return True


def make_handwritten_tps_sample(hw_pool, syl_pool, latin_fonts, rng, params):
    rows_n = rng.choice([2, 3, 4])
    cols_n = 5
    cell_w = rng.randint(115, 162)
    cell_h = rng.randint(102, 138)
    pad_top = rng.randint(44, 72)
    W = cols_n * cell_w + 80
    H = rows_n * cell_h + pad_top + 48
    bg, bg_style, is_dark = make_complex_surface_bg(W, H, rng)
    ink = sample_ink_color(rng, is_dark_bg=is_dark)

    labels = []
    distractor_crops = []

    # Campurkan label/header Latin di atas halaman atau di atas setiap baris (curing diag_01/07/08!)
    add_latin = latin_fonts and rng.random() < 0.52
    latin_col = ink * 0.9 if is_dark else sample_ink_color(rng, is_dark_bg=False)
    if add_latin:
        lf = rng.choice(latin_fonts)
        title = rng.choice(LATIN_DISTRACTOR_PHRASES)
        bb = draw_latin_block(bg, title, rng.randint(24, 90), rng.randint(8, 22), lf, latin_col)
        if bb and bb[2] - bb[0] > 16 and bb[3] - bb[1] > 10:
            distractor_crops.append(("latin_" + lf[1], bb))

    for rr in range(rows_n):
        cy_base = pad_top + rr * cell_h + cell_h // 2 + rng.randint(-6, 6)
        if add_latin and rng.random() < 0.35 and latin_fonts:
            lf_sm = rng.choice([f for f in latin_fonts if f[2] <= 24] or latin_fonts)
            cx_l = rng.randint(36, 90)
            y_cap = max(2, cy_base - cell_h // 2 - 2)
            bb = draw_latin_block(bg, rng.choice(["ha na ca", "latihan", "baris " + str(rr + 1)]), cx_l, y_cap, lf_sm, latin_col)
            if bb and bb[2] - bb[0] > 14:
                distractor_crops.append(("latin_" + lf_sm[1], bb))

        for cc in range(cols_n):
            cx_cell = 40 + cc * cell_w + cell_w // 2
            use_syl = syl_pool and rng.random() < 0.32
            if use_syl:
                raw_p, rel_b = rng.choice(syl_pool)
            else:
                cls_name = rng.choice(ALL_NGLEGENA)
                pool = hw_pool.get(cls_name, [])
                if not pool:
                    continue
                raw_p, _ = rng.choice(pool)
                rel_b = [(CLASS_TO_ID[cls_name], 0.0, 0.0, float(raw_p.shape[1]), float(raw_p.shape[0]))]

            target_h = rng.randint(46, 78)
            sc = target_h / max(12, raw_p.shape[0])
            nw = max(12, min(cell_w - 18, int(round(raw_p.shape[1] * sc))))
            nh = max(12, min(cell_h - 20, int(round(raw_p.shape[0] * sc))))
            resized = cv2.resize(raw_p, (nw, nh), interpolation=cv2.INTER_AREA)
            rel_scaled = [(c, b0 * nw / raw_p.shape[1], b1 * nh / raw_p.shape[0], b2 * nw / raw_p.shape[1], b3 * nh / raw_p.shape[0])
                          for c, b0, b1, b2, b3 in rel_b]

            warped, rel_w = tps_warp_patch(resized, rel_scaled, rng, amp=params["tps_amp"])
            styled, rel_s = modulate_stroke_and_shear(warped, rel_w, rng, shear_max=params["shear_max"])

            x0 = max(6, min(W - styled.shape[1] - 6, cx_cell - styled.shape[1] // 2 + rng.randint(-8, 8)))
            y0 = max(6, min(H - styled.shape[0] - 6, cy_base - styled.shape[0] // 2 + rng.randint(-6, 6)))
            if not paste_bw_patch(bg, styled, ink, x0, y0):
                continue
            for cid, bx0, by0, bx1, by1 in rel_s:
                labels.append((cid, x0 + bx0, y0 + by0, x0 + bx1, y0 + by1))

    out_im, out_lbl = apply_optical_degradations(bg, labels, rng, rot_max=params["rot_max"])
    return out_im, out_lbl, distractor_crops, f"tps_hw_{bg_style}", bg_style


def make_printed_mixed_sample(full_fonts, nopas_fonts, sentences, latin_fonts, rng, params):
    """Sintesis baris cetak multi-font OFL + TPS ringan + selipan header/caption Latin non-label."""
    is_full = rng.random() < 0.72 and bool(full_fonts)
    font_entry = rng.choice(full_fonts if is_full else (nopas_fonts or full_fonts))
    fname, fsize, pil_font = font_entry

    n_lines = rng.choice([1, 2, 3])
    line_clusters = []
    for _ in range(n_lines):
        if rng.random() < 0.22 and is_full:
            cand = rng.choice(RARE_COMPLETE_LINES)
        else:
            cand = rng.choice(sentences) if sentences else "ꦲꦤꦕꦫꦏ꧉"
        if not is_full:
            cand = "".join(c for c in cand if c != PANGKON)
        ok, _, clus = validate_javanese_unicode(cand, strict_55=True)
        if clus and len(clus) >= 2:
            line_clusters.append(clus[:rng.randint(5, 16)])
    if not line_clusters:
        line_clusters = [["ꦲ", "ꦤ", "ꦕ", "ꦫ", "ꦏ"]]

    W = rng.randint(680, 1080)
    top_margin = rng.randint(46, 76)
    line_step = int(fsize * rng.uniform(1.75, 2.25))
    latin_header = None
    if latin_fonts and rng.random() < 0.58:
        lf = rng.choice(latin_fonts)
        latin_txt = rng.choice(LATIN_DISTRACTOR_PHRASES)
        top_margin = max(top_margin, lf[2] + rng.randint(36, 54))
        latin_header = (lf, latin_txt)

    H = top_margin + len(line_clusters) * line_step + 48
    bg, bg_style, is_dark = make_complex_surface_bg(W, H, rng)
    ink = sample_ink_color(rng, is_dark_bg=is_dark)

    labels = []
    distractor_crops = []

    if latin_header is not None:
        lf, l_txt = latin_header
        cap_col = ink * 0.9 if is_dark else sample_ink_color(rng, is_dark_bg=False)
        bb = draw_latin_block(bg, l_txt, rng.randint(24, 100), rng.randint(6, 14), lf, cap_col)
        if bb and bb[2] - bb[0] > 18:
            distractor_crops.append(("latin_" + lf[1], bb))

    for li, clus_row in enumerate(line_clusters):
        cur_x = rng.randint(28, 64)
        cy = top_margin + li * line_step
        for cluster in clus_row:
            if cur_x >= W - 80:
                break
            alpha_mask, rel_b = render_cluster_cached(cluster, fname, fsize, pil_font)
            if alpha_mask is None or not rel_b:
                cur_x += int(fsize * 0.42)
                continue
            adv = alpha_mask.shape[1]
            if rng.random() < 0.65:
                warped, rel_w = tps_warp_patch(alpha_mask, rel_b, rng, amp=params["tps_amp"] * 0.55)
            else:
                warped, rel_w = alpha_mask, rel_b

            px0 = cur_x
            py0 = max(6, cy - warped.shape[0] // 2 + rng.randint(-4, 4))
            if px0 + warped.shape[1] >= W - 12:
                break
            if paste_bw_patch(bg, warped, ink, px0, py0):
                for cid, bx0, by0, bx1, by1 in rel_w:
                    labels.append((cid, px0 + bx0, py0 + by0, px0 + bx1, py0 + by1))
            cur_x += max(adv + rng.randint(-1, 3), warped.shape[1] + 2)

    out_im, out_lbl = apply_optical_degradations(bg, labels, rng, rot_max=params["rot_max"])
    return out_im, out_lbl, distractor_crops, f"printed_mix_{bg_style}", bg_style


def make_pure_hard_negative(latin_fonts, rng):
    """Halaman negatif keras murni (0 kotak Aksara Jawa): teks Latin, angka, garis tembok/buku."""
    W = rng.randint(560, 880)
    H = rng.randint(240, 520)
    bg, bg_style, is_dark = make_complex_surface_bg(W, H, rng)
    ink = sample_ink_color(rng, is_dark_bg=is_dark)
    distractor_crops = []

    if latin_fonts:
        n_lines = rng.randint(2, 5)
        for li in range(n_lines):
            lf = rng.choice(latin_fonts)
            txt = rng.choice(LATIN_DISTRACTOR_PHRASES) + " · " + rng.choice(["12345", "abcdefg", "mnpqrst", "OCR Test"])
            y = 18 + li * rng.randint(46, 74)
            if y + 36 >= H:
                break
            bb = draw_latin_block(bg, txt[:34], rng.randint(18, 60), y, lf, ink)
            if bb and bb[2] - bb[0] > 14 and bb[3] - bb[1] > 10:
                kind = "digits_latin" if rng.random() < 0.25 else ("latin_" + lf[1])
                distractor_crops.append((kind, bb))

    # Ambil juga patch garis/lipatan tembok tanpa teks
    if bg_style in ("wall_lines", "ruled_notebook", "shadow_crease") and W > 128 and H > 128:
        rx = rng.randint(0, W - 96)
        ry = rng.randint(0, H - 96)
        tag = "ruled_or_wall_lines" if bg_style in ("wall_lines", "ruled_notebook") else "paper_stain_or_crease"
        distractor_crops.append((tag, (rx, ry, rx + 88, ry + 88)))

    out_im, _ = apply_optical_degradations(bg, [], rng, rot_max=5.0)
    return out_im, [], distractor_crops, f"neg_{bg_style}", bg_style


def run_generator(n_train=2400, n_val=360, seed=108, tps_amp=0.14, rot_max=8.0, shear_max=4.5):
    if os.path.exists(OUT_SYNTH):
        shutil.rmtree(OUT_SYNTH)
    if os.path.exists(OUT_DISTRACTORS):
        shutil.rmtree(OUT_DISTRACTORS)

    for sp in ("train", "val"):
        os.makedirs(os.path.join(OUT_SYNTH, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_SYNTH, "labels", sp), exist_ok=True)
        os.makedirs(os.path.join(OUT_DISTRACTORS, "images", sp), exist_ok=True)

    full_fonts, nopas_fonts = discover_expanded_fonts()
    latin_fonts = discover_latin_fonts()
    hw_pools, syl_pools = load_v4_glyph_pools()
    np.random.seed(seed)
    rng = random.Random(seed)
    corpus_path = os.path.join(ROOT_DIR, "dataset", "external", "corpus_v2", "sentences_validated.jsonl")
    corpus_rows = []
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as cf:
            for idx_c, ln in enumerate(cf):
                if not ln.strip():
                    continue
                corpus_rows.append(json.loads(ln))
                if idx_c >= 14000:
                    break
    if corpus_rows:
        wts = select_weighted_sentences(corpus_rows)
        sentences = [r["unicode"] for r in rng.choices(corpus_rows, weights=wts, k=6000) if r.get("unicode")]
    else:
        sentences = ["꧋ꦮꦺꦢꦁꦧꦸꦧꦸꦏ꧀ꦠꦤ꧀ꦥꦒꦸꦭ꧈ꦲꦗꦔꦩꦸꦏ꧀ꦩꦸꦤ꧀ꦝꦏ꧀ꦠꦸꦮ꧉", "ꦲꦏ꧀ꦱꦫꦗꦮ\nꦩꦔꦤ꧀ꦱꦼꦒ"]

    params = {"tps_amp": tps_amp, "rot_max": rot_max, "shear_max": shear_max}
    manifest = []
    distractor_manifest = []
    bg_counter = Counter()
    mode_counter = Counter()
    class_box_counter = Counter()

    targets = [("train", n_train), ("val", n_val)]
    for sp, count in targets:
        for i in range(count):
            roll = rng.random()
            if roll < 0.48:
                img, abs_boxes, d_crops, mtag, bg_st = make_handwritten_tps_sample(
                    hw_pools[sp], syl_pools[sp], latin_fonts, rng, params
                )
            elif roll < 0.85:
                img, abs_boxes, d_crops, mtag, bg_st = make_printed_mixed_sample(
                    full_fonts, nopas_fonts, sentences, latin_fonts, rng, params
                )
            else:
                img, abs_boxes, d_crops, mtag, bg_st = make_pure_hard_negative(latin_fonts, rng)

            H, W = img.shape[:2]
            boxes = []
            for cid, x0, y0, x1, y1 in abs_boxes:
                xc = max(0.001, min(0.999, ((x0 + x1) / 2.0) / W))
                yc = max(0.001, min(0.999, ((y0 + y1) / 2.0) / H))
                bw = max(0.004, min(0.99, (x1 - x0) / W))
                bh = max(0.004, min(0.99, (y1 - y0) / H))
                boxes.append({"class": CLASSES_55[cid], "cid": cid, "bbox": [xc, yc, bw, bh]})

            if boxes:
                ok, _ = qa_validate_sample(img.shape, boxes, min_box_px=3, max_aspect=14.0)
                if not ok:
                    continue

            stem = f"sv4_{sp}_{i:05d}"
            ip = os.path.join(OUT_SYNTH, "images", sp, stem + ".jpg")
            lp = os.path.join(OUT_SYNTH, "labels", sp, stem + ".txt")
            cv2.imwrite(ip, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            write_yolo_label_file(lp, boxes)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            manifest.append({
                "image_id": stem,
                "source": "synth_v4_elastic_ofl",
                "mode": mtag,
                "bg_style": bg_st,
                "license": "SIL OFL 1.1 / CC0 / CC BY 4.0",
                "writer_id": f"sv4_{sp}_{stem}",
                "cluster_id": f"sv4_{sp}_{stem}",
                "split": sp,
                "phash": compute_phash64(gray),
                "dhash": compute_dhash64(gray),
                "image": os.path.relpath(ip, ROOT_DIR).replace("\\", "/"),
                "label": os.path.relpath(lp, ROOT_DIR).replace("\\", "/"),
                "boxes": boxes,
                "label_preview": f"{mtag[:14]} ({len(boxes)}b)",
            })
            bg_counter[bg_st] += 1
            mode_counter[mtag.split("_")[0]] += 1
            for b in boxes:
                class_box_counter[b["class"]] += 1

            # Ekspor hingga 2 non-aksara patch per gambar untuk Stage-2 training
            for d_idx, (dkind, (dx0, dy0, dx1, dy1)) in enumerate(d_crops[:2]):
                cx0, cy0 = max(0, int(dx0) - 2), max(0, int(dy0) - 2)
                cx1, cy1 = min(W, int(dx1) + 2), min(H, int(dy1) + 2)
                if cx1 - cx0 < 12 or cy1 - cy0 < 12:
                    continue
                patch_bgr = img[cy0:cy1, cx0:cx1]
                d_stem = f"neg_{sp}_{i:05d}_{d_idx}_{dkind}"
                d_ip = os.path.join(OUT_DISTRACTORS, "images", sp, d_stem + ".jpg")
                cv2.imwrite(d_ip, patch_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                distractor_manifest.append({
                    "image_id": d_stem,
                    "source": "synth_v4_non_aksara",
                    "distractor_class": dkind,
                    "split": sp,
                    "image": os.path.relpath(d_ip, ROOT_DIR).replace("\\", "/"),
                })

    mpath = save_v4_manifest("synth_v4", manifest)
    dpath = save_v4_manifest("non_aksara_distractors_v4", distractor_manifest)

    # 3 Contact sheet khusus untuk inspeksi visual Gerbang 1
    tps_items = [r for r in manifest if r["mode"].startswith("tps_hw") and r["boxes"]]
    wall_items = [r for r in manifest if r["bg_style"] in ("wall_lines", "ruled_notebook", "dark_mode") and r["boxes"]]
    neg_items = [r for r in manifest if r["mode"].startswith("neg_") or r["mode"].startswith("printed_mix")]
    build_contact_sheet(tps_items, "Synth v4 TPS Elastic Handwriting", os.path.join(REPORT_DIR, "v4_contact_synth_tps.jpg"), max_n=25, cols=5, cell_w=240, cell_h=155)
    build_contact_sheet(wall_items, "Synth v4 Multi-Ink & Wall/Ruled Bg", os.path.join(REPORT_DIR, "v4_contact_synth_multiink_wall.jpg"), max_n=25, cols=5, cell_w=240, cell_h=155)
    build_contact_sheet(neg_items, "Synth v4 Latin & Hard Negatives", os.path.join(REPORT_DIR, "v4_contact_synth_hardneg_latin.jpg"), max_n=25, cols=5, cell_w=240, cell_h=155)

    summary = {
        "manifest": os.path.relpath(mpath, ROOT_DIR),
        "distractors_manifest": os.path.relpath(dpath, ROOT_DIR),
        "total_images": len(manifest),
        "total_boxes": sum(class_box_counter.values()),
        "distinct_classes": len(class_box_counter),
        "distractor_patches": len(distractor_manifest),
        "bg_styles": dict(bg_counter),
        "modes": dict(mode_counter),
        "params": {"seed": seed, **params},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2400)
    ap.add_argument("--val", type=int, default=360)
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--tps-amp", type=float, default=0.14)
    ap.add_argument("--rot-max", type=float, default=8.0)
    ap.add_argument("--shear-max", type=float, default=4.5)
    args = ap.parse_args()
    run_generator(n_train=args.train, n_val=args.val, seed=args.seed, tps_amp=args.tps_amp, rot_max=args.rot_max, shear_max=args.shear_max)
