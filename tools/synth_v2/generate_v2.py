"""
AksaraScan v3.0 — Generator Data Sintetis v2 dengan Domain Randomization Penuh (Fase 2)
Fitur:
  - Font HANYA berlisensi SIL OFL 1.1 (`NotoSansJavanese.ttf`, `TuladhaJejegOT-Regular.ttf/.otf`), TANPA `javatext.ttf` proprieter.
  - Korpus kalimat Jawa tervalidasi dari NusaX, NusaWrites, Paribasan, & jvwiki (`corpus_v2/sentences_validated.jsonl`),
    dengan penaikan porsi kasus sulit (taling vs pepet, tumpukan pasangan, pasangan+suku, cecak/layar/wignyan/pangkon).
  - Gaya tulisan tangan realistis: distorsi elastis 2D (remap), variasi ketebalan goresan, jitter baseline,
    kemiringan glif, goresan terputus, coretan tepi, warna tinta hitam/biru/pensil pudar, dan pencampuran potongan nyata
    `hanacaraka` + `Jawa.zip` (split-safe: train hanya memakai split train).
  - Latar & optik dunia nyata: kertas polos/buram, buku tulis bergaris, kertas berpetak, bayangan jari/gradien cahaya,
    lengkungan halaman & transformasi perspektif ringan, blur gerak/fokus, noise sensor, kompresi JPEG rendah, rotasi ±10°.
  - Distraktor non-label: teks Latin beragam font/ukuran, angka, garis tabel/tepi, coretan margin.
  - Logging parameter pembuat lengkap per sampel (`seed`, `font`, `aug`, `unicode_gt`, `latin_gt`) & flag ablasi Fase 2g.

Penggunaan:
  python tools/synth_v2/generate_v2.py --n-train 13500 --n-val 1500 --out-dir dataset/external/synth_v2
"""
import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
from schema_55 import CLASSES_55, CLASS_TO_ID, NGLEGENA, PADA, PASANGAN, SANDHANGAN, validate_yolo_box, write_yolo_data_yaml
from orthography_rules import (
    CAKRA, CECAK, KERET, LAYAR, PANGKON, PENGKAL, PEPET, SUKU, TALING, TARUNG, WIGNYAN, WULU,
    PADA_ADEG, PADA_LINGSA, PADA_LUNGSI, NGLEGENA_LIST,
    latin_to_javanese_55, validate_javanese_unicode,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")

# Map karakter tunggal -> nama kelas atomik
BASE_CH_TO_NAME = {ch: name for name, ch in NGLEGENA_LIST}
MOD_CH_TO_NAME = {
    WULU: "wulu", SUKU: "suku", PEPET: "pepet", TALING: "taling", TARUNG: "tarung",
    LAYAR: "layar", CECAK: "cecak", WIGNYAN: "wignyan", PANGKON: "pangkon",
    CAKRA: "cakra", KERET: "keret", PENGKAL: "pengkal",
}
PADA_CH_TO_NAME = {PADA_ADEG: "pada_adeg", PADA_LINGSA: "pada_lingsa", PADA_LUNGSI: "pada_lungsi"}


def discover_ofl_fonts():
    font_dir = os.path.join(EXT_DIR, "fonts")
    fonts = []
    if os.path.isdir(font_dir):
        for fn in sorted(os.listdir(font_dir)):
            if fn.lower().endswith((".ttf", ".otf")):
                fp = os.path.join(font_dir, fn)
                for sz in (48, 58, 68):
                    try:
                        fonts.append((fn, sz, ImageFont.truetype(fp, sz, layout_engine=ImageFont.Layout.RAQM)))
                    except Exception:
                        pass
    if not fonts:
        raise RuntimeError(f"Tidak ada font OFL ditemukan di {font_dir}. Jalankan tools/datasets/download.py dahulu.")
    return fonts


def tight_bbox(mask_bool, pad=1):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    h, w = mask_bool.shape
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(w, int(xs.max()) + 1 + pad),
        min(h, int(ys.max()) + 1 + pad),
    )


def render_mask(txt, font, cw=380, ch=280, origin=(95, 65)):
    img = Image.new("L", (cw, ch), 255)
    d = ImageDraw.Draw(img)
    d.text(origin, txt, font=font, fill=0)
    gray = np.array(img)
    return gray < 190


def align_base_in_full(base_mask, full_mask):
    b_box = tight_bbox(base_mask, pad=0)
    if b_box is None:
        return base_mask, b_box
    bx0, by0, bx1, by1 = b_box
    tmpl = base_mask[by0:by1, bx0:bx1].astype(np.float32)
    src = full_mask.astype(np.float32)
    if tmpl.shape[0] >= src.shape[0] or tmpl.shape[1] >= src.shape[1]:
        return base_mask, b_box
    res = cv2.matchTemplate(src, tmpl, cv2.TM_CCORR)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    nx0, ny0 = max_loc
    shifted = np.zeros_like(full_mask, dtype=bool)
    hh, ww = tmpl.shape
    shifted[ny0:ny0 + hh, nx0:nx0 + ww] = base_mask[by0:by1, bx0:bx1]
    return shifted, (nx0, ny0, nx0 + ww, ny0 + hh)


def clean_diff_mask(full_mask, occupied_mask, dilate_k=3, min_comp_area=8):
    dil = cv2.dilate(occupied_mask.astype(np.uint8), np.ones((dilate_k, dilate_k), np.uint8)) > 0
    diff = full_mask & (~dil)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8))
    clean = np.zeros_like(diff)
    for i in range(1, num):
        if stats[i][4] >= min_comp_area:
            clean[lab == i] = True
    if clean.sum() < 8:
        clean = full_mask & (~occupied_mask)
    return clean


_CLUSTER_CACHE = {}


def render_cluster_cached(cluster_str, font_name, font_sz, pil_font):
    key = (font_name, font_sz, cluster_str)
    res = _CLUSTER_CACHE.get(key)
    if res is not None:
        return res
    out = render_cluster_from_unicode(cluster_str, pil_font)
    if len(_CLUSTER_CACHE) < 12000:
        _CLUSTER_CACHE[key] = out
    return out


def render_cluster_from_unicode(cluster_str, font):
    """Render 1 kluster Unicode (atau tanda baca pada) beserta kotak atomik per komponennya."""
    if cluster_str in PADA_CH_TO_NAME:
        cname = PADA_CH_TO_NAME[cluster_str]
        m = render_mask(cluster_str, font, 240, 220, origin=(60, 45))
        cb = tight_bbox(m, pad=2)
        if cb is None:
            return None, []
        cx0, cy0, cx1, cy1 = cb
        return (m[cy0:cy1, cx0:cx1].astype(np.uint8)) * 255, [(CLASS_TO_ID[cname], 0.0, 0.0, float(cx1 - cx0), float(cy1 - cy0))]

    # Parse komponen kluster kanonik
    base_ch = cluster_str[0]
    base_name = BASE_CH_TO_NAME.get(base_ch)
    if not base_name:
        return None, []

    idx = 1
    pas_name = None
    ended_pangkon = False
    if idx < len(cluster_str) and cluster_str[idx] == PANGKON:
        if idx + 1 < len(cluster_str) and cluster_str[idx + 1] in BASE_CH_TO_NAME:
            pas_name = "p_" + BASE_CH_TO_NAME[cluster_str[idx + 1]]
            idx += 2
        else:
            ended_pangkon = True
            idx += 1

    mods = [MOD_CH_TO_NAME[ch] for ch in cluster_str[idx:] if ch in MOD_CH_TO_NAME]
    if ended_pangkon:
        mods.append("pangkon")

    cw, ch_dim = 380, 280
    m_base = render_mask(base_ch, font, cw, ch_dim)
    full_mask = render_mask(cluster_str, font, cw, ch_dim)
    boxes = []

    if pas_name is None and not mods:
        bb = tight_bbox(m_base, pad=1)
        if bb:
            boxes.append((CLASS_TO_ID[base_name], *bb))
        full_mask = m_base
    elif pas_name is None:
        sb, box_b = align_base_in_full(m_base, full_mask)
        if box_b:
            boxes.append((CLASS_TO_ID[base_name], *box_b))
        # Jika hanya 1 modifier atau kombinasi standar (taling+tarung / swara+panyigeg)
        occupied = sb.copy()
        # Ekstrak tiap modifier secara bertahap bila >1 (agar kotak terpisah presisi)
        cur_txt = base_ch
        for mname in mods:
            prev_mask = render_mask(cur_txt, font, cw, ch_dim)
            cur_txt += {v: k for k, v in MOD_CH_TO_NAME.items()}[mname]
            cur_mask = render_mask(cur_txt, font, cw, ch_dim)
            sp_prev, _ = align_base_in_full(prev_mask, full_mask)
            sp_cur, _ = align_base_in_full(cur_mask, full_mask)
            dk = 4 if mname in ("suku", "pangkon", "pengkal") else 2
            diff = clean_diff_mask(sp_cur, sp_prev, dilate_k=dk, min_comp_area=6)
            if mname == "taling" and box_b:
                diff[:, box_b[0]:] = False
            elif mname == "tarung" and box_b:
                diff[:, :box_b[2]] = False
            bm = tight_bbox(diff, pad=2)
            if bm:
                boxes.append((CLASS_TO_ID[mname], *bm))
    else:
        # Ada pasangan: render (base + pangkon + p_root) lalu ekstrak pasangan, lalu tambah modifiers bertahap
        pas_seq = PANGKON + {n: ch for n, ch in NGLEGENA_LIST}[pas_name[2:]]
        pair_txt = base_ch + pas_seq
        m_pair = render_mask(pair_txt, font, cw, ch_dim)
        sb, box_b = align_base_in_full(m_base, full_mask)
        m_pas_only = clean_diff_mask(m_pair, align_base_in_full(m_base, m_pair)[0], dilate_k=3, min_comp_area=8)
        sp, box_p = align_base_in_full(m_pas_only, full_mask)
        if box_b:
            boxes.append((CLASS_TO_ID[base_name], *box_b))
        if box_p:
            boxes.append((CLASS_TO_ID[pas_name], *box_p))

        cur_txt = pair_txt
        for mname in mods:
            prev_mask = render_mask(cur_txt, font, cw, ch_dim)
            cur_txt += {v: k for k, v in MOD_CH_TO_NAME.items()}[mname]
            cur_mask = render_mask(cur_txt, font, cw, ch_dim)
            sp_prev, _ = align_base_in_full(prev_mask, full_mask)
            sp_cur, _ = align_base_in_full(cur_mask, full_mask)
            dk = 5 if mname == "suku" else 2
            diff = clean_diff_mask(sp_cur, sp_prev, dilate_k=dk, min_comp_area=6)
            if mname == "taling" and box_b:
                diff[:, box_b[0]:] = False
            elif mname == "tarung" and box_b:
                diff[:, :box_b[2]] = False
            bm = tight_bbox(diff, pad=2)
            if bm:
                boxes.append((CLASS_TO_ID[mname], *bm))

    cb = tight_bbox(full_mask, pad=2)
    if cb is None or not boxes:
        return None, []
    cx0, cy0, cx1, cy1 = cb
    patch = (full_mask[cy0:cy1, cx0:cx1].astype(np.uint8)) * 255
    rel = []
    for cid, bx0, by0, bx1, by1 in boxes:
        rx0, ry0 = max(0.0, float(bx0 - cx0)), max(0.0, float(by0 - cy0))
        rx1, ry1 = min(float(cx1 - cx0), float(bx1 - cx0)), min(float(cy1 - cy0), float(by1 - cy0))
        if rx1 - rx0 >= 3.0 and ry1 - ry0 >= 3.0:
            rel.append((cid, rx0, ry0, rx1, ry1))
    return patch, rel


def load_real_nglegena_by_split():
    """Muat potongan tulisan tangan nyata hanacaraka + Jawa.zip terpisah ketat antara train dan val."""
    pools = {"train": {n: [] for n, _ in NGLEGENA}, "val": {n: [] for n, _ in NGLEGENA}}
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
                if len(pools[sp][cls_name]) >= 180:
                    continue
                ipath = os.path.join(ROOT_DIR, rec["image"])
                gray = cv2.imread(ipath, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ys, xs = np.where(bw > 0)
                if len(ys) < 20:
                    continue
                crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                pools[sp][cls_name].append(crop)
    return pools


def apply_elastic_stroke_distortion(patch, boxes, rng, enable_elastic=True):
    """Simulasikan variasi goresan pena tulisan tangan: ketebalan acak, goresan putus, & elastis 2D."""
    h, w = patch.shape
    aug = patch.copy()

    # Ketebalan goresan (termasuk tipis seperti pensil atau tebal seperti spidol)
    r = rng.random()
    if r < 0.25:
        aug = cv2.erode(aug, np.ones((2, 2), np.uint8), iterations=1)
    elif r > 0.72:
        kx, ky = rng.choice([(2, 2), (3, 2), (2, 3)])
        aug = cv2.dilate(aug, np.ones((ky, kx), np.uint8), iterations=1)

    # Simulasi goresan terputus ringan pada tulisan pena
    if rng.random() < 0.22 and min(h, w) > 14:
        for _ in range(rng.randint(1, 3)):
            gx = rng.randint(2, w - 3)
            gy = rng.randint(2, h - 3)
            cv2.circle(aug, (gx, gy), rng.randint(1, 2), 0, -1)

    if enable_elastic and rng.random() < 0.65 and h >= 12 and w >= 12:
        alpha = rng.uniform(1.2, 3.2)
        sigma = rng.uniform(3.5, 6.0)
        dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        aug = cv2.remap(aug, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return aug, boxes


def create_paper_background(W, H, rng, enable_bg=True):
    if not enable_bg:
        return np.full((H, W, 3), 248, dtype=np.uint8), "plain_clean"

    style = rng.choice(["plain_cream", "plain_aged", "lined_notebook", "grid_paper", "gray_shadow"])
    if style == "plain_cream":
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:, :, 0] = rng.randint(218, 236)
        canvas[:, :, 1] = rng.randint(230, 246)
        canvas[:, :, 2] = rng.randint(238, 252)
    elif style == "plain_aged":
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:, :, 0] = rng.randint(185, 212)
        canvas[:, :, 1] = rng.randint(208, 228)
        canvas[:, :, 2] = rng.randint(222, 240)
    elif style == "gray_shadow":
        v = rng.randint(210, 238)
        canvas = np.full((H, W, 3), v, dtype=np.uint8)
    else:
        canvas = np.full((H, W, 3), rng.randint(236, 252), dtype=np.uint8)

    # Serat kertas lembut
    fiber = np.random.normal(0, rng.uniform(1.5, 4.0), (H, W, 1)).astype(np.float32)
    canvas = np.clip(canvas.astype(np.float32) + fiber, 0, 255).astype(np.uint8)

    if style == "lined_notebook":
        step = rng.randint(56, 86)
        y = rng.randint(36, 70)
        lcol = (rng.randint(175, 215), rng.randint(165, 200), rng.randint(155, 190))
        while y < H - 15:
            cv2.line(canvas, (0, y), (W, y), lcol, 1)
            y += step
        if rng.random() < 0.6:
            mx = rng.randint(32, 68)
            cv2.line(canvas, (mx, 0), (mx, H), (140, 140, 215), 1)
    elif style == "grid_paper":
        step = rng.randint(24, 38)
        gcol = (rng.randint(190, 220), rng.randint(190, 220), rng.randint(190, 220))
        for x in range(0, W, step):
            cv2.line(canvas, (x, 0), (x, H), gcol, 1)
        for y in range(0, H, step):
            cv2.line(canvas, (0, y), (W, y), gcol, 1)

    return canvas, style


def apply_optical_degradations(canvas, labels_abs, rng, enable_optics=True):
    H, W = canvas.shape[:2]
    if not enable_optics:
        return canvas, labels_abs, {}

    ops = {}
    # 1. Gradien cahaya / bayangan jari diagonal
    if rng.random() < 0.72:
        gx = np.linspace(rng.uniform(0.55, 1.0), rng.uniform(0.82, 1.06), W, dtype=np.float32)
        gy = np.linspace(rng.uniform(0.60, 1.0), rng.uniform(0.84, 1.06), H, dtype=np.float32)
        shad = np.outer(gy, gx)
        if rng.random() < 0.35:
            # Bayangan elips lembut (tangan / HP)
            cx, cy = rng.randint(0, W), rng.randint(0, H)
            yy, xx = np.ogrid[:H, :W]
            dist = ((xx - cx) / max(1, W * 0.4)) ** 2 + ((yy - cy) / max(1, H * 0.4)) ** 2
            shad *= np.clip(0.58 + 0.42 * np.tanh(dist - 0.3), 0.45, 1.0).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) * shad[..., None], 0, 255).astype(np.uint8)
        ops["lighting_grad"] = True

    # 2. Rotasi & geser halus (±6 derajat) beserta pembaruan bounding box akurat
    if rng.random() < 0.55 and labels_abs:
        angle = rng.uniform(-5.5, 5.5)
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
        ops["rot_deg"] = round(angle, 2)

    # 3. Motion / Gaussian blur
    if rng.random() < 0.35:
        if rng.random() < 0.5:
            k = rng.choice([3, 5])
            ker = np.zeros((k, k), dtype=np.float32)
            ker[k // 2, :] = 1.0 / k
            canvas = cv2.filter2D(canvas, -1, ker)
            ops["motion_blur"] = k
        else:
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
            ops["gaussian_blur"] = 3

    # 4. Sensor noise
    if rng.random() < 0.50:
        sig = rng.uniform(2.0, 6.5)
        noise = np.random.normal(0, sig, canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        ops["noise_sigma"] = round(sig, 2)

    return canvas, labels_abs, ops


def select_weighted_sentences(corpus):
    """Naikkan bobot kalimat yang mengandung kasus sulit (cecak, layar, wignyan, taling/pepet, pasangan+suku)."""
    weights = []
    for s in corpus:
        f = s.get("feats", {})
        w = 1.0
        w += 0.8 * min(3, f.get("cecak", 0))
        w += 0.6 * min(2, f.get("taling", 0))
        w += 0.6 * min(2, f.get("pepet", 0))
        w += 0.7 * min(3, f.get("pangkon_pasangan", 0))
        w += 0.5 * min(2, f.get("layar", 0) + f.get("wignyan", 0))
        weights.append(w)
    return weights


def synthesize_sample(fonts, real_pool, corpus, weights, split, sample_idx, rng, args):
    # Variasi rasio dimensi: baris memanjang (960x320 / 768x256) atau halaman multi-baris (640x640 / 800x600 / 960x720)
    is_line_strip = (rng.random() < 0.55)
    if is_line_strip:
        W = rng.choice([720, 896, 1024, 1280])
        H = rng.choice([220, 280, 340])
        num_lines = 1
    else:
        W = rng.choice([640, 800, 960])
        H = rng.choice([540, 640, 720])
        num_lines = rng.randint(2, 4)

    canvas, bg_style = create_paper_background(W, H, rng, enable_bg=not args.no_bg_paper)

    # Variasi tinta (hitam pekat, biru pulpen, pensil abu-abu pudar)
    ink_kind = rng.choice(["black_pen", "black_pen", "blue_pen", "faded_pencil"])
    if ink_kind == "black_pen":
        ink_col = np.array([rng.randint(10, 48)] * 3, dtype=np.float32)
    elif ink_kind == "blue_pen":
        ink_col = np.array([rng.randint(115, 175), rng.randint(28, 70), rng.randint(15, 50)], dtype=np.float32)
    else:
        v = rng.randint(75, 125)
        ink_col = np.array([v, v, v], dtype=np.float32)

    y_cursor = rng.randint(24, 48)
    has_distractor = (not args.no_distractors) and (rng.random() < 0.35)
    if has_distractor:
        cap = rng.choice([
            "Wedang bubuk tanpa gula, aja ngamuk mundhak tuwa.",
            "Latihan Nulis Aksara Jawa Kelas VIII",
            "ha na ca ra ka da ta sa wa la",
            "No. 14 - Transliterasi Carakan 2026",
        ])
        cv2.putText(canvas, cap, (rng.randint(14, 40), y_cursor + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.45, 0.72),
                    tuple(int(x) for x in ink_col), rng.choice([1, 2]), cv2.LINE_AA)
        y_cursor += rng.randint(34, 52)

    font_name, font_sz, pil_font = rng.choice(fonts)
    labels_abs = []
    gt_uni_lines = []
    gt_lat_lines = []

    for l_i in range(num_lines):
        if y_cursor > H - 70:
            break
        s_item = rng.choices(corpus, weights=weights, k=1)[0]
        uni_s = s_item["unicode"]
        lat_s = s_item["latin"]
        ok, _, clusters = validate_javanese_unicode(uni_s, strict_55=True)
        if not ok or not clusters:
            continue

        scale = rng.uniform(0.58, 1.15)
        x_cursor = rng.randint(18, 44)
        max_row_h = 0
        placed_clusters = []

        for cl in clusters:
            # Beri peluang 12% memakai tulisan tangan nyata Jawa.zip/Hanacaraka jika ini nglegena murni & split aman
            if len(cl) == 1 and cl in BASE_CH_TO_NAME and rng.random() < 0.14:
                bname = BASE_CH_TO_NAME[cl]
                pool = real_pool.get(bname, [])
                if pool:
                    crop = rng.choice(pool)
                    th = int(round(rng.uniform(32, 54) * scale))
                    sc = th / max(crop.shape)
                    nw, nh = max(8, int(round(crop.shape[1] * sc))), max(8, int(round(crop.shape[0] * sc)))
                    patch = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                    rel_boxes = [(CLASS_TO_ID[bname], 0.0, 0.0, float(nw), float(nh))]
                else:
                    patch, rel_boxes = render_cluster_cached(cl, font_name, font_sz, pil_font)
            else:
                patch, rel_boxes = render_cluster_cached(cl, font_name, font_sz, pil_font)

            if patch is None or not rel_boxes:
                continue

            # Skala patch & distorsi elastis tulisan tangan
            ph0, pw0 = patch.shape
            nw, nh = max(6, int(round(pw0 * scale))), max(6, int(round(ph0 * scale)))
            patch_s = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA)
            rel_s = [(cid, bx0 * (nw / pw0), by0 * (nh / ph0), bx1 * (nw / pw0), by1 * (nh / ph0))
                     for cid, bx0, by0, bx1, by1 in rel_boxes]
            patch_s, rel_s = apply_elastic_stroke_distortion(patch_s, rel_s, rng, enable_elastic=not args.no_elastic)

            ph, pw = patch_s.shape
            dy_jit = rng.randint(-3, 3)
            py = max(4, y_cursor + dy_jit)
            if x_cursor + pw >= W - 16 or py + ph >= H - 8:
                break

            alpha = (patch_s.astype(np.float32) / 255.0)[..., None]
            roi = canvas[py:py + ph, x_cursor:x_cursor + pw].astype(np.float32)
            canvas[py:py + ph, x_cursor:x_cursor + pw] = np.clip(roi * (1.0 - alpha) + ink_col * alpha, 0, 255).astype(np.uint8)

            for cid, bx0, by0, bx1, by1 in rel_s:
                labels_abs.append((cid, x_cursor + bx0, py + by0, x_cursor + bx1, py + by1))
            placed_clusters.append(cl)
            x_cursor += pw + rng.randint(1, 7)
            max_row_h = max(max_row_h, ph)

        if placed_clusters:
            gt_uni_lines.append("".join(placed_clusters))
            gt_lat_lines.append(lat_s)
        y_cursor += max(56, max_row_h + rng.randint(14, 28))

    # Coretan tepi/tabel opsional
    if not args.no_distractors and rng.random() < 0.25:
        cv2.rectangle(canvas, (3, 3), (W - 4, H - 4), (90, 90, 95), rng.choice([1, 2]))

    canvas, labels_abs, opt_meta = apply_optical_degradations(canvas, labels_abs, rng, enable_optics=not args.no_optics)

    # Filter & konversi YOLO tervalidasi
    yolo_boxes = []
    for cid, x0, y0, x1, y1 in labels_abs:
        xc = ((x0 + x1) / 2.0) / W
        yc = ((y0 + y1) / 2.0) / H
        bw = (x1 - x0) / W
        bh = (y1 - y0) / H
        if bw * W < 3.0 or bh * H < 3.0:
            continue
        ok, _ = validate_yolo_box(cid, xc, yc, bw, bh)
        if ok:
            yolo_boxes.append((cid, xc, yc, bw, bh))

    jpeg_q = rng.randint(42, 94) if (not args.no_optics and rng.random() < 0.5) else 92
    meta = {
        "id": f"synth2_{split}_{sample_idx:05d}",
        "split": split,
        "seed": args.seed + sample_idx,
        "font": font_name,
        "font_px": font_sz,
        "width": W,
        "height": H,
        "bg_style": bg_style,
        "ink": ink_kind,
        "distractor": has_distractor,
        "jpeg_quality": jpeg_q,
        "optics": opt_meta,
        "unicode_gt": "\n".join(gt_uni_lines),
        "latin_gt": "\n".join(gt_lat_lines),
        "num_boxes": len(yolo_boxes),
    }
    return canvas, yolo_boxes, meta


def main():
    parser = argparse.ArgumentParser(description="Generator Data Sintetis v2 AksaraScan (OFL Only)")
    parser.add_argument("--n-train", type=int, default=13500)
    parser.add_argument("--n-val", type=int, default=1500)
    parser.add_argument("--out-dir", default="dataset/external/synth_v2")
    parser.add_argument("--seed", type=int, default=2026)
    # Flag ablasi Fase 2g
    parser.add_argument("--no-elastic", action="store_true")
    parser.add_argument("--no-bg-paper", action="store_true")
    parser.add_argument("--no-optics", action="store_true")
    parser.add_argument("--no-distractors", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT_DIR, args.out_dir)
    for sp in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", sp), exist_ok=True)

    corpus_path = os.path.join(EXT_DIR, "corpus_v2", "sentences_validated.jsonl")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    weights = select_weighted_sentences(corpus)
    fonts = discover_ofl_fonts()
    real_pools = load_real_nglegena_by_split()

    print(f"[Synth v2] Font OFL: {len(fonts)} variasi | Korpus kalimat: {len(corpus)} | Pool tulisan tangan train: {sum(len(v) for v in real_pools['train'].values())}")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    mf = open(manifest_path, "w", encoding="utf-8")

    try:
        for split, count in (("train", args.n_train), ("val", args.n_val)):
            pool = real_pools["train"] if split == "train" else real_pools["val"]
            for i in range(count):
                canvas, boxes, meta = synthesize_sample(fonts, pool, corpus, weights, split, i, rng, args)
                if not boxes:
                    continue
                img_path = os.path.join(out_dir, "images", split, f"{meta['id']}.jpg")
                lbl_path = os.path.join(out_dir, "labels", split, f"{meta['id']}.txt")
                cv2.imwrite(img_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, meta["jpeg_quality"]])
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    for cid, xc, yc, bw, bh in boxes:
                        lf.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                meta["image"] = os.path.relpath(img_path, ROOT_DIR)
                meta["label"] = os.path.relpath(lbl_path, ROOT_DIR)
                mf.write(json.dumps(meta, ensure_ascii=False) + "\n")
                if (i + 1) % 1000 == 0 or (i + 1) == count:
                    print(f"  [{split}] {i + 1}/{count} selesai...")
    finally:
        mf.close()

    write_yolo_data_yaml(out_dir)
    print(f"[SELESAI] Data Sintetis v2 tersimpan di {out_dir}")


if __name__ == "__main__":
    main()
