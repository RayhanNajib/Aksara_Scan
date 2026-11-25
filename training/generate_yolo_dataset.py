"""
AksaraScan v2.5 Hybrid Multi-Font Dataset Generator (55 Kelas Atomik)
Mencakup:
  - 20 Aksara Nglegena (0..19) — gabungan tulisan tangan nyata (Jawa.zip) + 3 Font OpenType
  - 20 Aksara Pasangan (20..39) — 3 Font OpenType (Javanese Text, Noto Sans Javanese, Tuladha Jejeg OT)
  - 12 Sandhangan & Pangkon (40..51) — termasuk kombinasi nested Pasangan + Suku/Wulu/Pepet/Taling
  -  3 Tanda Baca Jawa / Pada (52..54) — pada_adeg (꧋), pada_lingsa (꧈), pada_lungsi (꧉)
  + Dukungan tata letak kalimat bersambung rapat (scriptio continua) & distraktor teks Latin non-label.
"""
import os
import random
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, "dataset_yolo")
JAWA_ZIP = os.path.join(ROOT_DIR, "dataset", "Jawa.zip")

FONT_PATHS = [
    "C:/Windows/Fonts/javatext.ttf",
    os.path.join(ROOT_DIR, "fonts", "NotoSansJavanese.ttf"),
    os.path.join(ROOT_DIR, "fonts", "TuladhaJejegOT-Regular.ttf"),
]

NGLEGENA = [
    ("ha", "\uA9B2"), ("na", "\uA9A4"), ("ca", "\uA995"), ("ra", "\uA9AB"), ("ka", "\uA98F"),
    ("da", "\uA9A2"), ("ta", "\uA9A0"), ("sa", "\uA9B1"), ("wa", "\uA9AE"), ("la", "\uA9AD"),
    ("pa", "\uA9A5"), ("dha", "\uA99D"), ("ja", "\uA997"), ("ya", "\uA9AA"), ("nya", "\uA99A"),
    ("ma", "\uA9A9"), ("ga", "\uA992"), ("ba", "\uA9A7"), ("tha", "\uA99B"), ("nga", "\uA994"),
]
PASANGAN = [(f"p_{name}", "\uA9C0" + ch) for name, ch in NGLEGENA]
SANDHANGAN = [
    ("wulu", "\uA9B6"),
    ("suku", "\uA9B8"),
    ("pepet", "\uA9BC"),
    ("taling", "\uA9BA"),
    ("tarung", "\uA9B4"),
    ("layar", "\uA982"),
    ("cecak", "\uA981"),
    ("wignyan", "\uA983"),
    ("pangkon", "\uA9C0"),
    ("cakra", "\uA9BF"),
    ("keret", "\uA9BD"),
    ("pengkal", "\uA9BE"),
]
PADA = [
    ("pada_adeg", "\uA9CB"),
    ("pada_lingsa", "\uA9C8"),
    ("pada_lungsi", "\uA9C9"),
]

ALL_CLASSES = [n for n, _ in NGLEGENA] + [n for n, _ in PASANGAN] + [n for n, _ in SANDHANGAN] + [n for n, _ in PADA]
CLASS_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}


def tight_bbox(mask_bool, pad=1):
    ys, xs = np.where(mask_bool)
    if len(ys) == 0:
        return None
    h, w = mask_bool.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return (x0, y0, x1, y1)


def render_mask(txt, font, canvas_w=360, canvas_h=260, origin=(90, 50)):
    img = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(img)
    draw.text(origin, txt, font=font, fill=0)
    gray = np.array(img)
    return gray < 185, gray


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
    h, w = tmpl.shape
    shifted[ny0:ny0 + h, nx0:nx0 + w] = base_mask[by0:by1, bx0:bx1]
    return shifted, (nx0, ny0, nx0 + w, ny0 + h)


def clean_diff_mask(full_mask, occupied_mask, dilate_k=4, min_comp_area=12):
    dil = cv2.dilate(occupied_mask.astype(np.uint8), np.ones((dilate_k, dilate_k), np.uint8)) > 0
    diff = full_mask & (~dil)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8))
    clean = np.zeros_like(diff)
    for i in range(1, num):
        if stats[i][4] >= min_comp_area:
            clean[lab == i] = True
    if clean.sum() < 10:
        clean = full_mask & (~occupied_mask)
    return clean


def render_pada_glyph(font, pada_name):
    ch = dict(PADA)[pada_name]
    cid = CLASS_TO_ID[pada_name]
    m, _ = render_mask(ch, font, 240, 220, origin=(60, 40))
    cb = tight_bbox(m, pad=2)
    if cb is None:
        return None, []
    cx0, cy0, cx1, cy1 = cb
    patch = (m[cy0:cy1, cx0:cx1].astype(np.uint8)) * 255
    return patch, [(cid, 0, 0, cx1 - cx0, cy1 - cy0)]


def render_syllable_cluster(font, base_idx, mod_type, sub_mod=None):
    b_name, b_ch = NGLEGENA[base_idx]
    b_id = CLASS_TO_ID[b_name]
    cw, ch_dim = 360, 260

    m_base, _ = render_mask(b_ch, font, cw, ch_dim)
    boxes = []

    if mod_type == "nglegena":
        full_mask = m_base
        box_b = tight_bbox(m_base)
        if box_b:
            boxes.append((b_id, *box_b))

    elif mod_type in ("wulu", "suku", "pepet", "layar", "cecak", "wignyan", "pangkon", "pengkal", "cakra", "keret"):
        m_ch = dict(SANDHANGAN)[mod_type]
        m_id = CLASS_TO_ID[mod_type]
        full_mask, _ = render_mask(b_ch + m_ch, font, cw, ch_dim)
        shifted_base, box_b = align_base_in_full(m_base, full_mask)
        diff = clean_diff_mask(full_mask, shifted_base, dilate_k=4 if mod_type in ("suku", "pangkon", "pengkal") else 2)
        box_m = tight_bbox(diff, pad=2)
        if box_b:
            boxes.append((b_id, *box_b))
        if box_m:
            boxes.append((m_id, *box_m))

    elif mod_type == "taling":
        t_id = CLASS_TO_ID["taling"]
        full_mask, _ = render_mask(b_ch + "\uA9BA", font, cw, ch_dim)
        shifted_base, box_b = align_base_in_full(m_base, full_mask)
        diff = clean_diff_mask(full_mask, shifted_base, dilate_k=2)
        if box_b:
            diff_left = diff.copy()
            diff_left[:, box_b[0]:] = False
            box_t = tight_bbox(diff_left) or tight_bbox(diff)
            boxes.append((b_id, *box_b))
            if box_t:
                boxes.append((t_id, *box_t))

    elif mod_type == "taling_tarung":
        t_id = CLASS_TO_ID["taling"]
        r_id = CLASS_TO_ID["tarung"]
        full_mask, _ = render_mask(b_ch + "\uA9BA\uA9B4", font, cw, ch_dim)
        shifted_base, box_b = align_base_in_full(m_base, full_mask)
        diff = clean_diff_mask(full_mask, shifted_base, dilate_k=2)
        if box_b:
            diff_left = diff.copy()
            diff_left[:, box_b[0]:] = False
            diff_right = diff.copy()
            diff_right[:, :box_b[2]] = False
            box_t = tight_bbox(diff_left)
            box_r = tight_bbox(diff_right)
            boxes.append((b_id, *box_b))
            if box_t:
                boxes.append((t_id, *box_t))
            if box_r:
                boxes.append((r_id, *box_r))

    elif mod_type.startswith("pasangan:"):
        p_idx = int(mod_type.split(":")[1])
        p_name, p_seq = PASANGAN[p_idx]
        p_id = CLASS_TO_ID[p_name]

        txt = b_ch + p_seq
        m_pair, _ = render_mask(txt, font, cw, ch_dim)
        sb_pair, box_b = align_base_in_full(m_base, m_pair)
        m_pas_only = clean_diff_mask(m_pair, sb_pair, dilate_k=3)
        box_p = tight_bbox(m_pas_only, pad=2)

        full_mask = m_pair
        if sub_mod is None:
            if box_b:
                boxes.append((b_id, *box_b))
            if box_p:
                boxes.append((p_id, *box_p))
        elif sub_mod in ("wulu", "suku", "pepet", "layar", "cecak", "wignyan"):
            sm_ch = dict(SANDHANGAN)[sub_mod]
            sm_id = CLASS_TO_ID[sub_mod]
            m_sub, _ = render_mask(txt + sm_ch, font, cw, ch_dim)
            sb2, box_b2 = align_base_in_full(m_base, m_sub)
            sp2, box_p2 = align_base_in_full(m_pas_only, m_sub)
            diff_sm = clean_diff_mask(m_sub, sb2 | sp2, dilate_k=5 if sub_mod == "suku" else 3)
            box_sm = tight_bbox(diff_sm, pad=2)
            full_mask = m_sub
            if box_b2:
                boxes.append((b_id, *box_b2))
            if box_p2:
                boxes.append((p_id, *box_p2))
            if box_sm:
                boxes.append((sm_id, *box_sm))
        elif sub_mod == "taling":
            t_id = CLASS_TO_ID["taling"]
            m_sub, _ = render_mask(txt + "\uA9BA", font, cw, ch_dim)
            sb2, box_b2 = align_base_in_full(m_base, m_sub)
            sp2, box_p2 = align_base_in_full(m_pas_only, m_sub)
            diff_t = clean_diff_mask(m_sub, sb2 | sp2, dilate_k=2)
            if box_b2:
                diff_t[:, box_b2[0]:] = False
            box_t = tight_bbox(diff_t, pad=2)
            full_mask = m_sub
            if box_b2:
                boxes.append((b_id, *box_b2))
            if box_p2:
                boxes.append((p_id, *box_p2))
            if box_t:
                boxes.append((t_id, *box_t))

    cb = tight_bbox(full_mask, pad=2)
    if cb is None or not boxes:
        return None, []
    cx0, cy0, cx1, cy1 = cb
    patch = (full_mask[cy0:cy1, cx0:cx1].astype(np.uint8)) * 255
    rel_boxes = []
    for cid, bx0, by0, bx1, by1 in boxes:
        rx0 = max(0, bx0 - cx0)
        ry0 = max(0, by0 - cy0)
        rx1 = min(cx1 - cx0, bx1 - cx0)
        ry1 = min(cy1 - cy0, by1 - cy0)
        if rx1 - rx0 >= 4 and ry1 - ry0 >= 4:
            rel_boxes.append((cid, rx0, ry0, rx1, ry1))
    return patch, rel_boxes


def load_real_nglegena_pool(max_per_class=70):
    pool = {name: [] for name, _ in NGLEGENA}
    if not os.path.exists(JAWA_ZIP):
        return pool
    with zipfile.ZipFile(JAWA_ZIP, "r") as z:
        by_cls = {name: [] for name, _ in NGLEGENA}
        for n in z.namelist():
            parts = n.split("/")
            if len(parts) >= 5 and parts[1] == "all_class" and parts[2] == "Vokal A" and parts[4]:
                cls_name = parts[3]
                if cls_name in by_cls:
                    by_cls[cls_name].append(n)
        for cls_name, files in by_cls.items():
            random.shuffle(files)
            for fn in files[:max_per_class]:
                arr = np.frombuffer(z.read(fn), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                ys, xs = np.where(bw > 0)
                if len(ys) < 20:
                    continue
                crop = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                pool[cls_name].append(crop)
    return pool


def augment_patch(patch, boxes, target_scale=None):
    h, w = patch.shape
    aug = patch.copy()
    r = random.random()
    if r < 0.22:
        aug = cv2.erode(aug, np.ones((2, 2), np.uint8), iterations=1)
    elif r > 0.74:
        aug = cv2.dilate(aug, np.ones((2, 2), np.uint8), iterations=1)

    scale = target_scale if target_scale is not None else random.uniform(0.48, 1.05)
    nw = max(6, int(round(w * scale)))
    nh = max(6, int(round(h * scale)))
    aug = cv2.resize(aug, (nw, nh), interpolation=cv2.INTER_AREA)

    new_boxes = []
    for cid, x0, y0, x1, y1 in boxes:
        nx0, ny0 = x0 * (nw / w), y0 * (nh / h)
        nx1, ny1 = x1 * (nw / w), y1 * (nh / h)
        if nx1 - nx0 >= 3 and ny1 - ny0 >= 3:
            new_boxes.append((cid, nx0, ny0, nx1, ny1))
    return aug, new_boxes


def synthesize_page(fonts, real_pool, img_size=416):
    bg_type = random.choice(["white", "cream", "gray"])
    if bg_type == "white":
        canvas = np.full((img_size, img_size, 3), random.randint(242, 255), dtype=np.uint8)
    elif bg_type == "cream":
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        canvas[:, :, 0] = random.randint(215, 232)
        canvas[:, :, 1] = random.randint(230, 245)
        canvas[:, :, 2] = random.randint(238, 252)
    else:
        canvas = np.full((img_size, img_size, 3), random.randint(220, 242), dtype=np.uint8)

    if random.random() < 0.42:
        line_y = random.randint(45, 85)
        step = random.randint(80, 110)
        line_col = (random.randint(155, 195), random.randint(155, 195), random.randint(150, 190))
        while line_y < img_size - 20:
            cv2.line(canvas, (0, line_y), (img_size, line_y), line_col, 1)
            line_y += step

    if random.random() < 0.82:
        ink_col = np.array([random.randint(8, 52)] * 3, dtype=np.float32)
    else:
        ink_col = np.array([random.randint(85, 145), random.randint(20, 55), random.randint(10, 40)], dtype=np.float32)

    all_labels = []
    y_cursor = random.randint(18, 38)

    # Opsional tambahkan teks Latin distraktor di atas halaman (agar model tidak salah deteksi teks Latin!)
    if random.random() < 0.25:
        latin_samples = [
            "Wedang bubuk tanpa gula, aja ngamuk.",
            "Soal Aksara Jawa Kelas VIII",
            "Transliterasi Latin -> Hanacaraka",
            "Mangan sega goreng nganggo tempe.",
        ]
        cv2.putText(canvas, random.choice(latin_samples), (14, y_cursor + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.35, 0.52), (40, 40, 45), 1, cv2.LINE_AA)
        y_cursor += random.randint(28, 42)

    is_continuous_sentence = (random.random() < 0.50)
    num_lines = random.randint(2, 4)

    for line_i in range(num_lines):
        if y_cursor > img_size - 65:
            break
        font = random.choice(fonts)
        line_scale = random.uniform(0.55, 0.92) if not is_continuous_sentence else random.uniform(0.45, 0.78)
        x_cursor = random.randint(10, 26)
        max_row_h = 0

        tokens_in_line = []
        if is_continuous_sentence and line_i == 0 and random.random() < 0.7:
            tokens_in_line.append(("pada", "pada_adeg"))

        n_words = random.randint(5, 9) if is_continuous_sentence else random.randint(3, 6)
        for t_idx in range(n_words):
            if is_continuous_sentence and t_idx == n_words // 2 and random.random() < 0.45:
                tokens_in_line.append(("pada", "pada_lingsa"))
            tokens_in_line.append(("syllable", None))

        if is_continuous_sentence and line_i == num_lines - 1 and random.random() < 0.75:
            tokens_in_line.append(("pada", "pada_lungsi"))

        for kind, val in tokens_in_line:
            if kind == "pada":
                patch, rel_boxes = render_pada_glyph(font, val)
                if patch is None:
                    continue
                patch, p_boxes = augment_patch(patch, rel_boxes, target_scale=line_scale)
            else:
                b_idx = random.randrange(len(NGLEGENA))
                b_name = NGLEGENA[b_idx][0]
                if not is_continuous_sentence and random.random() < 0.20 and real_pool[b_name]:
                    crop = random.choice(real_pool[b_name])
                    target_h = random.randint(32, 54)
                    sc = target_h / max(crop.shape)
                    nw, nh = max(10, int(round(crop.shape[1] * sc))), max(10, int(round(crop.shape[0] * sc)))
                    patch = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                    p_boxes = [(CLASS_TO_ID[b_name], 0.0, 0.0, float(nw), float(nh))]
                else:
                    r2 = random.random()
                    if r2 < 0.14:
                        mod_type, sub_mod = "nglegena", None
                    elif r2 < 0.54:
                        mod_type = random.choice([
                            "wulu", "suku", "suku", "pepet", "taling", "taling_tarung",
                            "layar", "cecak", "wignyan", "pangkon", "cakra", "keret", "pengkal"
                        ])
                        sub_mod = None
                    else:
                        p_idx = random.randrange(len(PASANGAN))
                        mod_type = f"pasangan:{p_idx}"
                        sub_mod = random.choice([None, "wulu", "suku", "suku", "pepet", "taling", "cecak", "layar", "wignyan"])
                    patch, rel_boxes = render_syllable_cluster(font, b_idx, mod_type, sub_mod)
                    if patch is None:
                        continue
                    patch, p_boxes = augment_patch(patch, rel_boxes, target_scale=line_scale)

            ph, pw = patch.shape
            if x_cursor + pw >= img_size - 8 or y_cursor + ph >= img_size - 6:
                break

            alpha = (patch.astype(np.float32) / 255.0)[..., None]
            roi = canvas[y_cursor:y_cursor + ph, x_cursor:x_cursor + pw].astype(np.float32)
            canvas[y_cursor:y_cursor + ph, x_cursor:x_cursor + pw] = np.clip(roi * (1.0 - alpha) + ink_col * alpha, 0, 255).astype(np.uint8)

            for cid, bx0, by0, bx1, by1 in p_boxes:
                gx0, gy0 = x_cursor + bx0, y_cursor + by0
                gx1, gy1 = x_cursor + bx1, y_cursor + by1
                xc = ((gx0 + gx1) / 2.0) / img_size
                yc = ((gy0 + gy1) / 2.0) / img_size
                bw_n, bh_n = (gx1 - gx0) / img_size, (gy1 - gy0) / img_size
                all_labels.append((cid, xc, yc, bw_n, bh_n))

            gap = random.randint(2, 7) if is_continuous_sentence else random.randint(8, 18)
            x_cursor += pw + gap
            max_row_h = max(max_row_h, ph)

        y_cursor += max(56, max_row_h + random.randint(12, 24))

    if random.random() < 0.6:
        gx = np.linspace(random.uniform(0.76, 1.0), random.uniform(0.86, 1.05), img_size)
        gy = np.linspace(random.uniform(0.78, 1.0), random.uniform(0.86, 1.05), img_size)
        canvas = np.clip(canvas.astype(np.float32) * np.outer(gy, gx)[..., None], 0, 255).astype(np.uint8)

    if random.random() < 0.45:
        noise = np.random.normal(0, random.uniform(1.5, 4.5), canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.30:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas, all_labels


def main(n_train=1400, n_val=240):
    random.seed(108)
    np.random.seed(108)

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    fonts = []
    for p in FONT_PATHS:
        if os.path.exists(p):
            for sz in (46, 54):
                fonts.append(ImageFont.truetype(p, sz, layout_engine=ImageFont.Layout.RAQM))
    print(f"[1/3] Memuat {len(fonts)} variasi font OpenType RAQM + tulisan tangan Jawa.zip...")
    real_pool = load_real_nglegena_pool(max_per_class=70)
    print(f"      Termuat {sum(len(v) for v in real_pool.values())} sampel tulisan tangan nyata.")

    for split, count in (("train", n_train), ("val", n_val)):
        print(f"[2/3] Menghasilkan {count} citra halaman {split} (55 kelas · 3 font · scriptio continua)...")
        for i in range(count):
            img, labels = synthesize_page(fonts, real_pool, img_size=416)
            cv2.imwrite(os.path.join(OUT_DIR, "images", split, f"aks_{split}_{i:04d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(OUT_DIR, "labels", split, f"aks_{split}_{i:04d}.txt"), "w", encoding="utf-8") as f:
                for cid, xc, yc, bw_n, bh_n in labels:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}\n")

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(ALL_CLASSES)}\n")
        f.write("names:\n")
        for idx, name in enumerate(ALL_CLASSES):
            f.write(f"  {idx}: {name}\n")

    print(f"[3/3] Selesai! Dataset 55 kelas tersimpan di {OUT_DIR}.")


if __name__ == "__main__":
    main()
