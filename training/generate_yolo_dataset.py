"""
AksaraScan v2.0 Hybrid Dataset Generator (Nglegena + Pasangan + Sandhangan)
Menghasilkan dataset deteksi objek format YOLO (images + labels + data.yaml)
dengan bounding box presisi piksel tingkat atomik:
  - 20 Aksara Nglegena (kelas 0..19) — gabungan tulisan tangan nyata (Jawa.zip) + OpenType RAQM
  - 20 Aksara Pasangan (kelas 20..39) — OpenType RAQM dengan augmentasi tulisan tangan
  - 12 Sandhangan & Pangkon (kelas 40..51) — wulu, suku, pepet, taling, tarung, layar, cecak, wignyan, pangkon, cakra, keret, pengkal
"""
import os
import io
import random
import zipfile
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT_DIR, "dataset_yolo")
JAWA_ZIP = os.path.join(ROOT_DIR, "dataset", "Jawa.zip")
FONT_PATH = "C:/Windows/Fonts/javatext.ttf"

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

ALL_CLASSES = [n for n, _ in NGLEGENA] + [n for n, _ in PASANGAN] + [n for n, _ in SANDHANGAN]
CLASS_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}

# Beberapa pasangan yang posisinya di sebelah kanan huruf induk (bukan di bawah)
RIGHT_PASANGAN = {"p_ha", "p_sa", "p_pa"}


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
    """Temukan posisi geser (dx, dy) mask huruf dasar di dalam mask gabungan."""
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


def render_syllable_cluster(font, base_idx, mod_type, sub_mod=None):
    """
    Merender satu gugus suku kata aksara Jawa dan mengekstrak mask ink +
    daftar bounding box tingkat atomik [(class_id, x0, y0, x1, y1), ...].
    """
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

    elif mod_type in ("wulu", "suku", "pepet", "layar", "cecak", "wignyan", "pangkon", "pengkal"):
        m_ch = dict(SANDHANGAN)[mod_type]
        m_id = CLASS_TO_ID[mod_type]
        full_mask, _ = render_mask(b_ch + m_ch, font, cw, ch_dim)
        shifted_base, box_b = align_base_in_full(m_base, full_mask)
        diff = full_mask & (~shifted_base)
        # Bersihkan artefak anti-aliasing 1 piksel di tepi sambungan
        diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        if diff.sum() < 18:
            diff = full_mask & (~shifted_base)
        box_m = tight_bbox(diff)
        if box_b:
            boxes.append((b_id, *box_b))
        if box_m:
            boxes.append((m_id, *box_m))

    elif mod_type in ("cakra", "keret"):
        m_ch = dict(SANDHANGAN)[mod_type]
        m_id = CLASS_TO_ID[mod_type]
        full_mask, _ = render_mask(b_ch + m_ch, font, cw, ch_dim)
        shifted_base, box_b = align_base_in_full(m_base, full_mask)
        diff = full_mask & (~shifted_base)
        box_m = tight_bbox(diff)
        if box_b:
            boxes.append((b_id, *box_b))
        if box_m:
            boxes.append((m_id, *box_m))

    elif mod_type == "taling":
        t_id = CLASS_TO_ID["taling"]
        full_mask, _ = render_mask(b_ch + "\uA9BA", font, cw, ch_dim)
        shifted_base, box_b = align_base_in_full(m_base, full_mask)
        diff = full_mask & (~shifted_base)
        if box_b:
            # Taling selalu berada di sebelah kiri huruf induk
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
        diff = full_mask & (~shifted_base)
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
        shifted_base, box_b = align_base_in_full(m_base, m_pair)
        diff_p = m_pair & (~shifted_base)
        box_p = tight_bbox(diff_p)

        full_mask = m_pair
        if box_b:
            boxes.append((b_id, *box_b))
        if box_p:
            boxes.append((p_id, *box_p))

        # Opsional tambahkan sandhangan pada pasangan! (mis. nki, nku, nko, nkar)
        if sub_mod in ("wulu", "suku", "pepet", "layar", "cecak", "wignyan"):
            sm_ch = dict(SANDHANGAN)[sub_mod]
            sm_id = CLASS_TO_ID[sub_mod]
            m_sub, _ = render_mask(txt + sm_ch, font, cw, ch_dim)
            shifted_pair, _ = align_base_in_full(m_pair, m_sub)
            diff_sm = m_sub & (~shifted_pair)
            box_sm = tight_bbox(diff_sm)
            full_mask = m_sub
            if box_sm:
                boxes.append((sm_id, *box_sm))
        elif sub_mod == "taling":
            t_id = CLASS_TO_ID["taling"]
            m_sub, _ = render_mask(txt + "\uA9BA", font, cw, ch_dim)
            shifted_pair, _ = align_base_in_full(m_pair, m_sub)
            diff_t = m_sub & (~shifted_pair)
            # Update koordinat box_b dan box_p terhadap pergeseran taling
            p_box = tight_bbox(m_pair)
            sp_box = tight_bbox(shifted_pair)
            if p_box and sp_box:
                dx = sp_box[0] - p_box[0]
                dy = sp_box[1] - p_box[1]
                boxes = [(cid, x0 + dx, y0 + dy, x1 + dx, y1 + dy) for (cid, x0, y0, x1, y1) in boxes]
                diff_t[:, sp_box[0]:] = False
            box_t = tight_bbox(diff_t)
            full_mask = m_sub
            if box_t:
                boxes.append((t_id, *box_t))

    # Crop rapat seluruh gugus agar mudah digabung ke kanvas halaman
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


def load_real_nglegena_pool(max_per_class=60):
    """Muat sampel tulisan tangan nyata Vokal A dari Jawa.zip untuk menjaga generalisasi."""
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


def augment_patch(patch, boxes):
    """Augmentasi ketebalan tinta & variasi kemiringan ringan sambil menjaga keakuratan box."""
    h, w = patch.shape
    aug = patch.copy()

    # Variasi ketebalan pulpen (tipis vs tebal)
    r = random.random()
    if r < 0.25:
        aug = cv2.erode(aug, np.ones((2, 2), np.uint8), iterations=1)
    elif r > 0.72:
        aug = cv2.dilate(aug, np.ones((2, 2), np.uint8), iterations=1)

    # Skala acak dengan mempertahankan rasio kotak
    scale = random.uniform(0.55, 1.05)
    nw = max(8, int(round(w * scale)))
    nh = max(8, int(round(h * scale)))
    aug = cv2.resize(aug, (nw, nh), interpolation=cv2.INTER_AREA)

    new_boxes = []
    for cid, x0, y0, x1, y1 in boxes:
        nx0 = x0 * (nw / w)
        ny0 = y0 * (nh / h)
        nx1 = x1 * (nw / w)
        ny1 = y1 * (nh / h)
        if nx1 - nx0 >= 4 and ny1 - ny0 >= 4:
            new_boxes.append((cid, nx0, ny0, nx1, ny1))
    return aug, new_boxes


def synthesize_page(fonts, real_pool, img_size=416):
    """Buat 1 halaman realistis (kertas bergaris/polos + bayangan) berisi 2–3 baris aksara."""
    # Latar kertas realistis (putih bersih, krem buku tulis, atau abu lembut)
    bg_type = random.choice(["white", "cream", "gray"])
    if bg_type == "white":
        base_val = random.randint(242, 255)
        canvas = np.full((img_size, img_size, 3), base_val, dtype=np.uint8)
    elif bg_type == "cream":
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        canvas[:, :, 0] = random.randint(215, 232)  # B
        canvas[:, :, 1] = random.randint(230, 245)  # G
        canvas[:, :, 2] = random.randint(238, 252)  # R
    else:
        v = random.randint(218, 242)
        canvas = np.full((img_size, img_size, 3), v, dtype=np.uint8)

    # Opsional tambahkan garis buku tulis SiDU tipis
    if random.random() < 0.45:
        line_y = random.randint(45, 85)
        step = random.randint(85, 115)
        line_col = (random.randint(155, 195), random.randint(155, 195), random.randint(150, 190))
        while line_y < img_size - 20:
            cv2.line(canvas, (0, line_y), (img_size, line_y), line_col, 1)
            line_y += step

    # Warna tinta (hitam jelaga atau biru pulpen)
    if random.random() < 0.78:
        ink_col = np.array([random.randint(12, 55)] * 3, dtype=np.float32)
    else:
        ink_col = np.array([random.randint(85, 145), random.randint(20, 55), random.randint(10, 40)], dtype=np.float32)

    all_labels = []
    num_lines = random.randint(2, 3)
    y_cursor = random.randint(22, 42)

    for _ in range(num_lines):
        if y_cursor > img_size - 70:
            break
        x_cursor = random.randint(14, 34)
        max_row_h = 0

        for _ in range(random.randint(3, 6)):
            b_idx = random.randrange(len(NGLEGENA))
            b_name = NGLEGENA[b_idx][0]
            mode_roll = random.random()

            if mode_roll < 0.22 and real_pool[b_name]:
                # Pakai tulisan tangan asli dari Jawa.zip
                crop = random.choice(real_pool[b_name])
                target_h = random.randint(36, 58)
                sc = target_h / max(crop.shape)
                nw = max(12, int(round(crop.shape[1] * sc)))
                nh = max(12, int(round(crop.shape[0] * sc)))
                patch = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                p_boxes = [(CLASS_TO_ID[b_name], 0.0, 0.0, float(nw), float(nh))]
            else:
                font = random.choice(fonts)
                r2 = random.random()
                if r2 < 0.16:
                    mod_type, sub_mod = "nglegena", None
                elif r2 < 0.56:
                    mod_type = random.choice([
                        "wulu", "suku", "pepet", "taling", "taling_tarung",
                        "layar", "cecak", "wignyan", "pangkon", "cakra", "keret", "pengkal"
                    ])
                    sub_mod = None
                else:
                    p_idx = random.randrange(len(PASANGAN))
                    mod_type = f"pasangan:{p_idx}"
                    sub_mod = random.choice([None, None, "wulu", "suku", "pepet", "taling", "cecak", "layar", "wignyan"])

                patch, rel_boxes = render_syllable_cluster(font, b_idx, mod_type, sub_mod)
                if patch is None:
                    continue
                patch, p_boxes = augment_patch(patch, rel_boxes)

            ph, pw = patch.shape
            if x_cursor + pw >= img_size - 10 or y_cursor + ph >= img_size - 8:
                break

            # Alpha blend tinta ke kertas
            alpha = (patch.astype(np.float32) / 255.0)[..., None]
            roi = canvas[y_cursor:y_cursor + ph, x_cursor:x_cursor + pw].astype(np.float32)
            blended = roi * (1.0 - alpha) + ink_col * alpha
            canvas[y_cursor:y_cursor + ph, x_cursor:x_cursor + pw] = np.clip(blended, 0, 255).astype(np.uint8)

            for cid, bx0, by0, bx1, by1 in p_boxes:
                gx0 = x_cursor + bx0
                gy0 = y_cursor + by0
                gx1 = x_cursor + bx1
                gy1 = y_cursor + by1
                xc = ((gx0 + gx1) / 2.0) / img_size
                yc = ((gy0 + gy1) / 2.0) / img_size
                bw = (gx1 - gx0) / img_size
                bh = (gy1 - gy0) / img_size
                all_labels.append((cid, xc, yc, bw, bh))

            x_cursor += pw + random.randint(8, 22)
            max_row_h = max(max_row_h, ph)

        y_cursor += max(68, max_row_h + random.randint(16, 30))

    # Tambah gradien pencahayaan / bayangan ringan (realisme kamera ponsel)
    if random.random() < 0.65:
        gx = np.linspace(random.uniform(0.75, 1.0), random.uniform(0.85, 1.05), img_size)
        gy = np.linspace(random.uniform(0.78, 1.0), random.uniform(0.85, 1.05), img_size)
        shade = np.outer(gy, gx)[..., None]
        canvas = np.clip(canvas.astype(np.float32) * shade, 0, 255).astype(np.uint8)

    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(1.5, 5.0), canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.35:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas, all_labels


def main(n_train=900, n_val=150):
    random.seed(42)
    np.random.seed(42)

    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, "labels", split), exist_ok=True)

    fonts = [
        ImageFont.truetype(FONT_PATH, sz, layout_engine=ImageFont.Layout.RAQM)
        for sz in (46, 54, 62)
    ]
    print("[1/3] Memuat pool tulisan tangan nyata dari Jawa.zip...")
    real_pool = load_real_nglegena_pool(max_per_class=70)
    total_real = sum(len(v) for v in real_pool.values())
    print(f"      Termuat {total_real} sampel tulisan tangan nyata.")

    for split, count in (("train", n_train), ("val", n_val)):
        print(f"[2/3] Menghasilkan {count} citra halaman {split} (Nglegena + Pasangan + Sandhangan)...")
        for i in range(count):
            img, labels = synthesize_page(fonts, real_pool, img_size=416)
            img_path = os.path.join(OUT_DIR, "images", split, f"aks_{split}_{i:04d}.jpg")
            lbl_path = os.path.join(OUT_DIR, "labels", split, f"aks_{split}_{i:04d}.txt")
            cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(lbl_path, "w", encoding="utf-8") as f:
                for cid, xc, yc, bw, bh in labels:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUT_DIR.replace(os.sep, '/')}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(ALL_CLASSES)}\n")
        f.write("names:\n")
        for idx, name in enumerate(ALL_CLASSES):
            f.write(f"  {idx}: {name}\n")

    print(f"[3/3] Selesai! Dataset tersimpan di {OUT_DIR} ({len(ALL_CLASSES)} kelas atomik).")


if __name__ == "__main__":
    main()
