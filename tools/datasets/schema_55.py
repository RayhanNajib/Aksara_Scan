"""
Skema Kanonik 55 Kelas AksaraScan + Utilitas Validasi Kotak & Hashing Perseptual.
"""
import csv
import os
import cv2
import numpy as np

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

CLASSES_55 = [n for n, _ in NGLEGENA] + [n for n, _ in PASANGAN] + [n for n, _ in SANDHANGAN] + [n for n, _ in PADA]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASSES_55)}
CATEGORIES_55 = (
    ["nglegena"] * 20
    + ["pasangan"] * 20
    + ["sandhangan"] * 12
    + ["pada"] * 3
)

# Pemetaan alias eksplisit dari variasi penamaan dataset publik ke 55 kelas kita
EXPLICIT_ALIASES = {
    **{n: n for n in CLASSES_55},
    "nya_": "nya",
    "nga_": "nga",
    "pasangan_ha": "p_ha", "pasangan_na": "p_na", "pasangan_ca": "p_ca", "pasangan_ra": "p_ra",
    "pasangan_ka": "p_ka", "pasangan_da": "p_da", "pasangan_ta": "p_ta", "pasangan_sa": "p_sa",
    "pasangan_wa": "p_wa", "pasangan_la": "p_la", "pasangan_pa": "p_pa", "pasangan_dha": "p_dha",
    "pasangan_ja": "p_ja", "pasangan_ya": "p_ya", "pasangan_nya": "p_nya", "pasangan_ma": "p_ma",
    "pasangan_ga": "p_ga", "pasangan_ba": "p_ba", "pasangan_tha": "p_tha", "pasangan_nga": "p_nga",
    "adeg-adeg": "pada_adeg", "adeg_adeg": "pada_adeg",
    "pada lingsa": "pada_lingsa", "lingsa": "pada_lingsa",
    "pada lungsi": "pada_lungsi", "lungsi": "pada_lungsi",
    "cakrawa": "cakra",
    "pangkat": None,  # Dibuang secara eksplisit (belum masuk 55 kelas)
}


def compute_dhash64(img_bgr_or_gray):
    """64-bit difference hash (tahan perubahan ukuran & pencahayaan ringan) untuk deduplikasi."""
    if img_bgr_or_gray is None:
        return None
    gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY) if img_bgr_or_gray.ndim == 3 else img_bgr_or_gray
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for b in diff.ravel():
        val = (val << 1) | (1 if b else 0)
    return f"{val:016x}"


def compute_tight_foreground_yolo_box(gray, pad=4):
    """Hitung bbox tinta Otsu yang dinormalisasi ke [xc, yc, w, h] dalam (0..1]."""
    h, w = gray.shape[:2]
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if len(ys) < 10:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0 = max(0, int(xs.min()) - pad)
        y0 = max(0, int(ys.min()) - pad)
        x1 = min(w, int(xs.max()) + 1 + pad)
        y1 = min(h, int(ys.max()) + 1 + pad)
    xc = max(0.0, min(1.0, ((x0 + x1) / 2.0) / w))
    yc = max(0.0, min(1.0, ((y0 + y1) / 2.0) / h))
    bw_n = max(1e-4, min(1.0, (x1 - x0) / w))
    bh_n = max(1e-4, min(1.0, (y1 - y0) / h))
    return xc, yc, bw_n, bh_n


def validate_yolo_box(cid, xc, yc, bw, bh):
    if not (0 <= cid < len(CLASSES_55)):
        return False, f"class_id {cid} di luar 0..{len(CLASSES_55)-1}"
    if bw <= 0 or bh <= 0:
        return False, f"lebar/tinggi <= 0 ({bw}, {bh})"
    x0, y0 = xc - bw / 2.0, yc - bh / 2.0
    x1, y1 = xc + bw / 2.0, yc + bh / 2.0
    if x0 < -1e-4 or y0 < -1e-4 or x1 > 1.0001 or y1 > 1.0001:
        return False, f"kotak keluar gambar: [{x0:.4f}, {y0:.4f}, {x1:.4f}, {y1:.4f}]"
    return True, ""


def write_yolo_data_yaml(out_dir, train_rel="images/train", val_rel="images/val"):
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.replace(os.sep, '/')}\n")
        f.write(f"train: {train_rel}\n")
        f.write(f"val: {val_rel}\n")
        f.write(f"nc: {len(CLASSES_55)}\n")
        f.write("names:\n")
        for i, name in enumerate(CLASSES_55):
            f.write(f"  {i}: {name}\n")
    return yaml_path
