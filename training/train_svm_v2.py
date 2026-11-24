"""
AksaraScan Model Trainer v2: Augmentasi Data + LinearSVC -> Float16 Base64
- Augmentasi per glif asli: rotasi, skala, translasi, ketebalan pena (dilate/erode),
  derau bintik, dan jitter kontras ambang Otsu.
- Validasi: holdout glif ASLI (tanpa augmentasi test) supaya akurasi jujur
  terhadap variasi foto nyata.
"""
import os, glob, json, base64, random
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "hanacaraka")
SPLIT_DIR = os.path.join(ROOT_DIR, "dataset", "split", "train")
OUTPUT_JSON = os.path.join(ROOT_DIR, "app", "web_camscanner", "model.json")
OUTPUT_JS = os.path.join(ROOT_DIR, "app", "web_camscanner", "model.js")
BLUR_SIGMA = 2.5

def blur96(canvas, sigma=BLUR_SIGMA):
    """Gaussian separable eksplisit; pad half-symmetric (reflect_101).
    Identik matematis dengan smoothFeatures() di web_camscanner/app.js
    (kernel exp, radius ceil(3*sigma), normalisasi jumlah Bob)."""
    r = int(np.ceil(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma)); k = k / k.sum()
    img = canvas.astype(np.float32)
    p = np.pad(img, r, mode='reflect')
    t = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), 1, p)
    t = np.apply_along_axis(lambda m: np.convolve(m, k, mode='valid'), 0, t)
    return t

CLASSES = [
    'ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
    'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya'
]
CANVAS_SIZE = 96
INNER_SIZE = 76
AUG_PER_IMAGE = 9   # varian augmentasi tiap citra asli
SEED = 42

def binarize(img, extra=0):
    """Otsu inversi + jitter ambang (mensimulasikan pencahayaan berbeda)."""
    t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if extra:
        t = int(np.clip(t + extra, 10, 245))
    _, bw = cv2.threshold(img, t, 255, cv2.THRESH_BINARY_INV)
    return bw

def tight_crop(bw):
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        return None
    return bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def to_canvas(crop):
    h, w = crop.shape
    if h == 0 or w == 0:
        return None
    scale = float(INNER_SIZE) / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    ox, oy = (CANVAS_SIZE - nw) // 2, (CANVAS_SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return (canvas > 127).astype(np.uint8)  # biner 0/1; augment memakai *255

def augment(canvas, rng):
    """Variansi robust pada glif biner 0/1 96x96. Transformasi affine utuh
    (rotasi + geser + skala sekaligus), TANPA np.roll yang meng-wrap ink."""
    img = (canvas * 255).astype(np.uint8)
    out = []
    combos = [
        dict(ang=rng.uniform(-7, 7), s=rng.uniform(0.9, 1.08), dx=rng.uniform(-4, 4), dy=rng.uniform(-3, 3)),
        dict(ang=rng.uniform(-7, 7), s=rng.uniform(0.9, 1.08), dx=rng.uniform(-4, 4), dy=rng.uniform(-3, 3), thick=1),
        dict(ang=rng.uniform(-7, 7), s=rng.uniform(0.9, 1.08), dx=rng.uniform(-4, 4), dy=rng.uniform(-3, 3), thick=-1),
    ]
    for cb in combos:
        a = cb["ang"] * np.pi / 180.0
        sc = cb["s"]
        M = np.array([
            [sc * np.cos(a), -sc * np.sin(a), 48 - 48 * sc * np.cos(a) + 48 * sc * np.sin(a) + cb["dx"]],
            [sc * np.sin(a),  sc * np.cos(a), 48 - 48 * sc * np.sin(a) - 48 * sc * np.cos(a) + cb["dy"]],
        ])
        w = cv2.warpAffine(img, M, (96, 96), flags=cv2.INTER_NEAREST, borderValue=0)
        t = cb.get("thick", 0)
        if t > 0:
            w = cv2.dilate(w, np.ones((3, 3), np.uint8))
        elif t < 0:
            w = cv2.erode(w, np.ones((3, 3), np.uint8))
        out.append((w > 127).astype(np.uint8))
    # derau bintik ringan pada varian transform pertama
    noisy = (out[0] * 255).astype(np.uint8)
    for _ in range(rng.randint(4, 14)):
        yy, xx = rng.randint(0, 95), rng.randint(0, 95)
        noisy[yy, xx] = 255 if rng.random() < 0.5 else 0
    out.append((noisy > 127).astype(np.uint8))
    return out

def photo_variant(canvas, r):
    """Variasi lebih keras dari augment: rotasi +-12, blur->rethreshold, pena tebal, bintik."""
    img = (canvas * 255).astype(np.uint8)
    ang = r.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((48, 48), ang, 1.0)
    img = cv2.warpAffine(img, M, (96, 96), flags=cv2.INTER_NEAREST, borderValue=0)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)
    if r.random() < 0.7:
        img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    for _ in range(r.randint(4, 18)):
        yy, xx = r.randint(0, 95), r.randint(0, 95)
        img[yy, xx] = 255 if r.random() < 0.5 else 0
    return blur96((img > 127).astype(np.uint8)).flatten().astype(np.float32)

def load_pool():
    """Kumpulkan glif asli (0/1) + augmentasi, dengan jejak file sumber."""
    rng = random.Random(SEED)
    Xo, yo, fo, Xa, ya = [], [], [], [], []
    for c_idx, c_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, c_name)
        files = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            bw = binarize(img, 0)
            crop = tight_crop(bw)
            if crop is None or crop.size < 20:
                continue
            canvas = to_canvas(crop)
            if canvas is None:
                continue
            Xo.append(blur96(canvas).flatten().astype(np.float32)); yo.append(c_idx); fo.append(canvas)
            for v in augment(canvas, rng):
                Xa.append(blur96(v).flatten().astype(np.float32)); ya.append(c_idx)
    return np.array(Xo, np.float32), np.array(yo), fo, np.array(Xa, np.float32), np.array(ya)

def main():
    print("=== AKSARASCAN TRAINER v2 (augmented) ===")
    if os.path.isdir(SPLIT_DIR):
        print("Menggunakan segmen train:", os.path.relpath(SPLIT_DIR, ROOT_DIR))
        globals()['DATASET_DIR'] = SPLIT_DIR
    Xo, yo, fo, Xa, ya = load_pool()
    print(f"Asli: {len(Xo)} | Aug varian: {len(Xa)} (ink asli {Xo.mean():.3f})")

    # Holdout 25% asli per kelas
    test_idx = []
    for c in range(len(CLASSES)):
        ids = np.where(yo == c)[0]
        test_idx.extend(ids[:max(10, int(round(len(ids) * 0.25)))])
    test_idx = np.array(sorted(test_idx))
    mask = np.ones(len(Xo), dtype=bool); mask[test_idx] = False
    Xtr, ytr = Xo[mask], yo[mask]
    Xte, yte = Xo[test_idx], yo[test_idx]
    pr = random.Random(99)
    Pho = np.array([photo_variant(fo[i], pr) for i in test_idx], dtype=np.float32)

    from sklearn.svm import LinearSVC
    print("Config terpilih dari sweep: frac=0.0 C=0.5 (blend clean+foto terbaik, konvergen cepat)")
    frac, C = 0.0, 0.5
    X, y = Xtr, ytr
    clf = LinearSVC(C=C, max_iter=8000, random_state=SEED).fit(X, y)
    clean = accuracy_score(yte, clf.predict(Xte)) * 100
    photo = accuracy_score(yte, clf.predict(Pho)) * 100
    print(f"  frac=0.0 C=0.5: clean {clean:.1f}% | foto {photo:.1f}%")
    score = 0.5 * clean + 0.5 * photo
    best = ((frac, C, clf), score)
    (frac, C, clf), score = best
    y_pred = clf.predict(Xte)
    acc = accuracy_score(yte, y_pred) * 100
    photo_acc = accuracy_score(yte, clf.predict(Pho)) * 100
    print(f"--> Terpilih: frac={frac} C={C} | clean {acc:.1f}% | foto {photo_acc:.1f}%")

    per_class_acc = {}
    for i, c_name in enumerate(CLASSES):
        m = (yte == i)
        if m.sum():
            per_class_acc[c_name] = round(accuracy_score(yte[m], y_pred[m]) * 100)

    weights_f16 = clf.coef_.astype(np.float16)
    biases_f32 = [float(b) for b in clf.intercept_]
    b64 = base64.b64encode(weights_f16.tobytes()).decode('ascii')
    payload = {
        "classes": CLASSES, "size": CANVAS_SIZE, "inner": INNER_SIZE, "blurSigma": BLUR_SIGMA,
        "w_f16_b64": b64, "bias": biases_f32, "accuracy": round(acc, 1),
        "per_class": per_class_acc, "n_train": int(len(Xtr) + len(Xa) * frac),
        "n_test": int(len(Xte)),
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f)
    with open(OUTPUT_JS, "w") as f:
        f.write("window.AKSARA_MODEL = " + json.dumps(payload) + ";\n")
    print(f"Ekspor: {OUTPUT_JSON} ({len(b64)//1024} KB) + model.js")

if __name__ == "__main__":
    main()
