"""
AksaraScan Model Trainer: LinearSVC ke Float16 Base64 JSON & JS
Mata Kuliah: PTIN236021 Pengolahan Citra dan Komputer Vision (Semester V)
Dosen Pengampu: Mahfud Jiono, S.Pd., M.Eng.
Penyusun: Muhammad Rayhan Najib (240533600012) - Kelompok 4
"""
import os, glob, json, base64
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "hanacaraka")
OUTPUT_JSON = os.path.join(ROOT_DIR, "app", "web_camscanner", "model.json")
OUTPUT_JS = os.path.join(ROOT_DIR, "app", "web_camscanner", "model.js")

CLASSES = [
    'ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
    'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya'
]
CANVAS_SIZE = 96
INNER_SIZE = 76

def preprocess_glyph(img_path):
    """Normalisasi identik dengan browser: Otsu -> Crop -> Scale 76px -> Canvas 96x96"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Binerisasi Otsu Inversi
    _, bw = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Crop rapat ke kotak pembatas tinta
    ys, xs = np.where(bw > 0)
    if len(ys) == 0: return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = bw[y0:y1, x0:x1]
    
    # Skala isotropik ke inner 76px
    h, w = crop.shape
    scale = float(INNER_SIZE) / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    
    # Tengahkan pada kanvas 96x96
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    ox = (CANVAS_SIZE - nw) // 2
    oy = (CANVAS_SIZE - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = resized
    
    return canvas.flatten() # 9.216 dimensi

def main():
    print("=" * 60)
    print("AKSARASCAN SVM MODEL TRAINER & EXPORTER")
    print("=" * 60)
    print("[1/4] Memuat dan memproses dataset...")
    X, y = [], []
    for c_idx, c_name in enumerate(CLASSES):
        folder = os.path.join(DATASET_DIR, c_name)
        files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
        for f in files:
            feat = preprocess_glyph(f)
            if feat is not None:
                X.append(feat)
                y.append(c_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Total sampel valid: {len(X)} dari 20 kelas.")

    # Stratified Holdout Split (75% Train, 25% Test)
    print("[2/4] Melakukan Stratified Holdout Split (75 : 25)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.247468, random_state=42, stratify=y
    )
    print(f"Partisi Data: {len(X_train)} Train, {len(X_test)} Test Holdout.")

    # Pelatihan Linear SVM One-vs-Rest
    print("[3/4] Melatih LinearSVC (C=1.0, max_iter=2500)...")
    clf = LinearSVC(C=1.0, max_iter=2500, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100.0
    print(f"--> Akurasi Holdout Evaluasi: {acc:.1f}%")

    per_class_acc = {}
    for i, c_name in enumerate(CLASSES):
        mask = (y_test == i)
        if np.sum(mask) > 0:
            c_acc = round(accuracy_score(y_test[mask], y_pred[mask]) * 100.0)
            per_class_acc[c_name] = c_acc

    # Ekspor Bobot ke Float16 Base64 JSON & JS
    print("[4/4] Mengekspor model terkompresi float16...")
    weights_f16 = clf.coef_.astype(np.float16) # (20, 9216) -> 184.320 float16
    biases_f32 = [float(b) for b in clf.intercept_]
    b64_weights = base64.b64encode(weights_f16.tobytes()).decode('ascii')

    model_payload = {
        "classes": CLASSES,
        "size": CANVAS_SIZE,
        "inner": INNER_SIZE,
        "w_f16_b64": b64_weights,
        "bias": biases_f32,
        "accuracy": round(acc, 1),
        "per_class": per_class_acc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(model_payload, f)

    with open(OUTPUT_JS, "w") as f:
        f.write("window.AKSARA_MODEL = " + json.dumps(model_payload) + ";\n")

    print(f"SUKSES! Berkas model berhasil diekspor:")
    print(f" - JSON: {OUTPUT_JSON} ({len(b64_weights)//1024} KB)")
    print(f" - JS  : {OUTPUT_JS}")

if __name__ == "__main__":
    main()
