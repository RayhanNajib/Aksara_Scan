"""
Pembagian dataset Hanacaraka ke 3 segmen: train / validation / test.
- Sumber : dataset/hanacaraka/ (1.579 citra, CC0-1.0 oleh vzrenggamani)
- Rasio  : 70% train, 15% validation, 15% test — deterministik per kelas (SEED 42)
- Output : dataset/split/{train,validation,test}/<kelas>/*.png + manifest.csv
- Cara   : python training/split_dataset.py
Jalankan SKALI sebelum training; train_svm_v2.py membaca folder split bila ada.
"""
import csv
import os
import random
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "dataset", "hanacaraka")
DST = os.path.join(ROOT, "dataset", "split")
CLASSES = [
    'ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma',
    'na', 'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya'
]
RATIOS = (("train", 0.70), ("validation", 0.15), ("test", 0.15))
SEED = 42

def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    rows = []
    for cls in CLASSES:
        files = sorted(
            f for f in os.listdir(os.path.join(SRC, cls))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        rng = random.Random(f"{SEED}:{cls}")
        rng.shuffle(files)
        n = len(files)
        n_tr = round(n * RATIOS[0][1])
        n_va = round(n * RATIOS[1][1])
        bounds = {
            "train": files[:n_tr],
            "validation": files[n_tr:n_tr + n_va],
            "test": files[n_tr + n_va:],
        }
        for split, names in bounds.items():
            out = os.path.join(DST, split, cls)
            os.makedirs(out, exist_ok=True)
            for f in names:
                shutil.copy2(os.path.join(SRC, cls, f), os.path.join(out, f))
                rows.append([f"{split}/{cls}/{f}", cls, split,
                             "CC0-1.0 github.com/vzrenggamani/aksarajawa-hanacaraka"])
    man = os.path.join(ROOT, "dataset", "manifest.csv")
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "class", "split", "source_license"])
        w.writerows(rows)
    per = {s: sum(1 for r in rows if r[2] == s) for s in bounds}
    print("Split selesai:", per, "| total", len(rows))
    print("Manifest ->", man)

if __name__ == "__main__":
    main()
