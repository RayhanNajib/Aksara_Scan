# Runbook Pelatihan GPU Penuh (Colab / T4 / A100) — AksaraScan v3.0

Mesin pengembangan lokal hanya memiliki CPU (`torch CPU`), sehingga pelatihan penuh multi-epoch resolusi `960`/`1280` dengan `3 seed` dijalankan menggunakan skrip berlisensi **Apache-2.0 / MIT** di repositori ini pada GPU Colab atau server CUDA Anda.

> **Keamanan Lisensi**: Seluruh skrip pelatihan (`training/yolox_model.py`, `training/train_yolox.py`, `training/train_crnn_track_b.py`) bersih dari `ultralytics` (`AGPL-3.0`) dan bebas dari font Windows proprieter.

---

## 1. Persiapan Lingkungan GPU

```bash
git clone https://github.com/RayhanNajib/Aksara_Scan.git
cd Aksara_Scan
pip install -r training/requirements.txt onnx onnxruntime-gpu onnxconverter-common
python tools/datasets/download.py --all-ok
python tools/datasets/convert_local_handwriting.py
python tools/datasets/convert_setilanaji.py
python tools/datasets/dedupe_and_validate.py
python tools/synth_v2/corpus_builder.py
python tools/synth_v2/generate_v2.py --n-train 13500 --n-val 1500 --out-dir dataset/external/synth_v2
```

---

## 2. Jalur Utama (Track A): Detektor YOLOX + P2 Small-Object Head

### Tahap 1 — Pralatih pada Sintetis v2 + Publik (120 epoch)
Membandingkan resolusi `960` vs `1280` serta arsitektur `yolox-nano`, `yolox-tiny` (rekomendasi keseimbangan < 20 MB), dan `yolox-s` (butuh FP16/INT8):

```bash
# Seed 42, 123, 2026 (Fase 3d multi-seed)
for SEED in 42 123 2026; do
  python training/train_yolox.py \
    --variant yolox-tiny \
    --use-p2 \
    --imgsz 960 \
    --stage 1 \
    --epochs 120 \
    --close-mosaic 15 \
    --mosaic 0.25 \
    --batch 16 \
    --lr0 0.001 \
    --seed $SEED \
    --out-dir runs_yolox/stage1_tiny_960_s$SEED
done
```

### Tahap 2 — Fine-Tuning Terarah pada Data Nyata & Hard Negatives (45 epoch, lr kecil)
Menggabungkan `eval/real_train.jsonl` (split writer-disjoint dari `eval/split_real.py`) + `dataset/external/hard_mined` (dari `tools/datasets/hard_negative_mine.py`):

```bash
python training/train_yolox.py \
  --variant yolox-tiny \
  --use-p2 \
  --imgsz 960 \
  --stage 2 \
  --epochs 45 \
  --close-mosaic 10 \
  --mosaic 0.10 \
  --batch 8 \
  --lr0 0.0002 \
  --resume runs_yolox/stage1_tiny_960_s42/yolox-tiny_img960_s42_best.pt \
  --train-sources eval/real_train.jsonl dataset/external/hard_mined dataset/external/converted_yolo/jawa_zip \
  --val-sources eval/real_val.jsonl \
  --seed 42 \
  --out-dir runs_yolox/stage2_finetune
```

### Tahap 3 — Close-Mosaic Polishing & Gerbang Kuantisasi (< 20 MB)
Skrip `train_yolox.py` otomatis melakukan verifikasi paritas PyTorch vs ONNX (`max |Δ| < 1e-4`), dan jika ukuran model > 20 MB akan menjalankan kuantisasi INT8 dengan gerbang penyusutan ukuran & akurasi. Untuk `yolox-s` (37 MB FP32), jalankan juga konversi FP16 (`onnxconverter-common`, ~18.6 MB) lalu verifikasi menggunakan harness evaluasi sebelum merilis.

---

## 3. Jalur Penantang (Track B): CRNN + BiLSTM + CTC Loss (Segmentation-Free)

Digunakan untuk membaca baris tulisan tangan tipis tanpa pemotongan kotak dan menjadi pemeriksa silang di UI:

```bash
python training/train_crnn_track_b.py \
  --epochs 40 \
  --batch 32 \
  --lr 0.001 \
  --out-dir runs_crnn
```

---

## 4. Gerbang Rilis Wajib (`FASE 5b`)

Jangan menimpa `app/web_camscanner/aksara_yolo.onnx` sebelum seluruh kondisi ini terpenuhi:
1. `python eval/run_eval.py --all --onnx runs_yolox/stage2_finetune/yolox-tiny_img960_s42.onnx --out-dir eval/reports/candidate`
2. CER rata-rata pada `eval/real_test.jsonl` turun nyata dibanding `baseline_v2.5`.
3. `recall` kelas kecil (`< 16 px`, terutama `cecak` & `wulu`) meningkat tanpa penurunan > 3 poin pada kelas mana pun.
4. Uji end-to-end Playwright (`node eval/test_e2e_assertions.mjs`) lulus 100% dengan 0 permintaan jaringan eksternal dan latensi < 2000 ms/baris.
