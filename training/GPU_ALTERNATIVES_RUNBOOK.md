# Panduan Alternatif GPU Notebook Gratis & GPU Lokal (GTX 1650 4GB) — AksaraScan v4.0

Di lingkungan Python Windows bawaan (`torch +cpu`), **Stage-2 Metric Embedder** dilatih langsung di CPU (`0.77 MB`), sedangkan pelatihan penuh **AksaraYOLOX-P2 (Apache-2.0)** (`640px` / `960px`, `--amp`) dapat dijalankan di:
1. **GPU Cloud Gratis** (**Kaggle `T4 x2`** / **Lightning AI `T4/L4`**) via `training/AksaraScan_v4_GPU_MultiPlatform.ipynb`, atau
2. **GPU Laptop Lokal (`NVIDIA GeForce GTX 1650 4 GB VRAM`)** memakai mode `--batch 8 --grad-accum 2 --amp` (**syarat disk Windows: kosongkan minimal `15 GB` di `C:`** sebelum memasang wheel PyTorch CUDA `~2.53 GB` terkompresi / `~5.5 GB` terekstrak).

> **Keamanan Kebocoran Data (Strict Zero Leakage):**
> Skrip `tools/build_gpu_bundle.py` hanya membungkus split `train` & `val` (`~189.5 MB` tidak terkompresi / `< 180 MB` zip). Data `test_heldout` (`hanacaraka`), `eval/heldout_real/`, dan `nusaaksara_eval_only_v4` **sengaja tidak dimasukkan** ke bundel GPU.

---

## Ringkasan Pilihan Platform Pengganti Google Colab

| Platform | GPU Gratis & Kuota | Kenapa Cocok untuk AksaraScan v4 | Cara Upload Data/Notebook |
|---|---|---|---|
| **1. Kaggle Notebooks** *(Rekomendasi Utama)* | `Tesla T4 x2 (30 GB)` atau `Tesla P100 (16 GB)` · **30 jam/minggu gratis** | Jarang putus mendadak, mendukung background *Save & Run All (Commit)* sehingga browser boleh ditutup! | Upload `training/AksaraScan_v4_GPU_MultiPlatform.ipynb` + Dataset `.zip` (`aksarascan_v4_gpu_bundle.zip`) |
| **2. Lightning AI Studios** *(Paling Mirip VS Code)* | `T4 / L4 GPU` · **~15–22 jam GPU/bulan gratis** + storage persisten | Tidak ada cell disconnect drama seperti Colab; Anda dapat terminal Linux & VS Code langsung di browser | Upload `.zip` via UI file explorer atau `git clone`, lalu jalankan perintah bash langsung |
| **3. Modal / Paperspace / HuggingFace Spaces** | Modal: kredit gratis `$30/bulan`; Saturn/Paperspace: GPU tier gratis | Eksekusi container bersih tanpa antrean notebook browser | Mount repo + `aksarascan_v4_gpu_bundle.zip` |
| **4. Google Colab** *(Fallback)* | `T4 15 GB` (dinamis) | Masih kompatibel memakai `.ipynb` yang sama jika Colab kembali normal | Upload `.ipynb` + `.zip` |

---

## Langkah Cepat (Berlaku untuk Semua Platform)

### Langkah 1 — Buat Bundel Training Lokal (`< 180 MB`, bebas `test_heldout`)
Jalankan di PC lokal dari folder `Aksara_Scan/`:
```bash
python tools/build_gpu_bundle.py --out dist_gpu/aksarascan_v4_gpu_bundle.zip
```
Hasilnya satu file `dist_gpu/aksarascan_v4_gpu_bundle.zip` berisi seluruh skrip pelatihan (`training/`), skema (`dataset/schema/`, `tools/datasets/schema_*.py`), serta seluruh gambar `train` + `val` yang sudah lolos audit zero-leakage Fase 1.

---

### Langkah 2A — Cara Pakai di Kaggle Notebooks (Tanpa Colab)
1. Buka **https://www.kaggle.com/datasets** -> **+ New Dataset** -> Upload `dist_gpu/aksarascan_v4_gpu_bundle.zip` -> Beri nama `aksarascan-v4-bundle` (**Private**).
2. Buka **https://www.kaggle.com/code** -> **+ New Notebook** -> **File -> Import Notebook** -> Pilih `training/AksaraScan_v4_GPU_MultiPlatform.ipynb`.
3. Di sidebar kanan **Notebook Options**:
   - **Accelerator**: pilih **GPU T4 x2** atau **GPU P100**
   - **Internet**: **ON** (hanya jika `onnx` belum terpasang; PyTorch & OpenCV sudah terpasang di Kaggle)
   - **Input**: **+ Add Input** -> pilih dataset `aksarascan-v4-bundle` milik Anda
4. Klik **Save Version -> Save & Run All (Commit)**. Anda boleh menutup tab browser — setelah `~45–90 menit`, buka tab **Output** dan unduh `aksarascan_v4_trained_onnx.zip`.

**Alternatif CLI Headless Kaggle (Tanpa Buka Editor Browser):**
```bash
pip install kaggle
# Simpan API token ke ~/.kaggle/kaggle.json (atau C:\Users\<user>\.kaggle\kaggle.json)
kaggle datasets create -p dist_gpu/  # dengan dataset-metadata.json
kaggle kernels push -p training/
```

---

### Langkah 2B — Cara Pakai di Lightning AI Studios (Terminal + VS Code Browser)
1. Login ke **https://lightning.ai** (bisa pakai akun GitHub).
2. Buat Studio baru (**PyTorch / T4 GPU**).
3. Di terminal Studio:
   ```bash
   # Unggah aksarascan_v4_gpu_bundle.zip ke workspace via drag-and-drop UI
   mkdir -p Aksara_Scan && cd Aksara_Scan
   unzip -q ../aksarascan_v4_gpu_bundle.zip
   pip install -q onnx onnxruntime-gpu PyYAML scipy
   ```
4. Jalankan pelatihan Stage-1 (`640px`, rekomendasi kecepatan WASM browser `~0.8–1.2 dtk`) & Stage-2 fine-tune:
   ```bash
   # Tahap 1: Pra-latih multi-sumber v4 (synth_v4 + hanacaraka + jawa_zip + syllables + teacher + hard negatives)
   python -u training/train_yolox.py \
     --variant yolox-tiny \
     --imgsz 640 \
     --stage 1 \
     --epochs 80 \
     --close-mosaic 15 \
     --batch 16 \
     --lr0 0.001 \
     --mosaic 0.25 \
     --seed 42 \
     --amp \
     --export-fp16 \
     --export-int8 \
     --out-dir runs_yolox/v4_stage1_tiny640

   # Tahap 2: Fine-tune nyata ber-LR rendah (fokus tulisan tangan, syllables, teacher, negatif)
   python -u training/train_yolox.py \
     --variant yolox-tiny \
     --imgsz 640 \
     --stage 2 \
     --epochs 40 \
     --close-mosaic 10 \
     --batch 12 \
     --lr0 0.00025 \
     --mosaic 0.10 \
     --seed 42 \
     --amp \
     --export-fp16 \
     --export-int8 \
     --resume runs_yolox/v4_stage1_tiny640/yolox-tiny_img640_s42_best.pt \
     --out-dir runs_yolox/v4_stage2_tiny640
   ```
   Untuk varian resolusi tinggi (`960px`, recall sandhangan ekstra kecil), jalankan juga dengan `--imgsz 960 --batch 8`.

---

### Langkah 2C — Opsi GPU Lokal (`NVIDIA GeForce GTX 1650 4 GB VRAM`, Driver >= 580)
> **Peringatan Ruang Disk:** Instalasi `torch` CUDA (`cu124`) di Windows membutuhkan `~2.53 GB` unduhan + `~5.5 GB` ruang terpasang (`~8 GB` total selama unpack `pip`). Pastikan drive `C:` memiliki **>= 15 GB ruang kosong** sebelum menjalankan langkah ini.

```bash
# 1. Pasang PyTorch CUDA 12.4 (hanya setelah memastikan ruang kosong C: >= 15 GB)
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# 2. Latih Stage-1 YOLOX-Tiny P2 aman untuk VRAM 4 GB (--batch 8 --grad-accum 2 = effective batch 16)
python -u training/train_yolox.py \
  --variant yolox-tiny --imgsz 640 --stage 1 \
  --epochs 80 --close-mosaic 15 \
  --batch 8 --grad-accum 2 --lr0 0.001 --mosaic 0.25 \
  --seed 42 --amp --export-fp16 \
  --out-dir runs_yolox/v4_stage1_tiny640

# 3. Fine-tune Stage-2 nyata ber-LR rendah
python -u training/train_yolox.py \
  --variant yolox-tiny --imgsz 640 --stage 2 \
  --epochs 40 --close-mosaic 10 \
  --batch 6 --grad-accum 2 --lr0 0.00025 --mosaic 0.10 \
  --seed 42 --amp --export-fp16 \
  --resume runs_yolox/v4_stage1_tiny640/yolox-tiny_img640_s42_best.pt \
  --out-dir runs_yolox/v4_stage2_tiny640
```

---

### Langkah 3 — Validasi & Pasang Model Drop-In (`install_yolox_release.py`)
Setelah memperoleh `.onnx` (`*_fp16.onnx`, `< 12 MB`) dari GPU cloud maupun pelatihan lokal GTX 1650:
1. Validasi I/O ONNX dan pasang atomik ke aplikasi Web (otomatis membuat backup di `models/backups/` serta memperbarui `imgsz`, `detector_arch`, `license="Apache-2.0"`, dan `sha256` di `app/web_camscanner/yolo_meta.json`):
   ```bash
   python tools/install_yolox_release.py --onnx runs_yolox/v4_stage2_tiny640/yolox-tiny_img640_s42_fp16.onnx
   ```
2. Naikkan versi cache di `app/web_camscanner/sw.js`, lalu jalankan evaluasi penuh & gerbang pra-rilis:
   ```bash
   python eval/eval_phase2_ablation.py --detector app/web_camscanner/aksara_yolo.onnx
   node eval/eval_phase5_browser.mjs && python -u eval/eval_phase5_analyze.py && node eval/eval_phase5_prerelease.mjs
   ```
3. Pastikan ukuran ONNX Stage-1 `< 20 MB`, lolos `node eval/test_e2e_assertions.mjs`, `node eval/test_fase3_enhancer.mjs`, dan `python tools/check_leakage.py`!
