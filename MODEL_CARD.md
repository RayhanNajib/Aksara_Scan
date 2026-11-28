# Model Card — AksaraScan v4.0 (Fase 2 Milestone)

## Ringkasan Model & Arsitektur Hibrid
**AksaraScan** adalah aplikasi pemindai dan penerjemah Aksara Jawa **100% di peramban** (*client-side ONNX Runtime Web WASM*) dengan arsitektur dua-tahap dan modul bahasa lokal:

1. **Detektor Tahap 1 (Track A1)**:
   - **Produksi Aktif saat ini (`C0`)**: YOLOv8n ONNX (`app/web_camscanner/aksara_yolo.onnx`, `12.11 MB` FP32, `416px`, `55` kelas).
   - **Detektor Pengganti Bebas AGPL-3.0 siap GPU (`C1`)**: Arsitektur custom **AksaraYOLOX + cabang resolusi tinggi P2 (`stride-4`)** di `training/yolox_model.py` + `training/train_yolox.py` (**Apache-2.0 / MIT**), mendukung manifes `v4`, contoh negatif murni `0-box` (Latin/garis dinding), bobot kurikulum bertahap, serta konversi **FP16 Weight Storage ONNX murni** (`yolox-tiny 640px = 10.79 MB`, `960px = 11.09 MB` — sudah terverifikasi berjalan mulus di Chromium ORT-Web 1.20.1 WASM). Catatan jujur: pelatihan penuh 80+40 epoch membutuhkan GPU via `training/AksaraScan_v4_GPU_MultiPlatform.ipynb` (lokal baru menjalankan *smoke test* CPU 2-epoch).
2. **Stage-2 Reference Prototype Metric Verifier (Track A2 — terlatih penuh di CPU lokal)**:
   - **Embedder (`app/web_camscanner/aksara_embed.onnx`, `0.77 MB` FP32, `64×64`)**: Depthwise-Separable ConvNet + Squeeze-Excite + Spatial Quadrant Mixer dengan **Sub-Center Cosine Margin Head (3 sub-pusat × 56 kelas)**. Berjalan di **1 ms** (batch 1) / **13 ms** (batch 16) di Chromium ORT-Web WASM!
   - **Bank Prototipe Referensi (`app/web_camscanner/reference_prototypes.json`, `251.5 KB`)**: `176` centroid L2-ternormalisasi dari **Spherical K-Means** strictly pada **`6.623` contoh `split="train"` saja** (terbagi per gaya `handwritten`, `printed_tps`, dan `non_aksara` tanpa menyentuh `val`/`heldout_real`), plus miniatur PNG `28×28` untuk *Reference Inspector Drawer*.
   - **Kalibrasi Tersekat pada `val` (`app/web_camscanner/stage2_calibration.json`)**: Menggabungkan skor detektor dengan probabilitas Stage-2 & kemiripan prototipe (`w_det=0.62`, `neg_reject_thr_lo=0.48`, `neg_reject_thr_hi=0.62`), menyaring **58,4%** false positive pada halaman negatif Latin/tembok di split `val`.
3. **Jalur Penantang Baris Segmentation-Free (Track B)**:
   - **CRNN + BiLSTM 2-lapis + CTC Loss** (`runs_crnn/aksara_crnn_ctc.onnx`, `9.86 MB`, `175 ms` p50 WASM). Hasil jujur Fase 2 pada `setilanaji_lines_v4 val`: CER mandiri masih `65.08%`, sehingga gerbang `val` menonaktifkan penggantian otomatis sampai dilatih ulang di GPU.
4. **Rekonstruktor Spasial, Validator Ortografi, & Mesin Bahasa**:
   - `orthography.js` (perbaikan peta vokal `na` vs `da` & pengelompokan baris), `javanese_lexicon.js` + `javanese_lm_engine.js` (`100%` lokal).

---

## Ruang Kelas (`dataset/schema/classes_v4.yaml`)
- **Tier 1 (Aktif — 55 kelas Aksara + 1 kelas Penolak Stage-2 = 56 kelas)**:
  - `20 Nglegena`, `20 Pasangan`, `12 Sandhangan & Pangkon`, `3 Tanda Baca Pada`
  - **Kelas #55 `non_aksara`** (Stage-2 saja): patch teks Latin cetak/tulisan tangan, angka, garis buku tulis, kotak kisi milimeter, retakan/garis tembok, serta `2.889` patch FP hasil tambang detektor di split `train`.
- **Tier 2 & Tier 3**: *Ditunda* (`deferred`) hingga tersedia `>= 100` kotak beranotasi nyata dari `>= 5` penulis independen.

---

## Ringkasan Evaluasi Jujur Fase 2 (`eval/reports/phase2_ablation.json`)
Evaluasi pada **set terkunci `eval/heldout_real/` (`n=42`, `SHA256SUMS.lock` `43` berkas `mismatches=0`, kalibrasi hanya pada `val`)**:

| Konfigurasi | Heldout Real CER (95% CI) | Heldout Grapheme F1 (95% CI) | Box Cls Acc (`n=120`) | Failures (`n=8`) CER / GF1 | Clean Synth v2 (`n=40`) CER / GF1 |
|---|---|---|---|---|---|
| **`C0`: Baseline `v3.1` (`aksara_yolo.onnx`)** | `37.36%` `[30.48, 44.70]` | `53.50%` `[44.75, 61.68]` | `63.33%` | `27.81%` / `85.67%` | `23.16%` / `81.94%` |
| **`C1`: YOLOX-Tiny P2 (smoke test 2-ep CPU)** | `100.00%` | `0.00%` | `0.00%` | `100.00%` / `0.00%` | — *(butuh GPU full epoch)* |
| **`C2`: `C0 + Stage-2 Embedder + Prototype Bank`** | **`34.48%`** `[28.06, 40.99]` | **`55.97%`** `[47.90, 63.60]` | **`64.17%`** | **`20.24%`** / **`86.92%`** | `23.54%` / `81.81%` *(stabil)* |
| **`C3`: `C2 + Track B CRNN`** | `34.48%` *(identik `C2`)* | `55.97%` | `64.17%` | `20.24%` / `86.92%` | `23.54%` / `81.81%` |

---

## Privasi, Lisensi, & Keterbatasan
- **0 permintaan jaringan eksternal** saat inferensi; opsi terjemahan daring baku **MATI**.
- Seluruh font & pustaka pelatihan patuh lisensi (**SIL OFL 1.1 / Apache-2.0 / MIT / CC0 / CC BY 4.0**; `NusaAksara` `CC BY-SA 4.0` digunakan ketat sebagai **HANYA-EVALUASI**).
- **Integrasi Browser Fase 3 (Aktif)**: `aksara_embed.onnx`, `reference_prototypes.json` (`176` prototipe), dan `stage2_calibration.json` sudah berjalan langsung di peramban via `app/web_camscanner/stage2_verifier.js`, bersama praproses adaptif Worker `app/web_camscanner/image_enhancer.js` + `app/web_camscanner/enhancer_worker.js`, aset ONNX Runtime Web lokal `app/web_camscanner/vendor/ort/`, dan cache offline `aksarascan-v14` (`sw.js`, terverifikasi lewat `eval/reports/phase3_verification.json`). Perombakan antarmuka *Reference Inspector Drawer* & UX dijadwalkan pada **Fase 4**.
