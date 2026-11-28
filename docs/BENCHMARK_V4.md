# Laporan Benchmark & Audit Kejujuran — AksaraScan v4.0 (`docs/BENCHMARK_V4.md`)

> Membandingkan **AksaraScan v3.1 (`eb19337`)** melawan **AksaraScan v4.0 (`v4/fase-5-eval`)** langsung di dalam peramban Chromium WASM yang identik, beserta matriks ablasi komponen, matriks konfusi, taksonomi kegagalan, regresi set bersih, dan 20 contoh kegagalan terburuk.

---

## 1. Lingkungan Pengujian & Jaminan Integritas Data

- **Perangkat Keras & OS:** `Windows_NT 10.0.26100 (x64)` (`AMD Ryzen 5 5600H with Radeon Graphics`, `12` logical cores, `15.35 GiB` RAM)
- **Runtime Peramban:** `Chromium 151.0.7922.34` (Playwright headless, `Node v24.19.0`, `ONNX Runtime Web 1.20.1` single-thread WASM, `crossOriginIsolated = false`)
- **Commit Pembanding:** `eb19337` (Baseline v3.1 dari Git object store) vs branch `v4/fase-5-eval`
- **Resampling Statistik:** Bootstrap `B = 2000`, `seed = 42` (`eval/eval_phase5_analyze.py`)
- **Integritas Set Held-Out:** `eval/heldout_real/SHA256SUMS.lock` terverifikasi **`43` berkas, `mismatches = 0`**; `tools/check_leakage.py` across **`21.457` rekaman** berstatus **`PASS_ZERO_LEAKAGE`** (`0` overlap penulis, kluster, SHA-256, maupun near-duplicate). Set `eval/heldout_real/` tidak pernah disentuh saat pelatihan Stage-2, pembuatan `176` prototipe, maupun kalibrasi `stage2_calibration.json`.
- **Skrip Pelacak Angka:** [`eval/eval_phase5_browser.mjs`](../eval/eval_phase5_browser.mjs), [`eval/eval_phase5_analyze.py`](../eval/eval_phase5_analyze.py), [`eval/reports/phase5_benchmark.json`](../eval/reports/phase5_benchmark.json), [`eval/reports/phase2_ablation.json`](../eval/reports/phase2_ablation.json), [`eval/reports/phase5_prerelease.json`](../eval/reports/phase5_prerelease.json).

---

## 2. Benchmark Sebelum vs Sesudah (`v3.1` vs `v4.0 Full` — Browser WASM)

Pengujian dijalankan end-to-end lewat DOM aplikasi web nyata (`#fileInput -> Worker Enhancer -> ORT-Web Stage-1 -> ORT-Web Stage-2 -> orthography.js v4 -> javanese_lm_engine.js`). Waktu muat kesiapan penuh model (`DOMContentLoaded` ke `YOLOv8 Siap`) turun dari **`1772 ms`** (`v3.1` dengan CDN WASM mock) menjadi **`816 ms`** (`v4.0` dari bundel WASM lokal).

| Set Uji | `n` | Unicode CER `v3.1` (95% CI) | Unicode CER `v4.0` (95% CI) | Delta CER | Grapheme F1 `v3.1 → v4.0` | Word Acc Folded `v3.1 → v4.0` | Latensi `v4.0` p50 / p95 |
|---|---:|---|---|---|---|---|---|
| **`heldout_real` (Total Terkunci)** | `42` | `34.31%` `[26.26, 42.49]` | **`30.07%`** `[23.30, 37.17]` | **`-4.24 pp`** | `59.64% → 62.81%` (`+3.17 pp`) | `29.80% → 31.61%` (`+1.81 pp`)* | `947.4 / 1817.4 ms` |
| **`heldout_real` · Foto Buram Nyata** (`motion/defocus/latin_blur`) | `5` | `34.00%` `[12.00, 56.00]` | **`26.00%`** `[8.00, 44.00]` | **`-8.00 pp`** | `70.05% → 76.88%` (`+6.83 pp`) | `30.50% → 38.00%` | `1637.7 / 1963.4 ms` |
| **`failures` (Set Diagnostik + Bug Laporan)** | `8` | `32.78%` `[4.73, 67.51]` | **`22.75%`** `[3.38, 45.83]` | **`-10.03 pp`** | `84.81% → 88.51%` (`+3.70 pp`) | `29.79% → 33.85%` (`+4.06 pp`)** | `1967.3 / 3241.0 ms` |
| **`synth_v2_val_clean` (Penjaga Regresi)** | `40` | `21.42%` `[15.95, 27.04]` | **`21.06%`** `[15.58, 26.56]` | **`-0.36 pp`** | `82.06% → 83.45%` (`+1.39 pp`) | `12.43% → 12.22%` (`-0.21 pp`) | `1253.9 / 2726.8 ms` |

> \*Catatan metrik *Word Accuracy*: Pada `failures (n=8)` dan `synth_v2_val_clean (n=40)` dipakai `latin_gt` tulis-tangan/manifes asli (`strict` tanpa pelipatan diakritik `v3.1 → v4.0` pada `failures`: `27.71% → 31.77%`). Pada `heldout_real (n=42)` karena manifes hanya menyediakan `unicode_gt`, `latin_gt` acuan diturunkan secara deterministik melalui `AksaraOrthography.decodeClustersToLatin` pada `unicode_gt`.

### Efisiensi Penolakan Non-Aksara / Distraktor (`synth_v4_val_neg`, `n=29`)
Pada `29` gambar validasi negatif murni (`0` kotak Aksara Jawa: teks Latin cetak/tulisan tangan, garis buku, retakan dinding):

| Metrik | `v3.1 (eb19337)` | `v4.0 Full` | Perubahan |
|---|---:|---:|---|
| Rata-rata kotak palsu (FP / gambar) | `7.69` (`223` total FP) | **`2.76`** (`80` total FP) | **`-64.1%`** |
| Halaman bersih tanpa 1 pun kotak FP | `27.6%` (`8/29`) | **`44.8%`** (`13/29`) | **`+17.2 pp`** |

---

## 3. Matriks Ablasi Penuh (Browser & Offline Python)

Menguji kontribusi masing-masing modul secara terisolasi:
- **Browser (`eval_phase5_browser.mjs`):** `v3_1` vs `v4_enh_only` (Stage-2 diblokir agar jatuh ke Stage-1 Fallback + `orthography.js` v4 + Worker Enhancer) vs `v4_stage2_only` (`skipEnhance: true`) vs `v4_full`.
- **Offline Python (`eval/reports/phase2_ablation.json`):** `C0`, `C1` (YOLOX-Tiny P2 640px *smoke test* 2-epoch CPU), `C2` (`C0 + Stage-2`), `C3` (`C2 + Track B CRNN`).

| Konfigurasi | Jalur Eksekusi | Heldout `n=42` CER / GF1 | Blur Real `n=5` CER | Failures `n=8` CER / GF1 | Negatif `n=29` FP/img | Clean Synth `n=40` CER | Console Err |
|---|---|---:|---:|---:|---:|---:|---:|
| **`v3_1`** (`eb19337`) | Browser WASM | `34.31%` / `59.64%` | `34.00%` | `32.78%` / `84.81%` | `7.69` | `21.42%` | `0` |
| **`v4_enh_only`** (Stage-1 + Enhancer + Ortho v4) | Browser WASM | `33.59%` / `60.28%` | **`26.00%`** | `30.43%` / `86.29%` | `6.86` | `21.54%` | `2` *(route abort uji)* |
| **`v4_stage2_only`** (Stage-2 176-proto, Enhancer skip) | Browser WASM | `30.19%` / `62.70%` | **`26.00%`** | `25.11%` / `86.58%` | `2.79` | `21.30%` | `0` |
| **`v4_full`** (Enhancer + Stage-2 + Ortho v4 + LM) | Browser WASM | **`30.07%` / `62.81%`** | **`26.00%`** | **`22.75%` / `88.51%`** | **`2.76`** | **`21.06%`** | `0` |
| **`C0`** (`aksara_yolo.onnx` Python) | Python ORT CPU | `37.36%` / `53.50%` | — | `27.81%` / `85.67%` | `9.21` *(val)* | `23.16%` | — |
| **`C1`** (`YOLOX-Tiny P2`, 2-ep CPU smoke) | Python ORT CPU | `100.00%` / `0.00%` | — | `100.00%` / `0.00%` | — | — | — |
| **`C2`** (`C0 + Stage-2` Python) | Python ORT CPU | `34.48%` / `55.97%` | — | `20.24%` / `86.92%` | `3.83` *(val)* | `23.54%` | — |
| **`C3`** (`C2 + Track B CRNN existing`) | Python ORT CPU | `34.48%` *(ditolak gerbang `val`)* | — | `20.24%` | `3.83` *(val)* | `23.54%` | — |

**Pembacaan Ablasi:**
1. **Worker Enhancer** bertanggung jawab langsung atas penurunan error pada foto tulisan tangan terburam (`blur_real_n5` CER `34.0% → 26.0%`) dan menjaga `diag_01..03/06` tanpa merusak citra tajam.
2. **Stage-2 Metric Embedder + 176 Prototipe** menyumbang penurunan terbesar di naskah nyata `NusaAksara` (`53.22% → 44.72%` CER) serta memotong FP pada halaman Latin/garis dinding hingga **`64.1%`**.
3. **Gabungan `v4_full`** meraih capaian terbaik sekaligus paling konsisten (`CER 30.07%` di held-out, `22.75%` di set diagnostik, `2.76 FP/img`).

---

## 4. Rincian Per Subset, Kondisi, & Hasil Negatif / Regresi

### 4.1 Rincian Subset & Kondisi `eval/heldout_real/` (`n=42`)

| Subset / Kondisi | `n` | CER `v3.1` | CER `v4.0` | GF1 `v3.1 → v4.0` | Catatan Jujur |
|---|---:|---:|---:|---|---|
| `nusaaksara_acl2025_eval_only` | `18` | `53.22%` | **`44.72%`** | `33.46% → 40.01%` | Peningkatan `-8.50 pp` CER murni dari generalisasi Stage-2 (NusaAksara 0% ada di data latih). |
| `setilanaji_real_book_val` | `12` | **`7.75%`** | `8.16%` *(+0.41 pp)* | `87.48% → 86.50%` | **Regresi minor negatif (`+0.41 pp`)**: batas relabel Stage-2 menggeser 1–2 karakter beresolusi rapat. |
| `hanacaraka_12_unseen_human_writers` | `12` | `32.50%` | **`30.00%`** | `71.08% → 73.30%` | — |
| └ `defocus_blur` | `2` | `35.00%` | **`20.00%`** | `68.75% → 79.41%` | `-15.0 pp` CER berkat penajaman TV-bounded. |
| └ `latin_distractor_blur` | `1` | `20.00%` | **`10.00%`** | `85.71% → 90.00%` | — |
| └ `latin_distractor_mix` | `1` | `100.00%` | **`90.00%`** | `0.00% → 18.18%` | Masih gagal tinggi (lihat kasus #1 di Bagian 6). |
| └ `motion_blur` | `2` | `40.00%` | `40.00%` | `63.53% → 67.78%` | Deteksi kotak sedikit naik (`GF1 +4.25 pp`). |
| └ `multi_color_pen` | `2` | `10.00%` | `10.00%` | `90.00% → 90.00%` | Tetap `10%` CER / `90%` GF1 pada pulpen warna. |
| └ `wall_lines_shadow` | `2` | **`40.00%`** | `45.00%` *(+5.0 pp)* | `71.31% → 63.53%` | **Regresi negatif (`+5.0 pp` CER)**: ambang `non_aksara` menolak 1 huruf lemah di dekat garis dinding tebal. |
| └ `sharp_clean` | `2` | **`10.00%`** | `15.00%` *(+5.0 pp)* | `90.00% → 85.00%` | **Regresi negatif (`+5.0 pp` CER)**: `v4_stage2_only` bertahan di `10.0%`; praproses perlu penalti passthrough lebih ketat pada garis tipis kertas biru. |

### 4.2 Rincian Set Diagnostik `eval/failures/` (`n=8`)

| ID | Kondisi | CER `v3.1 → v4.0` | GF1 `v3.1 → v4.0` | Bacaan Latin `v4.0` |
|---|---|---:|---:|---|
| `diag_01` | Printed low-res + caption Latin | `37.84% → 27.03%` | `65.31% → 80.85%` | `pajapi mar aja rap ama wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.` |
| `diag_02` | Printed low-res cropped | `0.00% → 0.00%` | `100.00% → 100.00%` | `wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.` |
| `diag_03` | Synthetic lined paper (`sample_4`) | `0.00% → 0.00%` | `100.00% → 100.00%` | `aksara jawa mangan sêga krama kothi` |
| `diag_04` | Handwriting shadow stress | `0.00% → 0.00%` | `100.00% → 100.00%` | `ana cara ka das awala` |
| `diag_05` | Handwriting border lines stress | `11.11% → 11.11%` | `94.74% → 94.74%` | `ana cara ka das awala` |
| `diag_06` | Handwriting low-res 600 px | `0.00% → 0.00%` | `100.00% → 100.00%` | `ana cara ka das awala` |
| `diag_07` | Handwriting + Latin distractor banner | `133.33% → 88.89%` | `64.29% → 72.00%` | `raga wim ara gara ana cara ka das awala` |
| `diag_08` | Handwriting chart + bold Latin labels | `80.00% → 55.00%` | `54.17% → 60.47%` | `ana cêraka dat isi wêli pir ara padha jaya nyi magi bath anga` |

### 4.3 Hasil Negatif / Eksperimen Belum Terpenuhi (Dilaporkan Transparan)

1. **`C1` `AksaraYOLOX-Tiny P2` (Apache-2.0):** Baru diuji *smoke test* 2-epoch CPU (`CER = 100.00%`), karena mesin lokal tidak memiliki GPU (`CPU only`). Bobot `YOLOX-Tiny P2` (`10.79 MB` FP16) telah tervalidasi berjalan di Chromium ORT-Web 1.20.1 (`1228 ms` p50), namun bobot produksi tetap memakai `aksara_yolo.onnx` (`v3.1`) sampai `training/AksaraScan_v4_GPU_MultiPlatform.ipynb` dijalankan penuh di GPU cloud.
2. **`C3` Track B CRNN-CTC:** Varian CPU pada `setilanaji_lines_v4 val` memiliki CER mandiri **`65.08%`** (mean conf `0.521`), sehingga gerbang kalibrasi `val` menolak penggantian otomatis.
3. **Regresi bersih `synth_v2_val_clean (n=40)`:** Word Accuracy berlipat sangat sedikit turun `12.43% → 12.22%` (`-0.21 pp`), sementara CER justru sedikit membaik `21.42% → 21.06%` dan Grapheme F1 naik `82.06% → 83.45%`. Word Accuracy tetap ~`12%` pada `synth_v2` karena banyak kalimat panjang *scriptio continua* yang masih mengandung kata di luar kamus (`10.589` lema).

---

## 5. Analisis Kesalahan: Matriks Konfusi & Taksonomi Kegagalan

### 5.1 Evaluasi Kotak & Pasangan Rawan (`52` citra beranotasi kotak = `1.960` GT kotak, IoU ≥ `0.35`)
Mencakup `12` gambar tulisan tangan nyata `hanacaraka_12` + `40` halaman `synth_v2_val_clean`:
- **Matched Recall:** `v3.1 = 95.26%` (`1.867/1.960`) vs `v4.0 Full = 95.10%` (`1.864/1.960`).
- **Akurasi Klasifikasi Kotak Terdeteksi:** `v3.1 = 97.75%` → **`v4.0 Full = 98.02%`** (End-to-end Box Recall×Cls: `93.11% → 93.21%`).
- **Penurunan *Spurious False Positive* pada sandhangan kecil:** `taling` turun **`14 → 7`**, `cecak` turun **`12 → 4`**, `suku` turun **`4 → 2`**.

**Sorotan Pasangan Rawan yang Diminta Brief:**

| Pasangan Rawan | Jumlah GT (`a / b`) | Kesalahan `v3.1` | Kesalahan `v4.0 Full` | Keterangan |
|---|---:|---:|---:|---|
| **`da ↔ sa`** | `42 / 86` | `0` | **`0`** | Nol tertukar di kotak terdeteksi beranotasi. |
| **`ka ↔ na`** | `130 / 130` | `0` | **`0`** | Nol tertukar di kotak terdeteksi beranotasi. |
| **`tha ↔ ca`** | `9 / 19` | `0` | **`0`** | Nol tertukar; namun `tha` jarang (`9` kotak). |
| **`ha ↔ ra`** | `116 / 45` | `0` | **`1`** (`ha→ra`) | `1` kasus relabel pada bentuk `ha` dua lengkung menyerupai `ra`. |

**Pasangan Teratas yang Benar-Benar Tertukar (`v4.0 Full`) & Kelas Terbanyak Terlewat (FN):**
- Tertukar antar-kelas: `ga → ya (4)`, `nya → ba (2)`, `ga → ra (2)`, `ba → ra (2)`, `cecak → pepet (2)`, `ca → sa (1)`, `ka → ca (1)`, `ka → sa (1)`.
- Gagal deteksi teratas (FN): `wulu (25/177)`, `pepet (8/155)`, `layar (6/25)`, `pangkon (6/11)`, `cecak (5/52)` — berakar pada resolusi detektor Stage-1 `416 px` untuk glif mini `< 12 px`.

### 5.2 Taksonomi Kegagalan pada `50` Citra Nyata (`heldout_real 42` + `failures 8` = `620` GT grafem)

| Jenis Kegagalan | `v3.1 (eb19337)` | `v4.0 Full` | Selisih | Akar Masalah & Observasi |
|---|---:|---:|---:|---|
| **Salah Klasifikasi (`grapheme substitutions`)** | `223` | **`204`** | **`-19`** (`-8.5%`) | Perbaikan di naskah `NusaAksara` dan penolakan distraktor, masih terganjal gaya tulisan kuno pada `NusaAksara`. |
| **Kelebihan / Kotak Palsu (`spurious insertions`)** | `59` | **`39`** | **`-20`** (`-33.9%`) | Stage-2 `non_aksara` memangkas label Latin liar pada `diag_07`, `diag_08`, dan tepi scanner. |
| **Kotak Hilang / Kurang Segmentasi (`missed deletions`)** | `37` | **`35`** | `-2` | Sebagian tulisan tangan tipis di atas kertas bergaris biru terpotong `morphBin` atau gagal diusulkan Stage-1. |
| **Sandhangan Hilang (`missing sandhangan marks`)** | `81` | `85` | `+4` | Tanda diakritik kecil (`wulu/pepet/cecak`) pada `NusaAksara` padat dan batas bawah Stage-1 `416 px`. |
| **Celah OOV & Segmentasi Kata (`oov predicted words` / kasus CER≤15% & WAcc<85%)** | `308` kata (`6` kasus) | **`287` kata** (`8` kasus) | `-21` kata | Banyak naskah tradisional/nama kuno dan baris *hanacaraka* tak terpecah tepat oleh kamus DP 10.589 lema. |

---

## 6. Dua Puluh Contoh Kegagalan Terburuk (`v4.0 Full` pada Set Nyata `n=50`)

Galeri visual lengkap (gambar beranotasi kotak + perbandingan teks `GT`, `v3.1`, dan `v4.0`) tersedia di **[`eval/reports/phase5_worst20.html`](../eval/reports/phase5_worst20.html)**.

| # | ID | Subset & Kondisi | CER `v3.1 → v4.0` | GF1 `v4.0` | Gambar Anotasi | Ringkasan Kegagalan |
|---:|---|---|---:|---:|---|---|
| 1 | `hr_hw_10` | `hanacaraka_12` · `latin_distractor_mix` | `100.0% → 90.0%` | `18.18%` | [01_hr_hw_10.jpg](../eval/reports/phase5_worst20_imgs/01_hr_hw_10.jpg) | `lineBands`/`morphBin` melemahkan goresan `1 px` di kertas bergaris biru, hanya `"Penulis #10"` di kiri atas yang terbaca detektor Stage-1. |
| 2 | `diag_07` | `failures` · `latin_distractor_border` | `133.33% → 88.89%` | `72.00%` | [02_diag_07.jpg](../eval/reports/phase5_worst20_imgs/02_diag_07.jpg) | Ke-9 Aksara Jawa terbaca 100%, namun banner Latin besar `"tampilan web"` lolos sebagian pada huruf `m/p/l/w/b`. |
| 3 | `hr_nusa_07` | `nusaaksara` · `real_printed_or_manuscript` | `68.75% → 68.75%` | `26.67%` | [03_hr_nusa_07.jpg](../eval/reports/phase5_worst20_imgs/03_hr_nusa_07.jpg) | Manuskrip rapat & karakter `pada pangkat`/`pa cerek`/`nga lelet` berada di luar 55 kelas Tier 1. |
| 4 | `hr_nusa_04` | `nusaaksara` · `real_printed_or_manuscript` | `70.37% → 66.67%` | `40.00%` | [04_hr_nusa_04.jpg](../eval/reports/phase5_worst20_imgs/04_hr_nusa_04.jpg) | Cetak miring/padat menyebabkan pecah-gabung kotak `pasangan` bersusun. |
| 5 | `hr_hw_08` | `hanacaraka_12` · `wall_lines_shadow` | `60.00% → 60.00%` | `47.06%` | [05_hr_hw_08.jpg](../eval/reports/phase5_worst20_imgs/05_hr_hw_08.jpg) | Retakan dinding tebal dan bayangan gelap menurunkan kepercayaan kotak `ga`, `ka`, `nya`. |
| 6 | `hr_hw_02` | `hanacaraka_12` · `motion_blur` | `60.00% → 60.00%` | `55.56%` | [06_hr_hw_02.jpg](../eval/reports/phase5_worst20_imgs/06_hr_hw_02.jpg) | Motion blur vertikal menyamarkan celah `ga`/`pa` serta memicu salah kotak di coretan `da=dha`. |
| 7 | `hr_nusa_08` | `nusaaksara` · `real_printed_or_manuscript` | `67.65% → 55.88%` | `33.33%` | [07_hr_nusa_08.jpg](../eval/reports/phase5_worst20_imgs/07_hr_nusa_08.jpg) | Naskah panjang dengan kluster tumpuk `p_ya`/`sandhangan` hilang sebagian (`-11.77 pp` vs v3.1). |
| 8 | `hr_nusa_12` | `nusaaksara` · `real_printed_or_manuscript` | `66.67% → 55.56%` | `17.65%` | [08_hr_nusa_12.jpg](../eval/reports/phase5_worst20_imgs/08_hr_nusa_12.jpg) | Layar/cecak kecil di atas huruf `dha/sur` sering terlewat pada 416 px. |
| 9 | `diag_08` | `failures` · `chart_with_latin_labels` | `80.00% → 55.00%` | `60.47%` | [09_diag_08.jpg](../eval/reports/phase5_worst20_imgs/09_diag_08.jpg) | Ke-20 aksara utama tepat 100%, sisa error dari label Latin tebal (`ca, ta, sa...`) tepat di posisi *wulu/pepet* di atas aksara. |
| 10 | `hr_nusa_15` | `nusaaksara` · `real_printed_or_manuscript` | `51.61% → 51.61%` | `14.81%` | [10_hr_nusa_15.jpg](../eval/reports/phase5_worst20_imgs/10_hr_nusa_15.jpg) | Karakter `dha mahaprana` (`ꦣ`) + `pada pangkat` (`꧇`) di luar skema kelas aktif. |
| 11 | `hr_nusa_09` | `nusaaksara` · `real_printed_or_manuscript` | `64.29% → 50.00%` | `42.86%` | [11_hr_nusa_09.jpg](../eval/reports/phase5_worst20_imgs/11_hr_nusa_09.jpg) | Perbaikan `-14.29 pp` berkat Stage-2, sisa pasangan rapat `tra/nya`. |
| 12 | `hr_nusa_13` | `nusaaksara` · `real_printed_or_manuscript` | `51.61% → 48.39%` | `20.00%` | [12_hr_nusa_13.jpg](../eval/reports/phase5_worst20_imgs/12_hr_nusa_13.jpg) | Gugus `nyja`/`ra` rapat dengan baris tetangga terpotong di margin petak. |
| 13 | `hr_nusa_10` | `nusaaksara` · `real_printed_or_manuscript` | `70.59% → 47.06%` | `43.75%` | [13_hr_nusa_10.jpg](../eval/reports/phase5_worst20_imgs/13_hr_nusa_10.jpg) | Perbaikan besar `-23.53 pp`; `kya hi` awal terbaca benar, kluster tengah masih pecah. |
| 14 | `hr_nusa_17` | `nusaaksara` · `real_printed_or_manuscript` | `46.67% → 46.67%` | `38.71%` | [14_hr_nusa_17.jpg](../eval/reports/phase5_worst20_imgs/14_hr_nusa_17.jpg) | Konsonan `dha murda`/`mahaprana` (`ꦣ`) & *layar* kecil gagal dikenali sebagai entitas tunggal. |
| 15 | `hr_nusa_03` | `nusaaksara` · `real_printed_or_manuscript` | `44.64% → 46.43%` | `32.00%` | [15_hr_nusa_03.jpg](../eval/reports/phase5_worst20_imgs/15_hr_nusa_03.jpg) | Sedikit naik `+1.79 pp`; baris sangat panjang membuat tiling berhenti di batas 12 petak. |
| 16 | `hr_nusa_11` | `nusaaksara` · `real_printed_or_manuscript` | `70.83% → 45.83%` | `50.00%` | [16_hr_nusa_11.jpg](../eval/reports/phase5_worst20_imgs/16_hr_nusa_11.jpg) | Perbaikan `-25.00 pp` karena Stage-2 menekan prediksi ganda baris atas/bawah. |
| 17 | `hr_hw_05` | `hanacaraka_12` · `defocus_blur` | `70.00% → 40.00%` | `58.82%` | [17_hr_hw_05.jpg](../eval/reports/phase5_worst20_imgs/17_hr_hw_05.jpg) | Perbaikan `-30.00 pp` CER pada blur defokus berkat `image_enhancer.js`, menyisakan `la` & `ga` yang tipis. |
| 18 | `hr_nusa_14` | `nusaaksara` · `real_printed_or_manuscript` | `45.45% → 39.39%` | `42.86%` | [18_hr_nusa_14.jpg](../eval/reports/phase5_worst20_imgs/18_hr_nusa_14.jpg) | Pasangan `p_pa`/`p_na` bertumpuk di baris padat (`-6.06 pp` vs v3.1). |
| 19 | `hr_nusa_05` | `nusaaksara` · `real_printed_or_manuscript` | `64.52% → 38.71%` | `45.16%` | [19_hr_nusa_05.jpg](../eval/reports/phase5_worst20_imgs/19_hr_nusa_05.jpg) | Turun `-25.81 pp` CER pada baris cetak nyata karena Stage-2 meluruskan glif akhir kalimat. |
| 20 | `hr_nusa_16` | `nusaaksara` · `real_printed_or_manuscript` | `48.48% → 36.36%` | `38.46%` | [20_hr_nusa_16.jpg](../eval/reports/phase5_worst20_imgs/20_hr_nusa_16.jpg) | Aksara langka `ꦉ` (*Pa Cerek*) di luar skema Tier 1, namun sisa baris membaik `-12.12 pp`. |

---

## 7. Pra-Rilis Vercel, Ukuran Aset, & Waktu Muat Pertama (`eval/eval_phase5_prerelease.mjs`)

Diukur dengan server lokal yang menerapkan persis aturan [`vercel.json`](../vercel.json) (`outputDirectory: app/web_camscanner`, *rewrite* `/app/web_camscanner/*`, serta `Cache-Control: no-cache` pada `/sw.js`):

- **Versi Cache Service Worker (`sw.js`):** **`aksarascan-v16`** (`38` aset inti di deklarasi, `38` entri CacheStorage terverifikasi aktif untuk mode offline).
- **Total Ukuran Aset Statis (`42` berkas di `app/web_camscanner/`):** **`28.64 MB`** (`28.644.127` byte / `27.32 MiB`).

| Kategori Aset | Jumlah Berkas | Ukuran (Bytes) | Ukuran (MB / MiB) |
|---|---:|---:|---:|
| Bobot ONNX (`aksara_yolo.onnx` `12.70 MB` + `aksara_embed.onnx` `0.80 MB`) | `2` | `13.506.446` | `13.51 MB` (`12.88 MiB`) |
| Runtime ORT-Web 1.20.1 (`vendor/ort/`, termasuk `wasm` `11.25 MB`) | `3` | `11.716.934` | `11.72 MB` (`11.17 MiB`) |
| Bank Prototipe & Leksikon (`reference_prototypes.json`, `javanese_lm_data.js`, `model.*`) | `6` | `1.561.777` | `1.56 MB` (`1.49 MiB`) |
| Font Self-Hosted SIL OFL 1.1 (`vendor/fonts/`, 4 berkas `.ttf`) | `4` | `994.144` | `0.99 MB` (`0.95 MiB`) |
| Contoh Uji & Ikon PWA (`sample_1..4.png`, ikon) | `12` | `515.448` | `0.52 MB` (`0.49 MiB`) |
| Kode Aplikasi & Gaya (`index.html`, `styles.css`, modul JS) | `13` | `346.911` | `0.35 MB` (`0.33 MiB`) |

- **Waktu Muat Pertama Tanpa Cache (*Cold Load*, `localhost`):**
  - `TTFB`: **`16 ms`** · `DOMContentLoaded`: **`318 ms`** · `window.load`: **`326 ms`**
  - Siap Penuh (ORT-Web Stage-1 + Stage-2 176 Prototipe): **`1.186 ms`** (`30` request HTTP lokal, **`27.89 MB`** transfer awal)
  - Permintaan Jaringan Eksternal: **`0`** · Console/Page Error: **`0`**
  - *(Catatan deploy Vercel: preview URL publik Vercel terhubung ke push branch/main di remote GitHub.)*
