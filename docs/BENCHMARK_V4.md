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

Pengujian dijalankan end-to-end lewat DOM aplikasi web nyata (`#fileInput -> Worker Enhancer -> ORT-Web Stage-1 -> ORT-Web Stage-2 -> orthography.js v4 -> javanese_lm_engine.js`). Waktu muat kesiapan penuh model (`DOMContentLoaded` ke `YOLOv8 Siap`) turun dari **`1832 ms`** (`v3.1`) menjadi **`825 ms`** (`v4.0` dari bundel WASM lokal).

| Set Uji | `n` | Unicode CER `v3.1` (95% CI) | Unicode CER `v4.0` (95% CI) | Delta CER | Grapheme F1 `v3.1 → v4.0` | Word Acc Folded `v3.1 → v4.0` | Latensi `v4.0` p50 / p95 |
|---|---:|---|---|---|---|---|---|
| **`heldout_real` (Total Terkunci)** | `42` | `34.31%` `[26.26, 42.49]` | **`28.68%`** `[22.24, 34.99]` | **`-5.63 pp`** | `59.64% → 63.52%` (`+3.88 pp`) | `29.80% → 31.17%` (`+1.37 pp`)* | `957.1 / 1691.8 ms` |
| **`heldout_real` · Foto Buram Nyata** (`motion/defocus/latin_blur`) | `5` | `34.00%` | **`30.00%`** | **`-4.00 pp`** | `70.05% → 75.93%` (`+5.88 pp`) | `30.50% → 28.00%` | — |
| **`failures` (Set Diagnostik + Bug Laporan)** | `8` | `32.78%` `[4.73, 67.51]` | **`2.01%`** `[0.00, 4.79]` | **`-30.77 pp`** | `84.81% → 98.72%` (`+13.91 pp`) | `29.79% → 44.30%` (`+14.51 pp`)** | `2855.1 / 4438.4 ms` |
| **`synth_v2_val_clean` (Penjaga Regresi)** | `40` | `21.42%` `[15.95, 27.04]` | **`20.67%`** `[14.97, 26.81]` | **`-0.75 pp`** | `82.06% → 82.79%` (`+0.73 pp`) | `12.43% → 12.67%` (`+0.24 pp`) | `1219.6 / 2361.7 ms` |

> \*Catatan metrik *Word Accuracy*: Pada `failures (n=8)` dan `synth_v2_val_clean (n=40)` dipakai `latin_gt` tulis-tangan/manifes asli (`strict` tanpa pelipatan diakritik `v3.1 → v4.0` pada `failures`: `27.71% → 42.22%`). Pada `heldout_real (n=42)` karena manifes hanya menyediakan `unicode_gt`, `latin_gt` acuan diturunkan secara deterministik melalui `AksaraOrthography.decodeClustersToLatin` pada `unicode_gt`.

### Efisiensi Penolakan Non-Aksara / Distraktor (`synth_v4_val_neg`, `n=29`)
Pada `29` gambar validasi negatif murni (`0` kotak Aksara Jawa: teks Latin cetak/tulisan tangan, garis buku, retakan dinding):

| Metrik | `v3.1 (eb19337)` | `v4.0 Full` | Perubahan |
|---|---:|---:|---|
| Rata-rata kotak palsu (FP / gambar) | `7.69` (`223` total FP) | **`1.52`** (`44` total FP) | **`-80.2%`** |
| Halaman bersih tanpa 1 pun kotak FP | `27.6%` (`8/29`) | **`41.4%` (`12/29`)** | **`+13.8 pp`** |

---

## 3. Matriks Ablasi Penuh (Browser & Offline Python)

Menguji kontribusi masing-masing modul secara terisolasi:
- **Browser (`eval_phase5_browser.mjs`):** `v3_1` vs `v4_enh_only` (Stage-2 diblokir agar jatuh ke Stage-1 Fallback + `orthography.js` v4 + Worker Enhancer) vs `v4_stage2_only` (`skipEnhance: true`) vs `v4_full`.
- **Offline Python (`eval/reports/phase2_ablation.json`):** `C0`, `C1` (YOLOX-Tiny P2 640px *smoke test* 2-epoch CPU), `C2` (`C0 + Stage-2`), `C3` (`C2 + Track B CRNN`).

| Konfigurasi | Jalur Eksekusi | Heldout `n=42` CER / GF1 | Blur Real `n=5` CER | Failures `n=8` CER / GF1 | Negatif `n=29` FP/img | Clean Synth `n=40` CER | Console Err |
|---|---|---:|---:|---:|---:|---:|---:|
| **`v3_1`** (`eb19337`) | Browser WASM | `34.31%` / `59.64%` | `34.00%` | `32.78%` / `84.81%` | `7.69` | `21.42%` | `0` |
| **`v4_enh_only`** (Stage-1 + Enhancer + Ortho v4) | Browser WASM | `32.96%` / `61.62%` | `38.00%` | `65.41%` / `82.40%` | `6.93` | `21.38%` | `2` *(route abort uji)* |
| **`v4_stage2_only`** (Stage-2 176-proto, Enhancer skip) | Browser WASM | **`28.33%` / `64.00%`** | **`24.00%`** | `11.74%` / `91.88%` | **`1.52`** | `27.99%` | `0` |
| **`v4_full`** (Enhancer + Stage-2 + Ortho v4 + LM) | Browser WASM | `28.68%` / `63.52%` | `30.00%` | **`2.01%` / `98.72%`** | **`1.52`** | **`20.67%`** | `0` |
| **`C0`** (`aksara_yolo.onnx` Python) | Python ORT CPU | `37.36%` / `53.50%` | — | `27.81%` / `85.67%` | `9.21` *(val)* | `23.16%` | — |
| **`C1`** (`YOLOX-Tiny P2`, 2-ep CPU smoke) | Python ORT CPU | `100.00%` / `0.00%` | — | `100.00%` / `0.00%` | — | — | — |
| **`C2`** (`C0 + Stage-2` Python) | Python ORT CPU | `34.48%` / `55.97%` | — | `20.24%` / `86.92%` | `3.83` *(val)* | `23.54%` | — |
| **`C3`** (`C2 + Track B CRNN existing`) | Python ORT CPU | `34.48%` *(ditolak gerbang `val`)* | — | `20.24%` | `3.83` *(val)* | `23.54%` | — |

**Pembacaan Ablasi:**
1. **Worker Enhancer (`image_enhancer.js` + `scanMask`)** memulihkan garis tipis `1 px` di kertas bergaris biru (`hr_hw_10`: `100% → 10%` CER) sekaligus menjaga `synth_v2_val_clean` (`20.67%` vs `27.99%` jika enhancer dilewati pada `v4_stage2_only`).
2. **Stage-2 Metric Embedder + Filter Konteks (`pruneLatinContextAndTwinLabels`)** menyumbang penurunan besar di `NusaAksara` (`53.22% → 45.38%` CER), menghilangkan kebocoran label Latin (`failures_n8` `32.78% → 2.01%`), dan memangkas FP pada halaman distraktor hingga **`80.2%`**.
3. **Gabungan `v4_full`** meraih keseimbangan produksi terbaik di semua 4 set sekaligus (`28.68%` held-out, `2.01%` diagnostik, `20.67%` regresi bersih, `1.52 FP/img`).

---

## 4. Rincian Per Subset, Kondisi, & Hasil Negatif / Regresi

### 4.1 Rincian Subset & Kondisi `eval/heldout_real/` (`n=42`)

| Subset / Kondisi | `n` | CER `v3.1` | CER `v4.0` | GF1 `v3.1 → v4.0` | Catatan Jujur |
|---|---:|---:|---:|---|---|
| `nusaaksara_acl2025_eval_only` | `18` | `53.22%` | **`45.38%`** | `33.46% → 38.23%` | Peningkatan `-7.84 pp` CER murni dari generalisasi Stage-2 (`NusaAksara` 0% di data latih). |
| `setilanaji_real_book_val` | `12` | **`7.75%`** | `8.16%` *(+0.41 pp)* | `87.48% → 86.50%` | **Regresi minor negatif (`+0.41 pp`)**: batas relabel Stage-2 menggeser 1 karakter beresolusi rapat. |
| `hanacaraka_12_unseen_human_writers` | `12` | `32.50%` | **`24.17%`** | `71.08% → 78.48%` | Turun **`-8.33 pp`** CER pada penulis tak terlihat. |
| └ `defocus_blur` | `2` | `35.00%` | **`25.00%`** | `68.75% → 81.25%` | `-10.0 pp` CER berkat penajaman TV-bounded. |
| └ `latin_distractor_blur` | `1` | `20.00%` | `20.00%` | `85.71% → 80.00%` | — |
| └ `latin_distractor_mix` (`hr_hw_10`) | `1` | `100.00%` | **`10.00%`** | `0.00% → 90.00%` | **Pulih `-90.0 pp`**: goresan `1 px` pada kertas bergaris biru kini terdeteksi utuh. |
| └ `motion_blur` | `2` | `40.00%` | `40.00%` | `63.53% → 68.57%` | Deteksi kotak meningkat (`GF1 +5.04 pp`). |
| └ `multi_color_pen` | `2` | **`10.00%`** | `15.00%` *(+5.0 pp)* | `90.00% → 85.00%` | **Regresi minor (`+5.0 pp`)**: pembatasan tarik kroma agar garis pastel tidak menghitam melemahkan 1 huruf tipis berwarna. |
| └ `wall_lines_shadow` | `2` | `40.00%` | `40.00%` | `71.31% → 61.05%` | CER seri; retakan tebal diagonal menyilang huruf masih menantang Stage-1. |
| └ `sharp_clean` | `2` | `10.00%` | `10.00%` | `90.00% → 90.00%` | Bersih, regresi sebelumnya telah pulih ke `10.0%`. |

### 4.2 Rincian Set Diagnostik `eval/failures/` (`n=8`)

| ID | Kondisi | CER `v3.1 → v4.0` | GF1 `v3.1 → v4.0` | Bacaan Latin `v4.0` |
|---|---|---:|---:|---|
| `diag_01` | Printed low-res + caption Latin | `37.84% → 0.00%` | `65.31% → 100.00%` | `wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.` |
| `diag_02` | Printed low-res cropped | `0.00% → 0.00%` | `100.00% → 100.00%` | `wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.` |
| `diag_03` | Synthetic lined paper (`sample_4`) | `0.00% → 0.00%` | `100.00% → 100.00%` | `aksara jawa mangan sêga krama kothi` |
| `diag_04` | Handwriting shadow stress | `0.00% → 0.00%` | `100.00% → 100.00%` | `ana cara ka das awala` |
| `diag_05` | Handwriting border lines stress | `11.11% → 0.00%` | `94.74% → 100.00%` | `ana cara ka das awala` |
| `diag_06` | Handwriting low-res 600 px | `0.00% → 0.00%` | `100.00% → 100.00%` | `ana cara ka das awala` |
| `diag_07` | Handwriting + Latin distractor banner | `133.33% → 11.11%` | `64.29% → 94.74%` | `ra ana cara ka das awala` |
| `diag_08` | Handwriting chart + bold Latin labels | `80.00% → 5.00%` | `54.17% → 95.00%` | `ana cara ka data saw ala padha jaya nya mangga bath anga` |

### 4.3 Hasil Negatif / Eksperimen Belum Terpenuhi (Dilaporkan Transparan)

1. **`C1` `AksaraYOLOX-P2` (Apache-2.0):** Baru diuji *smoke test* CPU (`train_yolox.py --grad-accum 2` + `yolox-tiny_img640_s42_fp16.onnx` `10.79 MB`). Meskipun laptop memiliki `NVIDIA GeForce GTX 1650` (`4096 MiB`), Python lokal terpasang bertipe CPU-only dan ruang bebas drive `C:` tersisa `7.95 GB` (tidak memadai untuk wheel CUDA `~2.53 GB` + `~5.5 GB` ekstrak aman). Bobot produksi tetap memakai `aksara_yolo.onnx` sampai GPU training dijalankan lewat Kaggle/Lightning (`training/AksaraScan_v4_GPU_MultiPlatform.ipynb`) dan dipasang dengan `tools/install_yolox_release.py`.
2. **`C3` Track B CRNN-CTC:** Varian CPU pada `setilanaji_lines_v4 val` memiliki CER mandiri **`65.08%`** (mean conf `0.521`), sehingga gerbang kalibrasi `val` menolak penggantian otomatis.
3. **Trade-off Recall Kotak Rapat & Regresi Lokal:** `matched_recall_pct` pada `1.960` kotak beranotasi turun tipis `95.26% → 94.39%` (`-0.87 pp`, meski CER akhir `synth_v2_val_clean` membaik `21.42% → 20.67%` dan Word Acc naik `12.43% → 12.67%`). Regresi lokal tercatat pada `multi_color_pen (+5.0 pp)`, `setilanaji (+0.41 pp)`, `hr_nusa_14 (+6.07 pp)`, dan `hr_nusa_13 (+3.23 pp)`.

---

## 5. Analisis Kesalahan: Matriks Konfusi & Taksonomi Kegagalan

### 5.1 Evaluasi Kotak & Pasangan Rawan (`52` citra beranotasi kotak = `1.960` GT kotak, IoU ≥ `0.35`)
Mencakup `12` gambar tulisan tangan nyata `hanacaraka_12` + `40` halaman `synth_v2_val_clean`:
- **Matched Recall:** `v3.1 = 95.26%` (`1.867/1.960`) vs `v4.0 Full = 94.39%` (`1.850/1.960`).
- **Akurasi Klasifikasi Kotak Terdeteksi:** `v3.1 = 97.75%` vs `v4.0 Full = 97.57%` (End-to-end Box Recall×Cls: `93.11% → 92.09%`).
- **Penurunan *Spurious False Positive* pada sandhangan kecil:** `taling` turun **`14 → 9`**, `cecak` turun **`12 → 3`**, `tarung` turun **`7 → 4`**, `suku` turun **`4 → 3`**.

**Sorotan Pasangan Rawan yang Diminta Brief:**

| Pasangan Rawan | Jumlah GT (`a / b`) | Kesalahan `v3.1` | Kesalahan `v4.0 Full` | Keterangan |
|---|---:|---:|---:|---|
| **`da ↔ sa`** | `42 / 86` | `0` | **`0`** | Nol tertukar di kotak terdeteksi beranotasi. |
| **`ka ↔ na`** | `130 / 130` | `0` | `1` (`ka→na`) | `1` kesalahan di `130` instans `ka`. |
| **`tha ↔ ca`** | `9 / 19` | `0` | **`0`** | Nol tertukar; namun `tha` jarang (`9` kotak). |
| **`ha ↔ ra`** | `116 / 45` | `0` | **`0`** | Nol tertukar di kotak terdeteksi beranotasi. |

**Pasangan Teratas yang Benar-Benar Tertukar (`v4.0 Full`) & Kelas Terbanyak Terlewat (FN):**
- Tertukar antar-kelas: `ga → ya (4)`, `cecak → wulu (4)`, `ha → ya (2)`, `ga → ha (2)`, `ga → ra (2)`, `p_nga → nga (2)`, `suku → p_ma (2)`, `cecak → pepet (2)`.
- Gagal deteksi teratas (FN): `wulu (29/177)`, `pepet (8/155)`, `layar (8/25)`, `ha (5/116)`, `la (5/38)`, `ba (5/33)`, `cecak (5/52)`, `pangkon (5/11)` — berakar pada resolusi detektor Stage-1 `416 px` untuk glif mini `< 12 px`.

### 5.2 Taksonomi Kegagalan pada `50` Citra Nyata (`heldout_real 42` + `failures 8` = `620` GT grafem)

| Jenis Kegagalan | `v3.1 (eb19337)` | `v4.0 Full` | Selisih | Akar Masalah & Observasi |
|---|---:|---:|---:|---|
| **Salah Klasifikasi (`grapheme substitutions`)** | `223` | **`196`** | **`-27`** (`-12.1%`) | Penurunan kesalahan pengenalan di `NusaAksara` (`-7.84 pp`), `hanacaraka_12` (`-8.33 pp`), dan set diagnostik. |
| **Kelebihan / Kotak Palsu (`spurious insertions`)** | `59` | **`17`** | **`-42`** (`-71.2%`) | Pemangkasan besar label Latin `diag_01/07/08` lewat Stage-2 & `pruneLatinContextAndTwinLabels`. |
| **Kotak Hilang / Kurang Segmentasi (`missed deletions`)** | `37` | **`32`** | **`-5`** | Pemulihan garis tipis `1 px` (`hr_hw_10`); sisa terjadi saat retakan tembok memutus huruf (`hr_hw_08`). |
| **Sandhangan Hilang (`missing sandhangan marks`)** | `81` | `112` | `+31` | Terutama pada manuskrip padat `NusaAksara` + beberapa tanda mikro yang ikut tersaring pada ambang konservatif; memerlukan detector `AksaraYOLOX-P2` @ 640/960 px. |
| **Celah OOV & Segmentasi Kata (`oov predicted words` / kasus CER≤15% & WAcc<85%)** | `308` kata (`6` kasus) | **`275` kata** (`10` kasus) | `-33` kata | Banyak naskah tradisional/nama kuno dan baris *hanacaraka* tak terpecah tepat oleh kamus DP 10.589 lema. |

---

## 6. Dua Puluh Contoh Kegagalan Terburuk (`v4.0 Full` pada Set Nyata `n=50`)

Galeri visual lengkap (gambar beranotasi kotak + perbandingan teks `GT`, `v3.1`, dan `v4.0`) tersedia di **[`eval/reports/phase5_worst20.html`](../eval/reports/phase5_worst20.html)**. Catatan utama: **`hr_hw_10` (`100% → 10%`), `diag_07` (`133.33% → 11.11%`), dan `diag_08` (`80% → 5%`) telah keluar dari 20 kasus terburuk**.

| # | ID | Subset & Kondisi | CER `v3.1 → v4.0` | GF1 `v4.0` | Gambar Anotasi | Ringkasan Kegagalan |
|---:|---|---|---:|---:|---|---|
| 1 | `hr_nusa_07` | `nusaaksara` · `real_printed_or_manuscript` | `68.75% → 68.75%` | `26.67%` | [01_hr_nusa_07.jpg](../eval/reports/phase5_worst20_imgs/01_hr_nusa_07.jpg) | Manuskrip rapat dengan derau bintik serta karakter `pada pangkat`/`pa cerek` di luar 55 kelas Tier 1. |
| 2 | `hr_nusa_04` | `nusaaksara` · `real_printed_or_manuscript` | `70.37% → 66.67%` | `40.00%` | [02_hr_nusa_04.jpg](../eval/reports/phase5_worst20_imgs/02_hr_nusa_04.jpg) | Cetak miring/padat menyebabkan pecah-gabung kotak `pasangan` bersusun. |
| 3 | `hr_nusa_08` | `nusaaksara` · `real_printed_or_manuscript` | `67.65% → 61.76%` | `29.63%` | [03_hr_nusa_08.jpg](../eval/reports/phase5_worst20_imgs/03_hr_nusa_08.jpg) | Naskah panjang dengan kluster tumpuk `p_ya`/`sandhangan` hilang sebagian (`-5.89 pp` vs v3.1). |
| 4 | `hr_hw_08` | `hanacaraka_12` · `wall_lines_shadow` | `60.00% → 60.00%` | `42.11%` | [04_hr_hw_08.jpg](../eval/reports/phase5_worst20_imgs/04_hr_hw_08.jpg) | Retakan dinding diagonal memotong langsung lengkung tipis 1 px (`sa/ga/ka/nya`). |
| 5 | `hr_hw_02` | `hanacaraka_12` · `motion_blur` | `60.00% → 60.00%` | `57.14%` | [05_hr_hw_02.jpg](../eval/reports/phase5_worst20_imgs/05_hr_hw_02.jpg) | Motion blur vertikal menyamarkan celah `ga`/`pa` serta memicu salah kotak di coretan `da=dha`. |
| 6 | `hr_nusa_12` | `nusaaksara` · `real_printed_or_manuscript` | `66.67% → 55.56%` | `18.18%` | [06_hr_nusa_12.jpg](../eval/reports/phase5_worst20_imgs/06_hr_nusa_12.jpg) | Layar/cecak kecil di atas huruf `dha/sur` sering terlewat pada 416 px. |
| 7 | `hr_nusa_13` | `nusaaksara` · `real_printed_or_manuscript` | `51.61% → 54.84%` | `14.29%` | [07_hr_nusa_13.jpg](../eval/reports/phase5_worst20_imgs/07_hr_nusa_13.jpg) | Gugus `nyja`/`ra` rapat dengan baris tetangga terpotong di margin petak (`+3.23 pp`). |
| 8 | `hr_nusa_05` | `nusaaksara` · `real_printed_or_manuscript` | `64.52% → 51.61%` | `27.59%` | [08_hr_nusa_05.jpg](../eval/reports/phase5_worst20_imgs/08_hr_nusa_05.jpg) | Membaik `-12.91 pp` dibanding v3.1 meski gugus padat tengah masih tercampur. |
| 9 | `hr_nusa_14` | `nusaaksara` · `real_printed_or_manuscript` | `45.45% → 51.52%` | `29.63%` | [09_hr_nusa_14.jpg](../eval/reports/phase5_worst20_imgs/09_hr_nusa_14.jpg) | Pasangan `p_pa`/`p_na` bertumpuk di baris padat (`+6.07 pp`). |
| 10 | `hr_nusa_09` | `nusaaksara` · `real_printed_or_manuscript` | `64.29% → 50.00%` | `42.86%` | [10_hr_nusa_09.jpg](../eval/reports/phase5_worst20_imgs/10_hr_nusa_09.jpg) | Perbaikan `-14.29 pp` berkat Stage-2, sisa pasangan rapat `tra/nya`. |
| 11 | `hr_hw_05` | `hanacaraka_12` · `defocus_blur` | `70.00% → 50.00%` | `62.50%` | [11_hr_hw_05.jpg](../eval/reports/phase5_worst20_imgs/11_hr_hw_05.jpg) | Perbaikan `-20.00 pp` CER pada blur defokus berkat `image_enhancer.js`. |
| 12 | `hr_nusa_17` | `nusaaksara` · `real_printed_or_manuscript` | `46.67% → 46.67%` | `38.71%` | [12_hr_nusa_17.jpg](../eval/reports/phase5_worst20_imgs/12_hr_nusa_17.jpg) | Konsonan `dha murda`/`mahaprana` (`ꦣ`) & *layar* kecil gagal dikenali sebagai entitas tunggal. |
| 13 | `hr_nusa_03` | `nusaaksara` · `real_printed_or_manuscript` | `44.64% → 46.43%` | `32.00%` | [13_hr_nusa_03.jpg](../eval/reports/phase5_worst20_imgs/13_hr_nusa_03.jpg) | Baris sangat panjang; kluster padat akhir kalimat tergeser `+1.79 pp`. |
| 14 | `hr_nusa_10` | `nusaaksara` · `real_printed_or_manuscript` | `70.59% → 44.12%` | `38.71%` | [14_hr_nusa_10.jpg](../eval/reports/phase5_worst20_imgs/14_hr_nusa_10.jpg) | Perbaikan besar `-26.47 pp`; `kya hi` awal terbaca benar, kluster tengah masih pecah. |
| 15 | `hr_nusa_16` | `nusaaksara` · `real_printed_or_manuscript` | `48.48% → 42.42%` | `38.46%` | [15_hr_nusa_16.jpg](../eval/reports/phase5_worst20_imgs/15_hr_nusa_16.jpg) | Aksara langka `ꦉ` (*Pa Cerek*) di luar skema Tier 1, namun sisa baris membaik `-6.06 pp`. |
| 16 | `hr_nusa_15` | `nusaaksara` · `real_printed_or_manuscript` | `51.61% → 35.48%` | `30.77%` | [16_hr_nusa_15.jpg](../eval/reports/phase5_worst20_imgs/16_hr_nusa_15.jpg) | Perbaikan `-16.13 pp`; karakter `dha mahaprana` (`ꦣ`) + `pada pangkat` (`꧇`) di luar skema kelas aktif. |
| 17 | `hr_nusa_01` | `nusaaksara` · `real_printed_or_manuscript` | `36.07% → 34.43%` | `34.62%` | [17_hr_nusa_01.jpg](../eval/reports/phase5_worst20_imgs/17_hr_nusa_01.jpg) | Kalimat panjang 61 grafem dengan pasangan beruntun. |
| 18 | `hr_nusa_11` | `nusaaksara` · `real_printed_or_manuscript` | `70.83% → 33.33%` | `46.67%` | [18_hr_nusa_11.jpg](../eval/reports/phase5_worst20_imgs/18_hr_nusa_11.jpg) | Perbaikan sangat besar `-37.50 pp` karena Stage-2 menekan prediksi ganda baris atas/bawah. |
| 19 | `hr_nusa_06` | `nusaaksara` · `real_printed_or_manuscript` | `41.67% → 29.17%` | `64.00%` | [19_hr_nusa_06.jpg](../eval/reports/phase5_worst20_imgs/19_hr_nusa_06.jpg) | Membaik `-12.50 pp` meski awalan/imbuhan rapat masih terpenggal. |
| 20 | `hr_seti_03` | `setilanaji` · `real_scanned_book_line` | `25.00% → 25.00%` | `50.00%` | [20_hr_seti_03.jpg](../eval/reports/phase5_worst20_imgs/20_hr_seti_03.jpg) | Kehilangan *layar* & *cecak* kecil akhir kata (`puter -> putêh`, `gotong -> gato`). |

---

## 7. Pra-Rilis Vercel, Ukuran Aset, & Waktu Muat Pertama (`eval/eval_phase5_prerelease.mjs`)

Diukur dengan server lokal yang menerapkan persis aturan [`vercel.json`](../vercel.json) (`outputDirectory: app/web_camscanner`, *rewrite* `/app/web_camscanner/*`, serta `Cache-Control: no-cache` pada `/sw.js`):

- **Versi Cache Service Worker (`sw.js`):** **`aksarascan-v17`** (`38` aset inti di deklarasi, `38` entri CacheStorage terverifikasi aktif untuk mode offline).
- **Total Ukuran Aset Statis (`42` berkas di `app/web_camscanner/`):** **`28.66 MB`** (`28.659.569` byte / `27.33 MiB`).

| Kategori Aset | Jumlah Berkas | Ukuran (Bytes) | Ukuran (MB / MiB) |
|---|---:|---:|---:|
| Bobot ONNX (`aksara_yolo.onnx` `12.70 MB` + `aksara_embed.onnx` `0.80 MB`) | `2` | `13.506.446` | `13.51 MB` (`12.88 MiB`) |
| Runtime ORT-Web 1.20.1 (`vendor/ort/`, termasuk `wasm` `11.25 MB`) | `3` | `11.716.934` | `11.72 MB` (`11.17 MiB`) |
| Bank Prototipe & Leksikon (`reference_prototypes.json`, `javanese_lm_data.js`, `model.*`) | `6` | `1.561.852` | `1.56 MB` (`1.49 MiB`) |
| Font Self-Hosted SIL OFL 1.1 (`vendor/fonts/`, 4 berkas `.ttf`) | `4` | `994.144` | `0.99 MB` (`0.95 MiB`) |
| Contoh Uji & Ikon PWA (`sample_1..4.png`, ikon) | `12` | `515.448` | `0.52 MB` (`0.49 MiB`) |
| Kode Aplikasi & Gaya (`index.html`, `styles.css`, modul JS) | `13` | `362.278` | `0.36 MB` (`0.35 MiB`) |
| Berkas Pelengkap (`sw.js`, dll.) | `2` | `2.467` | `0.00 MB` |

- **Waktu Muat Pertama Tanpa Cache (*Cold Load*, `localhost`):**
  - `TTFB`: **`14 ms`** · `DOMContentLoaded`: **`252 ms`** · `window.load`: **`260 ms`**
  - Siap Penuh (ORT-Web Stage-1 + Stage-2 176 Prototipe): **`1.046 ms`** (`30` request HTTP lokal, **`27.90 MB`** transfer awal)
  - Permintaan Jaringan Eksternal: **`0`** · Console/Page Error: **`0`**
  - *(Catatan deploy Vercel: preview URL publik Vercel terhubung ke push branch/main di remote GitHub.)*
