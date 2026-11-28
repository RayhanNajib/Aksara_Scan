# AksaraScan v4.0

Pemindai, verifikator ortografi, dan penerjemah Aksara Jawa **100% lokal di peramban** berbasis WebAssembly (`ONNX Runtime Web 1.20.1`), Web Worker, dan PWA offline (`aksarascan-v17`).

---

## Pembaruan Utama v4.0 (Changelog dari v3.1 → v4.0)

1. **Praproses Citra Adaptif Off-Thread (`image_enhancer.js`, `enhancer_worker.js`):**
   - Analisis otomatis tingkat ketajaman (`estimateBlur` Marziliano dwi-sumbu), garis buku/dinding memanjang >= 55% lebar citra, dan tinta berwarna dalam ruang CIELAB (`coloredInkRatio`) tanpa memblokir *main thread* (`0` *long task* >= 100 ms).
   - *Passthrough* aman pada foto tajam (`MAE = 0`), peredam bayangan, penekan garis lurus dengan perlindungan persimpangan `±3 px` (aman untuk garis buku biru), penguat goresan tipis `1 px` (`reinforceUltrathinInk` + masker proposal `scanMask` bersih), CLAHE adaptif terbatas, serta penajaman TV-bounded & Richardson-Lucy terarah.
2. **Stage-2 Reference Prototype Metric Verifier (`aksara_embed.onnx`, `0.80 MB`):**
   - Menguji ulang deteksi ambigu dan menyaring tulisan Latin, angka, label tabel di atas huruf, serta retakan latar lewat **56 kelas** (`55` kelas Aksara + kelas penolak `#55 non_aksara`) dan **`176` prototipe acuan** dari split `train` (`reference_prototypes.json`, `257.6 KB`).
   - Memangkas *false positive* halaman non-aksara di peramban dari `7.69` menjadi `1.52` kotak/gambar (`-80.2%`).
3. **Perbaikan Ortografi & Dekoder Beam (`orthography.js`):**
   - Memperbaiki peta vokal `'\uA99F'` (Na Murda → `'na'`), validasi blok Unicode `U+A980..U+A9DF`, serta pemanfaatan lattice kandidat `topK` di Dekoder Beam dengan leksikon lokal `10.589` lema.
4. **Kemandirian Offline 100% (`vendor/ort/` & `vendor/fonts/`):**
   - Mem-bundel lokal `onnxruntime-web@1.20.1` (`11.72 MB`) dan `4` font SIL OFL 1.1 (`Newsreader`, `Plus Jakarta Sans`, `Noto Sans Javanese` Regular/Bold, `0.99 MB`) dengan Service Worker cache **`aksarascan-v17`** (`38` aset) — **`0` permintaan jaringan eksternal** saat muat maupun pindai.
5. **Antarmuka Split Workbench, Inspektor Referensi, Batch, & Riwayat Lokal (`v4_features.js`, `DESIGN.md` v4.0.0):**
   - Layar awal ringkas tanpa panel pengaturan terbuka, beralih ke tampilan kerja `52% / 48%` lengkap dengan slider Before/After, zoom, rotasi 90°, koreksi kemiringan, **Inspektor Referensi Top-3** beserta thumbnail tulisan tangan, **Mode Batch Multi-Halaman** (`.txt/.md/.json`), riwayat lokal `IndexedDB`, serta kontras WCAG `>= 5.95:1` di tema Gelap & Terang.

---

## Hasil Benchmark Jujur (`v3.1` vs `v4.0`, Browser Chromium WASM)

Seluruh angka berikut diukur langsung pada perangkat lokal yang sama (`AMD Ryzen 5 5600H`, Windows 11, Chromium `151.0.7922.34`, `B = 2000` bootstrap 95% CI). Detail penuh, regresi minor, matriks konfusi, dan galeri 20 kasus terburuk dimuat di [`docs/BENCHMARK_V4.md`](docs/BENCHMARK_V4.md).

| Set Uji | `n` | CER `v3.1 → v4.0` | Grapheme F1 `v3.1 → v4.0` | Catatan |
|---|---:|---:|---:|---|
| **`eval/heldout_real/`** (Terkunci SHA-256) | `42` | `34.31% → 28.68%` (`-5.63 pp`) | `59.64% → 63.52%` (`+3.88 pp`) | Latensi median p50 **`957.1 ms`** (`hr_hw_10`: `100% → 10%`) |
| **Subset Foto Buram Nyata** (`heldout_real`) | `5` | `34.00% → 30.00%` (`-4.00 pp`) | `70.05% → 75.93%` (`+5.88 pp`) | Blur defokus & gerak nyata |
| **`eval/failures/`** (Set Diagnostik) | `8` | `32.78% → 2.01%` (`-30.77 pp`) | `84.81% → 98.72%` (`+13.91 pp`) | `diag_01` `0%`, `diag_07` `11.11%`, `diag_08` `5.00%` |
| **Regresi Bersih (`synth_v2_val_clean`)** | `40` | `21.42% → 20.67%` (`-0.75 pp`) | `82.06% → 82.79%` (`+0.73 pp`) | Word Acc naik (`12.43% → 12.67%`) |
| **Distraktor Murni (`synth_v4_val_neg`)** | `29` | `7.69 → 1.52` FP/gambar | Bersih `27.6% → 41.4%` | Penolakan Latin/garis tanpa Aksara (`-80.2%` FP) |

---

## Dokumentasi, Dataset, Lisensi, & Pelatihan

- [Laporan Benchmark & Kesalahan v4.0 (`docs/BENCHMARK_V4.md`)](docs/BENCHMARK_V4.md) & [Galeri 20 Kegagalan Terburuk](eval/reports/phase5_worst20.html)
- [Kartu Model (`MODEL_CARD.md`)](MODEL_CARD.md) & [Sistem Desain (`DESIGN.md`)](DESIGN.md)
- [Statistik Dataset & Anti-Kebocoran (`docs/DATASET_V4.md`)](docs/DATASET_V4.md) & [Audit Kebersihan (`21.457` rekaman)](tools/check_leakage.py)
- [Audit Lisensi & Resolusi v4.0 (`docs/AUDIT.md`)](docs/AUDIT.md) · [Inventaris Lisensi (`dataset/LICENSES.md`)](dataset/LICENSES.md) · [Kredit Dataset (`DATASET_CREDITS.md`)](DATASET_CREDITS.md)
- [Log Progres Kumulatif (`docs/V4_PROGRESS.md`)](docs/V4_PROGRESS.md) & [Alternatif Pelatihan GPU Non-Colab (`training/GPU_ALTERNATIVES_RUNBOOK.md`)](training/GPU_ALTERNATIVES_RUNBOOK.md)

---

## Menjalankan & Memverifikasi Secara Lokal

Menjalankan aplikasi web (`app/web_camscanner/` memiliki total `42` berkas statis / `28.66 MB` sesuai `vercel.json`):

```bash
cd app/web_camscanner
npx serve .
```

Menjalankan rangkaian verifikasi otomatis Fase 5:

```bash
node eval/test_e2e_assertions.mjs
node eval/test_fase3_enhancer.mjs
node eval/test_fase4_ui.mjs
node eval/eval_phase5_prerelease.mjs
python tools/check_leakage.py
```
