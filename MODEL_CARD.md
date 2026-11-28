# Model Card — AksaraScan v4.0

## Ringkasan Model & Arsitektur Hibrid
**AksaraScan** adalah aplikasi pemindai dan penerjemah Aksara Jawa **100% di peramban** (*client-side ONNX Runtime Web WASM*) dengan arsitektur pra-proses Worker, deteksi dua tahap, rekonstruksi ortografi Unicode, dan model bahasa lokal:

1. **Pre-Scan Adaptive Enhancer (`app/web_camscanner/image_enhancer.js`, `enhancer_worker.js`)**:
   - Berjalan off-thread melalui Web Worker (*Transferable* `ArrayBuffer`, `0` *long task* >= 100 ms di main thread).
   - Menilai tingkat ketajaman (`estimateBlur`: gradien kontras-ternormalisasi + lebar tepi Marziliano dwi-sumbu X/Y + simetri aksial), ink warna CIELAB (`coloredInkRatio`), serta kelurusan garis buku/dinding >= 55% lebar citra (`lineClutterScore`).
   - Melakukan *passthrough* nir-destruktif pada foto tajam (`MAE = 0`), perataan bayangan lokal, peredaman garis lurus panjang dengan perlindungan persimpangan `±3 px`, CLAHE adaptif terbatas pada latar redup, serta penajaman TV-bounded + Richardson-Lucy (dibatasi blur berat simetris).
2. **Detektor Tahap 1 (`app/web_camscanner/aksara_yolo.onnx`, `12.70 MB` FP32)**:
   - YOLOv8n `416 px`, `55` kelas Tier 1 (diperoleh dari baseline v3.1).
   - **Detektor Pengganti Bebas AGPL-3.0 siap GPU (`AksaraYOLOX-P2`)**: Arsitektur custom di [`training/yolox_model.py`](training/yolox_model.py) + [`training/train_yolox.py`](training/train_yolox.py) (**Apache-2.0 / MIT**, 0 `ultralytics`) + ekspor **FP16 Weight Storage ONNX** (`10.79 MB` @ `640 px` tervalidasi WASM `1228 ms` p50). Belum menjadi model produksi karena lingkungan lokal berstatus CPU-only (baru menjalankan *smoke test* 2-epoch); siap dilatih via [`training/AksaraScan_v4_GPU_MultiPlatform.ipynb`](training/AksaraScan_v4_GPU_MultiPlatform.ipynb).
3. **Stage-2 Reference Prototype Metric Verifier (`aksara_embed.onnx`, `0.80 MB` FP32)**:
   - Depthwise-Separable ConvNet + Squeeze-Excite + Spatial Quadrant Mixer dengan **Sub-Center Cosine Margin Head (3 sub-pusat × 56 kelas)**, dijalankan di browser via [`app/web_camscanner/stage2_verifier.js`](app/web_camscanner/stage2_verifier.js) (`1 ms` batch-1 / `13 ms` batch-16 di WASM).
   - Dilatih penuh di CPU pada `23.261` crop split `train` + `2.889` patch *hard negative* Latin/garis dinding (`split="train"`). Akurasi validasi terisolasi (`n=4.278`): **`90.51%`** tulisan tangan nyata, **`non_aksara_f1 = 92.62%`**.
   - **Bank Prototipe Referensi (`reference_prototypes.json`, `257.6 KB`)**: `176` centroid L2-ternormalisasi hasil Spherical K-Means dari **`6.623` contoh `split="train"`** saja (disediakan thumbnail `28×28` di Laci Inspektor UI), bebas total dari data `val` dan `heldout_real`.
   - **Kalibrasi Tersekat (`stage2_calibration.json`)**: Dibekukan eksklusif di split `val`, dilengkapi fallback otomatis ke Stage-1 bila berkas Stage-2 tidak tersedia.
4. **Jalur Penantang Baris Segmentation-Free (`Track B CRNN-CTC`)**:
   - Arsitektur CRNN + BiLSTM 2-lapis + CTC Loss (`9.86 MB`, `175 ms` WASM). Varian awal CPU pada `setilanaji_lines_v4 val` meraih CER mandiri `65.08%`, sehingga ditolak otomatis oleh gerbang `val` dan tidak disertakan di bundel web v4.0.
5. **Rekonstruktor Spasial, Validator Ortografi, & Mesin Bahasa**:
   - [`orthography.js`](app/web_camscanner/orthography.js) (memperbaiki bug `'ꦟ' ('\uA99F') → 'na'`, validasi ketat `U+A980..U+A9DF`, serta Dekoder Beam berbasis *lattice* kandidat `topK`).
   - [`javanese_lexicon.js`](app/web_camscanner/javanese_lexicon.js) + [`javanese_lm_data.js`](app/web_camscanner/javanese_lm_data.js) (`10.589` lema lokal).

---

## Ruang Kelas (`dataset/schema/classes_v4.yaml`)
- **Tier 1 (Aktif — 55 kelas Aksara + #55 Penolak Stage-2 = 56 kelas)**:
  - `20 Nglegena`, `20 Pasangan`, `12 Sandhangan & Pangkon`, `3 Tanda Baca Pada`
  - **Kelas #55 `non_aksara`** (Stage-2 saja): teks Latin cetak/tulis tangan, angka, garis buku tulis, grid, retakan dinding, serta `2.889` patch FP dari split `train`.
- **Tier 2 & Tier 3 (Ditunda di v4.0):** Aksara Murda, Aksara Swara mandiri, `ꦉ` (*Pa Cerek*), `ꦊ` (*Nga Lelet*), `꦳` (*Cecak Telu*), dan Angka Jawa (`꧐–꧙`) ditunda hingga tersedia minimal `100` kotak nyata dari `>= 5` penulis independen.

---

## Ringkasan Benchmark End-to-End Browser WASM ([`docs/BENCHMARK_V4.md`](docs/BENCHMARK_V4.md))

Diukur pada **Chromium `151.0.7922.34` (`ORT-Web 1.20.1`)**, bootstrap `B = 2000`, `seed = 42`, dengan kunci integritas `eval/heldout_real/SHA256SUMS.lock` (`43` berkas, `mismatches = 0`):

| Set Uji | `n` | CER `v3.1 (eb19337)` | CER `v4.0 Full` | GF1 `v3.1 → v4.0` | Latensi `v4.0` p50 |
|---|---:|---:|---:|---|---|
| **`eval/heldout_real/`** | `42` | `34.31%` `[26.26, 42.49]` | **`30.07%`** `[23.30, 37.17]` | `59.64% → 62.81%` | `947.4 ms` |
| **Subset Foto Buram Nyata** | `5` | `34.00%` `[12.00, 56.00]` | **`26.00%`** `[8.00, 44.00]` | `70.05% → 76.88%` | `1637.7 ms` |
| **`eval/failures/`** | `8` | `32.78%` `[4.73, 67.51]` | **`22.75%`** `[3.38, 45.83]` | `84.81% → 88.51%` | `1967.3 ms` |
| **Regresi Bersih (`synth_v2_val`)** | `40` | `21.42%` `[15.95, 27.04]` | **`21.06%`** `[15.58, 26.56]` | `82.06% → 83.45%` | `1253.9 ms` |
| **Distraktor Murni (`synth_v4_val_neg`)** | `29` | `7.69` FP/gambar | **`2.76` FP/gambar** | Bersih `27.6% → 44.8%` | `526.4 ms` |

---

## Privasi, Lisensi, & Keterbatasan
- **`0` permintaan jaringan eksternal** saat pemuatan halaman maupun inferensi (ORT-Web `1.20.1` dan `4` font SIL OFL 1.1 di-host mandiri di `app/web_camscanner/vendor/`); terjemahan daring baku **MATI**.
- **Kebersihan Lisensi ([`docs/AUDIT.md`](docs/AUDIT.md), [`dataset/LICENSES.md`](dataset/LICENSES.md)):** Seluruh jalur pelatihan v4 & artefak baru bebas font proprieter (`javatext.ttf` dieliminasi total) dan bebas kode AGPL-3.0; `NusaAksara` (`CC BY-SA 4.0`) digunakan ketat sebagai **HANYA-EVALUASI**. Detektor Stage-1 terpasang (`aksara_yolo.onnx`) masih berstatus transisi warisan v3.1 sampai bobot `AksaraYOLOX P2` (Apache-2.0) dilatih penuh di GPU.
- **Keterbatasan Utama:** Goresan pensil/pena `1 px` sangat tipis di kertas bergaris biru dapat terlewat oleh proposer Stage-1 (contoh `hr_hw_10`), sandhangan `< 12 px` (`wulu/cecak/layar`) pada foto naskah rapat dibatasi resolusi Stage-1 `416 px`, dan label Latin cetak tepat di atas aksara tetap rentan salah dibaca sebagai sandhangan vokal (`diag_08`).
