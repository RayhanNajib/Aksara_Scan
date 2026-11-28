# Log Kumulatif Progres AksaraScan v4.0 (`docs/V4_PROGRESS.md`)

> Berkas ini dibaca pada awal setiap sesi dan diperbarui di akhir setiap sesi sebagai rekam jejak terverifikasi untuk transisi AksaraScan v3.1 → v4.0.

---

### 2026-11-27 — Pra-Flight & Baseline v3.1 — Verifikasi Awal (Read-Only)
- **Dikerjakan:**
  - Audit status repositori pada branch `main`, commit baseline **`eb19337`** (`chore(icon): replace decorative favicon with clean vector ha (U+A9B2) glyph`).
  - Memeriksa `README.md`, `docs/AUDIT.md`, `MODEL_CARD.md`, `DATASET_CREDITS.md`, `DESIGN.md`, `app/web_camscanner/`, `tools/`, `training/`, dan `eval/`.
  - Menjalankan baseline v3.1 lengkap tanpa memodifikasi kode/bobot:
    1. `node Aksara_Scan/eval/test_e2e_assertions.mjs`
    2. `python eval/run_eval.py --parity eval/parity_20.jsonl --failures eval/failures/manifest.jsonl --yolo-val dataset/external/synth_v3` (keluaran sementara di direktori scratch agar berkas laporan repo tidak tertimpa sebelum persetujuan Fase 1).
  - Memverifikasi lingkungan komputasi lokal: Python `3.10.0`, PyTorch `2.13.0+cpu`, **`torch.cuda.is_available() == False` (CPU only)**.

- **Hasil terukur Baseline v3.1 (Commit `eb19337`, `aksara_yolo.onnx` = `12.11 MB`):**
  - **Uji E2E Chromium (`eval/test_e2e_assertions.mjs`):** `PASS` pada `02_wedang_bubuk_crop.png` (`1156 ms`), `sample_4.png` (`558 ms`), tema Gelap/Terang (`184`/`192`), `0` console error, `0` request jaringan eksternal saat inferensi.
  - **Set Diagnostik (`eval/failures/manifest.jsonl`, `n=8`):**
    - Rata-rata Unicode CER: **`32.78%`** · Rata-rata Grapheme F1: **`84.81%`** · Rata-rata Word Accuracy: **`27.71%`**.
    - Kasus gagal utama: `diag_01` (ada teks Latin di atas aksara: CER `37.84%`, GraF1 `65.31%`), `diag_07` (CER `133.33%`, GraF1 `64.29%`), `diag_08` (CER `80.0%`, GraF1 `54.17%`).
  - **Validasi `dataset/external/synth_v3/val` (`n=278` citra, `8.158` GT boxes, `52` kelas aktif pada `imgsz=416`):**
    - Overall **`91.67% mAP50`** / **`74.70% mAP50-95`** (Nglegena `92.65%`, Pasangan `92.48%`, Pada `90.87%`, **Sandhangan `88.80% mAP50` / `67.85% mAP50-95`**).
    - **Objek kecil `< 16 px` (`gt_416 = 1811`):** recall hanya **`77.03%`** (`tp=1395`).
    - Konfusi top: `pepet ↔ wulu` (28), `nya ↔ ba` (16), `cecak ↔ wulu` (15), `tha ↔ nga` (20 dua arah), `na ↔ ka` (9), `ca ↔ sa` (9), `da ↔ na` (9), `ta ↔ ha` (8), `wa ↔ pa` (8), `ma ↔ sa` (8).
  - **Paritas `eval/parity_20.jsonl` (`n=20`):**
    - Paritas tensor identik maks selisih `0.001465`; namun pra-proses JS (`prepareYoloInputCanvas` Unsharp/IllumNorm) vs Python `cv2` menghasilkan selisih output ONNX mentah hingga `135.22`, kecocokan kotak IoU≥0.70 = `490/561`, dan kecocokan string Unicode persis browser = `7/20`.

- **Selisih brief vs kenyataan (Verifikasi Bagian 4):**
  1. **Jumlah font (`dataset/external/fonts/`):** Ada **12 berkas font** (`10` keluarga unik; `NotoSansJavanese` punya Regular + Bold sehingga 11 varian gaya, `TuladhaJejegOT-Regular` ada format `.ttf` + `.otf`). Seluruh 12 berkas terverifikasi berlisensi terbuka di header TTF/OTF (`11` SIL OFL 1.1 + `1` Apache-2.0 `Sehulbari_Arif_Budiarto.ttf`). `javatext.ttf` tidak ada di repo.
  2. **Jumlah Aksara Murda:** Blok Unicode standar `U+A980..U+A9DF` memuat **9** huruf bernama `MURDA` (`Ka U+A991`, `Ga U+A993`, `Ca U+A996`, `Nya U+A998`, `Na U+A99F`, `Ta U+A9A1`, `Pa U+A9A6`, `Ba U+A9A8`, `Sa U+A9AF`), sementara pengajaran Carakan umumnya memakai 7–8 huruf dan dataset `RayFox52/Aksara-jawa-carakan-murda` berisi `27` kelas (`20` Nglegena + **`7` Murda**: `ba, ga, ka, na, pa, sa, ta`). **Temuan bug nyata di `app/web_camscanner/orthography.js:52-53`:** `'\uA99F'` (Na Murda) keliru dipetakan ke `'da'` (seharusnya `'na'`), dan `'\uA998'` (Nya Murda) ke `'ja'`.
  3. **Pemakaian `setilanaji/aksara-jawa-ocr`:** Tersedia **`2.209` baris** di `dataset/external/line_ocr/setilanaji/index.jsonl` (`2.319` berkas mentah), dan benar tepat **`67` citra train** (`134` berkas `.png`+`.txt`) dipakai di `converted_yolo/setilanaji_teacher/`.
  4. **Lokasi aktual di kode:**
     - `prepareYoloInputCanvas`: `app/web_camscanner/app.js:580-719` (sesuai).
     - `topK`: `app/web_camscanner/app.js:907` dan `943` (berdekatan dengan rentang 900–945).
     - Panel kiri: `app/web_camscanner/index.html:96-220`; `wayang-corner`: `app/web_camscanner/index.html:130-133` (persis sesuai).
     - **Temuan pelanggaran Aturan Besi #2:** `app/web_camscanner/index.html:435` masih memuat `onnxruntime-web@1.20.1/dist/ort.min.js` dari `cdn.jsdelivr.net` dan `sw.js` belum mem-precache aset runtime ORT/WASM secara lokal (sehingga berisiko gagal saat mode offline penuh tanpa cache CDN sebelumnya). Also `Aksara_Scan/DESIGN.md` v3.0.0 sudah ada di root repo.
  5. **Kodepoint kelas baru (`U+A980..U+A9DF`):** Tervalidasi via Python `unicodedata`: `Pa Cerek` (`ꦉ U+A989`), `Nga Lelet` (`ꦊ U+A98A`), `Swara` (`ꦄ U+A984, ꦆ U+A986, ꦈ U+A988, ꦌ U+A98C, ꦎ U+A98E`), `Rekan / Cecak Telu` (`꦳ U+A9B3`), dan `Angka Jawa` (`꧐–꧙ U+A9D0..U+A9D9`) semuanya sah berada dalam blok `U+A980..U+A9DF`.
  6. **Status sumber eksternal:**
     - Hugging Face API OK: `setilanaji/aksara-jawa-ocr` (`apache-2.0` eksplisit); `NusaAksara/NusaAksara` (`license: None` pada metadata); `RayFox52/Aksara-jawa-carakan-murda` (`license: None` pada metadata); `hermanshid/yolo-aksara-jawa` (`license: None`).
     - GitHub API OK: `hermansh-id/Aksara-Jawa-OCR` (`MIT`), `IqbalLx/Hanacaraka-AI` (`license: None`).
     - Mendeley Data OK (`HTTP 200`): `vfj32bpjsf/1` (sudah lokal di `dataset/indonesian-scripts/`) & `ms6sjjpzwr/1` (`CC BY 4.0`, mengandung arsip British Library EAP yang perlu cek ketentuan).
     - Roboflow Universe (`pa-relate/aksara-jawa`, `febgabut/aksara-jawa-classification`, dll.): mengembalikan `HTTP 403` untuk akses tanpa sesi/API key; perlu unduhan manual berizin sesuai `tools/datasets/MANUAL.md`.
  7. **Skill desain UI:** Seluruh 8 skill (`impeccable`, `claude-design`, `popular-web-designs`, `design-md`, `humanizer`, `frontend-design`, `dogfood`, `adversarial-ux-test`) terpasang dan aktif di `C:\Users\WA\.agents\skills\`.

- **Keputusan:**
  - Belum ada perubahan kode/model/dataset; menunggu persetujuan Rayhan untuk mengeksekusi **Fase 1** pada branch `v4/fase-1-dataset`.
- **Berikutnya:**
  - Menunggu konfirmasi Rayhan: (a) izin memulai Fase 1 (`v4/fase-1-dataset`), (b) apakah perlu menyiapkan API key/unduhan manual Roboflow/Kaggle atau mulai dari sumber terbuka terverifikasi terlebih dahulu, dan (c) kebijakan terhadap sumber tanpa tag lisensi eksplisit (`NusaAksara` untuk evaluasi-saja vs `DIBLOKIR`).

---

### 2026-11-27 — Fase 1 (`v4/fase-1-dataset`) — Ekspansi Dataset, Skema v4, Anti-Kebocoran, & Penolak Latin/Multi-Tinta
- **Dikerjakan:**
  - Dibuat branch `v4/fase-1-dataset` dari `eb19337`.
  - Diaudit lisensi seluruh sumber data, 12 font Aksara Jawa + 3 font Latin SIL OFL, korpus teks, pustaka model, dan rujukan teori CV (Tanaka et al., Gold & Zesch, SWT Epshtein et al., CIELAB color separation, TPS) di `dataset/LICENSES.md` dan `docs/REFERENCES.md` Bagian 5.
  - Diterbitkan skema kelas bertingkat `dataset/schema/classes_v4.yaml` + `tools/datasets/schema_v4.py` (55 kelas Tier 1 aktif penuh; Tier 2/3 ditunda otomatis karena < 100 kotak nyata dari >= 5 penulis; kelas penolak `non_aksara_distractors` disiapkan untuk Stage-2).
  - Dibangun & dijalankan pipeline ingest terpisah per sumber (`tools/datasets/ingest_hanacaraka_v4.py`, `ingest_jawa_zip_v4.py`, `ingest_setilanaji_v4.py`, `ingest_nusaaksara_eval.py`, `ingest_roboflow.py`, `ingest_kaggle.py`). Kebocoran historis 79 penulis di `hanacaraka` & 2 duplikat lintas-split `setilanaji` dihilangkan penuh.
  - Dibangun & dijalankan generator sintetis `tools/generate_v4_elastic_hw.py` (**TPS Elastic Warp**, modulasi ketebalan goresan, rotasi ±8°, shear ±4,5°, PSF defocus/motion blur, pulpen biru/merah/hijau/pensil/dark-mode, latar garis buku/retakan tembok/grid/lipatan, teks Latin campuran + pure hard negative + bank patch penolak `non_aksara_distractors_v4`).
  - Dibangun & dikunci set uji `eval/heldout_real/` (`42` citra, `26` penulis/dokumen unik, terkunci `eval/heldout_real/SHA256SUMS.lock`), diaudit penuh oleh `tools/check_leakage.py`, dan dirangkum otomatis di `docs/DATASET_V4.md`.

- **Hasil terukur (Gerbang 1):**
  - `check_leakage.py` (`18.568` rekaman lintas 8 manifes + `eval/failures` + `heldout_real` + sampel app): **`PASS_ZERO_LEAKAGE`** (`writer_overlap=0`, `cluster_overlap=0`, `sha256_overlap=0`, `near_duplicate_cross_split=0`, exit code `0`).
  - `eval/heldout_real/SHA256SUMS.lock`: **`43` berkas diverifikasi, `mismatches=0`**.
  - Reprodusibilitas `ingest_hanacaraka_v4.py --seed 42`: dua kali eksekusi menghasilkan manifes ber-SHA256 identik (`c60bc8d1d1e1a147...`).
  - Distribusi total v4: `hanacaraka_v4` (`1.085 tr / 239 va / 238 te` dari 55/12/12 penulis unik), `jawa_zip_v4` (`9.426 tr / 1.634 va`), `jawa_syllables_v4` (`1.248 tr / 221 va`), `setilanaji_lines_v4` (`2.059 tr / 132 va / 16 te`), `nusaaksara_eval_only_v4` (`48` `HANYA-EVALUASI`), `synth_v4` (`1.050` citra, `22.794` kotak mencakup **55/55 kelas**, plus `1.050` patch `non_aksara_distractors_v4`).
  - Seluruh contact sheet (`v4_contact_hanacaraka.jpg`, `v4_contact_jawa_zip.jpg`, `v4_contact_syllables.jpg`, `v4_contact_nusaaksara_eval.jpg`, `v4_contact_synth_tps.jpg`, `v4_contact_synth_multiink_wall.jpg`, `v4_contact_synth_hardneg_latin.jpg`, `v4_contact_heldout_real.jpg`) telah diperiksa secara visual.

- **Keputusan:**
  - Kelas Tier 2 & Tier 3 tetap `deferred` pada Fase 1 sampai ada setidaknya 100 kotak beranotasi nyata dari >= 5 penulis independen; `NusaAksara` (`CC BY-SA 4.0`) dipakai ketat sebagai `HANYA-EVALUASI`.
  - Siap masuk Gerbang 1 untuk review Rayhan sebelum **Fase 2 (`v4/fase-2-model`)**.
- **Berikutnya:**
  - Meminta persetujuan Rayhan pada Gerbang 1 & menentukan jalur GPU / target metrik awal untuk Fase 2 (`YOLOX P2` + `aksara_embed.onnx` + `reference_prototypes.json` + ablasi Track B).

---

## Fase 2 — Pelatihan Track A1/A2, Bank Prototipe Referensi, & Ablasi C0..C3 (Branch `v4/fase-2-model`)
- **Dikerjakan (ringkas):**
  - **Kompatibilitas Chromium ORT-Web 1.20.1 WASM** (`eval/bench_phase2_wasm.mjs`, `eval/reports/phase2_wasm_bench.json`): seluruh model berjalan mulus di browser single-thread (`coi=false`, 0 op error), termasuk skema konversi murni-ONNX **FP16 Weight Storage** (`CAST` di awal graf tanpa `onnxconverter-common`): `yolox-tiny P2 640px` menyusut menjadi **`10.79 MB`** (`p50=1228 ms`), `960px` menjadi **`11.09 MB`** (`p50=2768 ms`), varian INT8 `640/960` (`5.74 / 6.07 MB`, `791 / 1765 ms`), `yolox-nano P2 640px` (`3.95 MB`, `285 ms`), serta **`aksara_embed.onnx` (`0.77 MB`, `1 ms` batch 1 / `13 ms` batch 16)**.
  - **Peningkatan Trainer Detektor YOLOX P2 & Bundel GPU Multi-Platform (Pengganti Colab)**: `training/train_yolox.py` & `training/yolox_model.py` ditingkatkan agar menerima manifes `v4`, sampel negatif murni `0-box` (Latin/garis tembok), bobot kurikulum bertahap, pengaman batch `num_fg=0`, log `history.json`, serta konversi FP16/INT8. Dijalankan *smoke test* CPU nyata 2 epoch (`runs_yolox/v4_smoke/`, `max|PT-ORT|=6.10e-5`, `FP16W=10.79 MB`). Dibuat pula jalur pelatihan GPU bebas Google Colab (`training/GPU_ALTERNATIVES_RUNBOOK.md`, `training/AksaraScan_v4_GPU_MultiPlatform.ipynb`, dan `tools/build_gpu_bundle.py`: `208.1 MB` raw `train+val`, mengecualikan `254` data `heldout`).
  - **Penambangan Hard Negative Persegi & Pelatihan Penuh Stage-2 Metric Verifier (CPU 12-core)**: ditambang `2.889` crop persegi dari FP detektor `split="train"` + karakter Latin SIL OFL (`tools/datasets/mine_stage2_hard_negatives.py`; diaudit ulang oleh `check_leakage.py` across `21.457` rekaman: `PASS_ZERO_LEAKAGE`). `training/train_stage2_embedder.py` melatih model `56` kelas (`16` epoch, `23.261` crop latih) hingga konvergen (`app/web_camscanner/aksara_embed.onnx`, `0.77 MB`), lalu `training/build_reference_prototypes.py` membangun `app/web_camscanner/reference_prototypes.json` (`176` prototipe across `56` kelas, `251.5 KB`, 100% dari split `train`).
  - **Kalibrasi Tersekat di `val` & Uji Ablasi Bootstrap `eval/eval_phase2_ablation.py`**: `stage2_calibration.json` dibekukan semata pada `val` (bebas dari `writer_id` heldout) lalu diuji pada `eval/heldout_real/` (`n=42`), `eval/failures/` (`n=8`), dan regresi bersih `synth_v2 val` (`n=40`).

- **Hasil terukur (Gerbang 2):**
  - **Stage-2 Val Terisolasi (`n=4.278` crop `val`)**: akurasi blended Aksara **`85.73%`**, akurasi tulisan tangan nyata **`90.51%`** (`hw_syllables` **`94.29%`**), **`non_aksara_f1` = `92.62%`** (`recall 97.52%`, `neg_latin_sq_win` `97.24%`, `neg_fullpage_subcrop` `97.41%`).
  - **Kalibrasi `val`**: `NegFP/gambar` pada halaman negatif murni turun dari `9.21` (`C0`) -> **`3.83` (`C2`)** (`-58.4%`); akurasi kotak multi-domain naik `69.12% -> 69.94%`.
  - **`eval/heldout_real/` (`n=42`, `SHA256SUMS.lock` `43` berkas, `mismatches=0`)**:
    - `C0 (v3.1)`: `CER = 37.36%` `[30.48, 44.70]`, `GF1 = 53.50%` `[44.75, 61.68]`, `BoxClsAcc = 63.33%`
    - **`C2 (+ Stage-2 & Prototipe)`**: **`CER = 34.48%`** `[28.06, 40.99]` (**`-2.88%`**), **`GF1 = 55.97%`** `[47.90, 63.60]` (**`+2.47%`**), **`BoxClsAcc = 64.17%`**; `nusaaksara` turun `CER 51.96% -> 46.62%` (`-5.34%`).
    - `C1 (YOLOX 2-epoch CPU smoke)`: `CER = 100.00%` (validasi pipa & ekspor ONNX saja; bobot penuh menunggu eksekusi GPU cloud).
    - `C3 (+ Track B CRNN existing)`: `CRNN` mandiri `val CER = 65.08%`, gerbang menolak penggantian sehingga `C3 == C2`.
  - **`eval/failures/` (`n=8`)**: `C0` (`CER 27.81% / GF1 85.67%`) -> **`C2` (`CER 20.24% / GF1 86.92%`, `-7.57%` CER)** dengan peningkatan besar pada `handwriting_chart_with_latin_labels` (`GF1 68.18% -> 77.27%`) dan `handwriting_latin_distractor_border` (`CER 155.56% -> 100.00%`).
  - **Penjaga Regresi `synth_v2/val` bersih (`n=40`)**: `C0` (`CER 23.16% / GF1 81.94%`) -> `C2` (`CER 23.54% / GF1 81.81%`, delta `< 0.4%`).

- **Keputusan & Berikutnya (Gerbang 2):**
  - Berhenti di **Gerbang 2** untuk melaporkan hasil kepada Rayhan, menyerahkan paket notebook GPU alternatif non-Colab, dan meminta konfirmasi target metrik perbaikan sebelum integrasi penuh JS/enhancer **Fase 3 (`v4/fase-3-enhancer`)**.

---

## Fase 3 — Adaptive Enhancer Worker, Stage-2 Verifier Browser, ORT Lokal, & Mode Offline (Branch `v4/fase-3-enhancer`)
- **Dikerjakan (ringkas):**
  - **Kemandirian Aset Offline (Zero-CDN Runtime):** Memindah-bundel ONNX Runtime Web `1.20.1` ke lokal `app/web_camscanner/vendor/ort/` (`ort.min.js`, `ort-wasm-simd-threaded.mjs`, `ort-wasm-simd-threaded.wasm`), memperbarui `index.html` memakai aset lokal saja, mengatur `ort.env.wasm.wasmPaths = new URL('./vendor/ort/', window.location.href).href`, serta meningkatkan Service Worker `sw.js` ke cache **`aksarascan-v14`** (`33` aset inti + bobot ONNX + prototipe referensi di-precache).
  - **Engine Praproses Adaptif Off-Thread (`image_enhancer.js` + `enhancer_worker.js` + `app.js`):**
    - `estimateBlur` berbasis lereng gradien kontras-ternormalisasi + lebar transisi tepi kontigu Marziliano dwi-sumbu X/Y + simetri aksial, pemisahan tinta warna oponen CIELAB (`coloredInkRatio`), detektor garis lurus tipis berkesinambungan `>= 55%` lebar (`lineClutterScore`, anti salah-picu bayangan lebar), dan kuantil tinta tipis `0.2%` (`p002`) untuk naskah jarang.
    - `enhance`: *passthrough* nir-destruktif pada foto tajam & bersih (`mode: 'passthrough_sharp'`, `MAE = 0`); perata pencahayaan lokal (`flattenShadow`); penekan garis buku horizontal (`suppressLongStraightLines` dengan perlindungan persimpangan `±3 px` & opsi `removeVertLines`); CLAHE adaptif terbatas pada latar redup/kontras rendah; serta penajaman TV-bounded `[locMin, locMax]` dengan *noise coring*, perlindungan inti goresan adaptif (`inkProtectThr`), pembobotan penggelapan asimetris inti tinta, dan pembatasan Richardson-Lucy pada blur berat simetris (`berat`). Berjalan via `enhancer_worker.js` (*Transferable* `ArrayBuffer`) dengan pelindung `try/catch` fallback sinkron di `app.js`.
  - **Stage-2 Browser Verifier (`stage2_verifier.js`) & Upgrade `orthography.js`:**
    - Memuat `app/web_camscanner/aksara_embed.onnx`, `app/web_camscanner/reference_prototypes.json` (`176` prototipe), dan `app/web_camscanner/stage2_calibration.json` secara aman dengan *fallback* otomatis ke mode detektor Tahap-1 saja bila salah satu berkas gagal dimuat.
    - `orthography.js` memperbaiki pemetaan `'\uA99F'` (Na Murda) ke `'na'`, memutakhirkan `refineLineWithConservativeBeam` untuk memakai lattice `topK` + `pasOverride` + `isUncertain`, serta mengekspos `isUncertain`, `topK`, dan `referenceMatches` pada tiap cluster.

- **Hasil terukur (Gerbang 3 — `eval/reports/phase3_verification.json`):**
  - **Uji Unit `image_enhancer.js` (`node eval/test_fase3_enhancer.mjs [1/4]`):**
    - Progresi `estimateBlur` Gaussian (`tajam 1.0 > sigma 1.0 ringan 0.5916 > sigma 1.8 sedang 0.2634 > sigma 3.0 berat 0.0188`) dan Motion (`k5 sedang 0.2559 > k9 berat 0.1784`) monoton.
    - Foto tajam: `mode: 'passthrough_sharp'`, `MAE = 0`.
    - Kontinuitas goresan tipis buram (`sigma = 1.2`): komponen terhubung `12 -> 12`, `inkPixels = 1820 -> 1820`, `ridgeCount = 720 -> 720`, ketajaman naik (`blurAfter.sharpnessScore > blurBefore.sharpnessScore`).
    - Pulpen biru + garis buku sintetis: `suppressedPx = 630`, `coloredInkRatio = 0.0589`.
  - **Uji Browser Chromium (`[2/4]..[4/4]` + `eval/test_e2e_assertions.mjs`):**
    - Web Worker `1200×900` (`useRL: true`): selesai dalam `393 ms` dengan **`0` main-thread long task >= 100 ms** (`longTasks: []`).
    - Komparasi blur (`raw` vs `enhanced`): mean CER stabil `15% -> 15%` (`hr_hw_04_defocus_blur`: `0% -> 0%`, `diag_04_shadow`: `0% -> 0%`, `diag_06_lowres600`: `0% -> 0%`, `hr_hw_02_motion_blur`: `60% -> 60%` — tidak memburuk pada *motion blur* nyata sekaligus bebas amplifikasi noise).
    - Fallback Stage-2 gagal muat (`serviceWorkers: 'block'` + abort `aksara_embed.onnx`/`reference_prototypes.json`): status tampil `Stage-1 Fallback (Stage-2 tidak tersedia)`, `stage2Stats.active = false`, `sample_4.png` tetap terbaca `"aksara jawa mangan sêga krama kothi"`.
    - Mode Offline PWA penuh (`sw.js aksarascan-v14`, `33` entri di cache, `setOffline(true)` + reload): status `YOLOv8 Siap · 92.18% mAP50 + Stage-2 (176 prototipe)`, inferensi `02_wedang_bubuk_crop.png` berhasil 100% offline (`"wédang bubuk tanpa gula, aja ngamuk mundhak tuwa."`).
    - E2E Assertions (`eval/test_e2e_assertions.mjs`): **`0` console error, `0` request jaringan eksternal saat inferensi**, latensi `02_wedang_bubuk_crop.png = 1205 ms`, `sample_4.png = 1178 ms`.
  - **Lock integritas held-out:** `eval/heldout_real/SHA256SUMS.lock` **`43` berkas diverifikasi, `mismatches=0`**.

- **Keputusan & Berikutnya (Gerbang 3):**
  - Seluruh praproses adaptif Worker, Stage-2 Verifier browser, aset ORT lokal, dan PWA offline telah lolos seluruh gerbang teknis. Berhenti di **Gerbang 3** untuk meminta persetujuan Rayhan sebelum memulai **Fase 4 (`v4/fase-4-ui`: Perombakan UI/UX Anti-Slop bergaya ScanRead + Inspector Referensi & Koreksi)**.

---

## Fase 4 — Perombakan UI/UX (ScanRead-Inspired + Anti-Slop) (Branch `v4/fase-4-ui`)
- **Dikerjakan (ringkas):**
  - **Self-Hosted Font SIL OFL (`app/web_camscanner/vendor/fonts/`):** Mengganti CDN Google Fonts dengan self-hosting lokal (`Newsreader-Regular.ttf`, `PlusJakartaSans-Regular.ttf`, `NotoSansJavanese.ttf`, `NotoSansJavanese-Bold.ttf`, total ~994 KB), memperbarui `sw.js` ke **`aksarascan-v15`** (`38` aset di cache), sehingga pemuatan halaman maupun pemindaian menghasilkan **`0` permintaan jaringan eksternal**.
  - **Sistem Desain & Token WCAG AA+ (`DESIGN.md` v4.0.0 & `styles.css`):** Mempertahankan palet Malam Keraton (`dark`) & Kertas Dluwang (`light`), menaikkan keterbacaan teks sekunder & placeholder ke `>= 5.95:1` di kedua tema, serta men-temakan permukaan peramban (`::selection`, scrollbar, `:focus-visible`, `caret-color`, `tabular-nums`). Menghapus penuh `.wayang-corner`, *kicker* kapital, teks gradasi, kartu bertumpuk, dan border samping `> 1px`.
  - **Arsitektur Progressive Disclosure & Split Workbench (`index.html`, `app.js`):**
    - Topbar sticky 64 px (`Pemindai`, `Batch`, `Lab & Referensi`, `Tentang` + status engine + tombol `Pindai Baru` + pengalih tema) + footer 1 baris.
    - Layar awal hanya menampilkan hero ringkas, dropzone luas (`PNG · JPG · WebP · Foto Buram · Tinta Warna · Mode Gelap`), 4 contoh uji, riwayat pindai lokal, dan 3 langkah proses nyata (tanpa kartu hasil kosong atau pengaturan terbuka).
    - Setelah gambar masuk, beralih halus ke Split Workbench `52% / 48%` dengan micro-toolbar (*Kotak Deteksi*, *Bandingkan Asli/Ditajamkan Before/After*, *Biner*, *Putar 90°*, *Luruskan*, *Zoom*), panel kontrol *Praproses Citra* yang terhubung langsung ke `image_enhancer.js`, serta pesan pemulihan ramah untuk kanvas kosong/buram berat.
  - **Fitur Terkendali (`v4_features.js` + `stage2_verifier.js`):**
    - **Inspektor Referensi Stage-2**: klik kotak atau kartu glif menampilkan potongan pengguna, skor kemiripan acuan (`"Cocok Ref pepet · tulisan tangan (90%)"`), Top-3 kandidat lengkap dengan thumbnail `handwritten` & `printed_tps`, pencarian 55 kelas, dan riwayat koreksi lokalStorage.
    - **Mode Batch Multi-Halaman**: antrean multi-gambar memakai fungsi inferensi bersama (`runYoloInferenceCore` / `processImageFileBatch`), ekspor `.txt`, `.md`, `.json`, dan `Salin Semua`.
    - **Riwayat Pindai Lokal (`IndexedDB` `aksarascan_history_db`)**: menyimpan hingga 12 riwayat terakhir (termasuk pulihkan & hapus 1-klik).
    - **Penjelajah Bank Prototipe (56 Kelas)** & **Tab Tentang (Mode Read)**: menampilkan 176 prototipe acuan per kelas serta atribusi dataset/font terbuka (`CC BY 4.0`, `CC BY-SA 4.0`, `Apache-2.0`, `MIT`, `SIL OFL 1.1`).

- **Hasil terukur (Gerbang 4 — `eval/reports/phase4_verification.json` & tangkapan layar `eval/artifacts/fase4_screenshots/`):**
  - **Matriks `1440 px` & `390 px` × `Dark` & `Light` (`eval/test_fase4_ui.mjs`):**
    - Kontras minimum token: **`5.95:1`** (Dark) dan **`6.03:1`** (Light); kontras minimum elemen aktif live DOM: **`8.16:1`** (Dark) dan **`6.40:1`** (Light).
    - Audit Anti-Slop di seluruh sel matriks: `.wayang-corner = 0`, `gradientText = 0`, `uppercaseHeaders = 0`, `nestedCards = 0`, `thickBorders = 0`.
    - Overflow horizontal halaman `390 px` (termasuk setelah injeksi kata OOV 52 huruf): **`0 px`**, dan keempat pill navigasi muat bersih tanpa terpotong.
    - Fungsional terverifikasi: progressive disclosure (`colRatio = 0.52`), slider Before/After compare + Zoom (`120%`), Inspektor Referensi Top-3 (`6` thumbnail), Bank Prototipe (`56` kelas), Batch `2/2` selesai + ekspor, IndexedDB history tersimpan & pulih, kasus ekstrem `blank.png` (`emptyRecovery`) dan `2600×1800` (`1500×1038 px · skala dari 2600×1800`), serta `Tab` keyboard focus (`outlineWidth = 2px solid`).
  - **Regresi Pasca-Edit Akhir (`eval/test_e2e_assertions.mjs` + `eval/test_fase3_enhancer.mjs`):**
    - `test_e2e_assertions.mjs`: **PASS** (`02_wedang_bubuk_crop.png` `1207 ms`, `sample_4.png` `1078 ms`, kontras tema konsisten, `0` console error, `0` request jaringan eksternal).
    - `test_fase3_enhancer.mjs`: **PASS `[1/4]..[4/4]`** (Worker `418 ms` `0` long task, blur CER `15% -> 15%`, fallback Stage-2, mode offline penuh `sw.js aksarascan-v15` `38` entri cache `stage2Ms = 49`).
  - **Tangkapan layar terinspeksi visual:** `01_landing_1440_dark.png`, `02_workbench_1440_dark.png`, `03_inspector_1440_dark.png`, `04_workbench_1440_light.png`, `05_landing_390_dark.png`, `06_workbench_390_dark.png`, `07_workbench_390_light.png`.

- **Keputusan & Berikutnya (Gerbang 4):**
  - Berhenti di **Gerbang 4** untuk mempresentasikan hasil perombakan UI/UX dan meminta persetujuan Rayhan sebelum memulai **Fase 5 (`v4/fase-5-eval`: Audit Otomatis, Benchmark Jujur v3.1 vs v4.0, Analisis Kesalahan, Regresi, & Pra-Rilis Vercel)**.

---

## Fase 5 — Audit Akhir, Benchmark Browser v3.1 vs v4.0, Regresi, & Pra-Rilis (Branch `v4/fase-5-eval`)
- **Dikerjakan (ringkas):**
  - **Benchmark Live Browser WASM 4-Mode (`eval/eval_phase5_browser.mjs` + `eval/eval_phase5_analyze.py`):** Mengevaluasi commit baseline historis **`eb19337` (`v3_1`)** secara langsung dari Git object store melawan **`v4_enh_only`**, **`v4_stage2_only`**, dan **`v4_full`** di **Chromium `151.0.7922.34` (`ORT-Web 1.20.1`)** pada `119` citra (`heldout_real n=42`, `failures n=8`, `synth_v2_val_clean n=40`, `distractors_v4_val n=29`) dengan Bootstrap `B = 2000`, `seed = 42`.
  - **Analisis Kesalahan & Galeri Terburuk (`eval/reports/phase5_benchmark.json`, `eval/reports/phase5_worst20.html`):** Mengukur matriks konfusi across `52` citra beranotasi kotak (`1.960` GT kotak, sorotan `da/sa`, `ka/na`, `tha/ca`, `ha/ra`), taksonomi kegagalan across `50` citra nyata (`620` GT grafem), serta menyusun galeri visual 20 kasus gagal terburuk beserta inspeksi gambarnya (`hr_hw_10`, `diag_07`, `hr_hw_02`, `diag_08`, dll.).
  - **Perluasan Suite E2E & Audit Pra-Rilis (`eval/test_e2e_assertions.mjs` + `eval/eval_phase5_prerelease.mjs`):** Memperluas asersi positif ke seluruh matriks **Desktop (`1440×900`) & Mobile (`390×844`) × Tema Gelap & Terang**, Batch Multi-Halaman, Fallback Stage-1, dan PWA Offline (`aksarascan-v16`, `38` entri CacheStorage). Mengaudit server statis paritas [`vercel.json`](../vercel.json) (`42` berkas = `28.64 MB`, muat pertama `318 ms DCL` / `1186 ms ONNX+Stage2`).
  - **Sinkronisasi Lisensi & Dokumentasi Rilis:** Memutakhirkan atribusi **CC BY 4.0** & status jujur Stage-1 warisan di tab `Tentang` ([`index.html`](../app/web_camscanner/index.html)), [`dataset/LICENSES.md`](../dataset/LICENSES.md), [`docs/AUDIT.md`](AUDIT.md), [`DATASET_CREDITS.md`](../DATASET_CREDITS.md), [`MODEL_CARD.md`](../MODEL_CARD.md), [`README.md`](../README.md), dan menerbitkan [`docs/BENCHMARK_V4.md`](BENCHMARK_V4.md).

- **Hasil terukur (Gerbang 5):**
  - **`heldout_real` (`n=42`, `SHA256SUMS.lock` `43` berkas, `mismatches=0`):** Browser CER **`34.31% → 30.07%`** (`-4.24 pp`), Grapheme F1 **`59.64% → 62.81%`** (`+3.17 pp`), Word Acc **`29.80% → 31.61%`**, p50 **`947.4 ms`**, p95 **`1817.4 ms`**.
  - **Subset Foto Buram Nyata (`n=5`):** CER **`34.00% → 26.00%`** (`-8.00 pp`), GF1 **`70.05% → 76.88%`** (`+6.83 pp`).
  - **`failures` (`n=8`):** CER **`32.78% → 22.75%`** (`-10.03 pp`), GF1 **`84.81% → 88.51%`** (`+3.70 pp`), Word Acc folded **`29.79% → 33.85%`**.
  - **Penjaga Regresi `synth_v2_val_clean` (`n=40`):** CER **`21.42% → 21.06%`**, GF1 **`82.06% → 83.45%`**, Word Acc folded `12.43% → 12.22%` (regresi minor `-0.21 pp`, plus regresi lokal di `setilanaji +0.41 pp`, `wall_lines_shadow +5.0 pp`, dan `sharp_clean +5.0 pp` dilaporkan jujur).
  - **Distraktor Non-Aksara (`n=29`):** FP/gambar turun **`7.69 → 2.76` (`-64.1%`)**.
  - **Audit Kebocoran & Lisensi:** `check_leakage.py` across `21.457` rekaman **`PASS_ZERO_LEAKAGE`**; `0` font proprieter; `0` AGPL di seluruh jalur/artefak baru v4 (dengan pencatatan jujur bahwa bobot Stage-1 terpasang `aksara_yolo.onnx` masih berstatus transisi warisan sampai `AksaraYOLOX-P2` Apache-2.0 dilatih di GPU).
  - **Suite Uji Terakhir Pasca-Edit (`node eval/test_e2e_assertions.mjs`, `test_fase3_enhancer.mjs`, `test_fase4_ui.mjs`, `eval_phase5_prerelease.mjs`):** Semuanya **PASS** (`0` console error, `0` permintaan jaringan eksternal, Service Worker `aksarascan-v16`).

- **Keputusan & Berikutnya (Gerbang 5 — Rilis v4.0 + Tambahan Perbaikan Sesi Penutup):**
  - **Perbaikan Masalah Tersisa Fase 5 (Goresan `1 px` Kertas Bergaris, Padat Sandhangan, Label Latin Chart, & Kesiapan Installer `AksaraYOLOX-P2`):**
    1. *Goresan `1 px` & Garis Buku Biru:* Membatasi `chromaBoost` (`image_enhancer.js`) agar hanya menarik tinta gelap (`luma < localBgL * 0.72`), menaikkan batas `suppressLongStraightLines` (`72 → 115`) dengan `suppressedMask` + `reinforceUltrathinInk()`, memindah pembersihan tepi `clearBorderNoise` setelah `stripHorizontalLinesBinary` (`app.js`), serta membangun `scanMask` terlindung garis horizontal sebelum `clearBorderNoise`. Hasil: **`hr_hw_10` pulih `100.00% → 10.00%` CER (`GF1 0% → 90%`)** dan uji sintetis `1 px` anti-aliased di kertas bergaris biru mendeteksi `12` glif + memadamkan `5.994 px` garis.
    2. *Label Latin / Tabel di Atas Aksara (`stage2_verifier.js`):* Memperluas `pruneLatinContextAndTwinLabels` dengan penjaga pita `rejectedBoxes`, pasangan kembar mikro tumpang-tindih, serta deteksi baris label mikro terlepas di atas huruf dasar (`gap >= 0.17..0.25 * baseH`, berdasar statistik `synth_v4` `n=3.413` di mana 98% sandhangan asli memiliki `gap/baseH <= 0.158`). Hasil: **`failures_n8` turun `32.78% (v3.1) → 2.01%` CER (`diag_01 = 0%`, `diag_05 = 0%`, `diag_07 = 11.11%`, `diag_08 = 5.00%`)**, sementara `distractors_v4_val_n29` turun ke **`1.52 FP/img` (`-80.2%` vs v3.1)**.
    3. *Sandhangan Padat & Kesiapan `AksaraYOLOX-P2`:* Memasang cadangan grid padat `ts = 360 px` di `app.js`, menambahkan `--grad-accum` di [`training/train_yolox.py`](../training/train_yolox.py) untuk GPU laptop VRAM 4 GB (`GTX 1650`), membuat skrip validasi & instalasi atomik [`tools/install_yolox_release.py`](../tools/install_yolox_release.py), serta menaikkan Service Worker [`sw.js`](../app/web_camscanner/sw.js) ke **`aksarascan-v17`**.
    4. *Audit GPU Lokal vs Cloud:* Terverifikasi `NVIDIA GeForce GTX 1650` (`4096 MiB`), namun drive `C:` hanya menyisakan `7.95 GB` (sedangkan wheel PyTorch CUDA membutuhkan `~2.53 GB` unduhan + `~5.5 GB` terpasang), sehingga pelatihan penuh GPU `AksaraYOLOX-P2` direkomendasikan lewat [`training/AksaraScan_v4_GPU_MultiPlatform.ipynb`](../training/AksaraScan_v4_GPU_MultiPlatform.ipynb) (Kaggle T4 x2 / Lightning AI) atau setelah mengosongkan `>= 15 GB` di `C:`.
    5. *Hasil Benchmark Penutup Post-Fix:* `heldout_real_n42` **`34.31% → 28.68%`** CER (`-5.63 pp`, GF1 `63.52%`), `failures_n8` **`32.78% → 2.01%`** CER (`GF1 98.72%`), `synth_v2_val_clean` **`21.42% → 20.67%`** CER (`WAcc 12.67%`), `neg29` **`1.52 FP/img`** (`42` berkas = `28.66 MB`, `1.046 ms` muat penuh, `aksarascan-v17`). Menunggu persetujuan akhir Rayhan untuk merge ke `main`, push GitHub/Vercel, dan pilihan pelatihan GPU `AksaraYOLOX-P2`.


