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

