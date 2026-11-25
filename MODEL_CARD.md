# Model Card — AksaraScan v3.0

## Ringkasan Model & Pipeline
**AksaraScan** adalah aplikasi pemindai dan penerjemah Aksara Jawa **100% di peramban** (*client-side WASM/ONNX*) berbasis deteksi objek atomik **55 kelas** dan jalur penalaran ortografi-bahasa hibrid:

- **Jalur Utama (Track A)**:
  1. **Detektor**: YOLOv8n ONNX (`aksara_yolo.onnx`, `12.11 MB` FP32 `v3.1.0-blur-hw`, `92.18% mAP50`, `91.72% Precision`, `85.74% Recall` pada 796 citra validasi campuran blur optik, tulisan tangan, 11 font OFL/Apache-2.0, dan `Jawa.zip`; arsitektur YOLOX Apache-2.0 + `P2` stride-4 juga tersedia di `training/yolox_model.py`), dipadukan dengan normalisasi bayangan & penajaman adaptif (*Unsharp Mask*) serta *scale-matched tiling* di `app/web_camscanner/app.js`.
  2. **Rekonstruktor Spasial & Validator Ortografi**: Pengelompokan baris, pemasangan `pasangan` (termasuk `p_ha`/`p_sa`/`p_pa` kanan), pemasangan `sandhangan`, dan validator kanonik blok Unicode `U+A980..U+A9DF` (`app/web_camscanner/orthography.js`).
  3. **Dekoder Beam Konservatif & Mesin Bahasa Jawa**: Pencarian perbaikan terarah berdasar kamus kurasi + `AKSARA_LM_DATA` (`10.589` unigram, `4.500` entri kamus, `560` konteks char-trigram dari `866.150` token korpus) serta aturan afiks produktif Jawa (`app/web_camscanner/javanese_lm_engine.js`).
- **Jalur Penantang (Track B)**: Pengenal baris *segmentation-free* **CRNN + BiLSTM 2-lapis + CTC Loss** (`training/train_crnn_track_b.py` -> `aksara_crnn_ctc.onnx`, `9.86 MB`) dilatih pada data baris `setilanaji/aksara-jawa-ocr` + `synth_v2`.

---

## Ruang Kelas (55 Kelas Atomik)
- **20 Nglegena**: `ha, na, ca, ra, ka, da, ta, sa, wa, la, pa, dha, ja, ya, nya, ma, ga, ba, tha, nga`
- **20 Pasangan**: `p_ha .. p_nga`
- **12 Sandhangan & Pangkon**: `wulu, suku, pepet, taling, tarung, layar, cecak, wignyan, pangkon, cakra, keret, pengkal`
- **3 Tanda Baca Pada**: `pada_adeg, pada_lingsa, pada_lungsi`
- **Catatan Keterbatasan Kelas**: Aksara *Murda*, *Swara* mandiri, *Rekan* (selain dukungan validator Unicode), serta *Angka Jawa* belum menjadi kelas deteksi atomik tersendiri pada rilis 55 kelas ini.

---

## Sumber Data & Kepatuhan Lisensi
- **Dataset Terkonversi & Terverifikasi**:
  - `hanacaraka` (`CC0-1.0`, `1.453` kotak tervalidasi)
  - Mendeley `Jawa.zip` (`10.931` kotak Nglegena + `1.469` silabus bersandhangan nyata `jawa_syllables_boxed` dengan kotak atomik `wulu`/`pepet`/`taling`/`tarung` terverifikasi; label `Vokal E` dan `Vokal È` telah diperbaiki setelah inspeksi visual)
  - `setilanaji/aksara-jawa-ocr` (`Apache-2.0`, `2.059` train + `150` val baris Unicode setelah deduplikasi dHash; `67` baris nyata berpenyelarasan guru tepat dipakai untuk *fine-tuning* Track A)
  - Korpus Bahasa Jawa: `IndoNLP/nusax`, `IndoNLP/nusa-writes`, `nsulistiyawan/sastra-jawa` (`MIT`), `afrizalha/Gatra-1-Javanese`, `DGurgurov/javanese_conceptnet` (`MIT`), `thesimonharms/public-javanese-dataset` (`CC-BY-SA-3.0`/`MIT`), Wikipedia Jawa (`jvwiki-latest-pages-articles.xml.bz2`).
- **Generator Sintetis v2 & v3 (`tools/synth_v2/generate_v2.py` & `generate_v3_blur_hw.py`)**: HANYA menggunakan font berlisensi **SIL OFL 1.1 / Apache-2.0** (`NotoSansJavanese` Regular & Bold, `TuladhaJejegOT-Regular.{ttf,otf}`, `Sehulbari_Arif_Budiarto`, `Bharatayuddha`, `Chandrasa`, `Righma_Ciddhi`, `TUNJUNG_BIRU`, `Tunjung_putih`, `Warastra`, `YBG_Kramawirya`), **bebas dari font proprieter Windows (`javatext.ttf`)** maupun font non-komersial (`NC-ND`).
- **Pelatihan Bebas AGPL-3.0**: Skrip pelatihan baru (`training/yolox_model.py`, `training/train_yolox.py`, `training/train_crnn_track_b.py`) menggunakan PyTorch/YOLOX (**Apache-2.0 / MIT**) — bebas dari `ultralytics` (`AGPL-3.0`).

---

## Privasi, Keamanan, & Batasan Pengukuran
- **Privasi 100% Lokal**: Selama inferensi gambar, **0 permintaan jaringan eksternal** dikirim keluar dari peramban. Terjemahan daring opsional berada dalam kondisi **MATI secara baku** (*opt-in explicit*).
- **Pengukuran Jujur (`eval/reports/v3.0.md`)**:
  - Evaluasi lengkap mencakup *Golden Parity Set* (`eval/parity_20.jsonl`, `n=20`) dan *Diagnostic Failure Set* (`eval/failures/manifest.jsonl`, `n=8`).
  - `eval/failures/` adalah set diagnostik kecil — klaim generalisasi dunia nyata penuh menunggu kumpulan foto multi-penulis (`eval/real_val`/`eval/real_test` terpisah lewat `eval/split_real.py`) dan pelatihan GPU penuh sesuai `training/COLAB_GPU_RUNBOOK.md`.
- **Mode Kegagalan Diketahui**:
  1. Tulisan tangan berpena tipis dengan bayangan gradien gelap atau teks Latin tepat di atas huruf (`diag_04..diag_08`) memerlukan pelatihan ulang bobot detektor di GPU (`imgsz 960/1280` dengan *P2* head) karena bobot detektor v2.5 saat ini belum dilatih pada `synth_v2` dan `hard_mined` penuh.
  2. Tanda diakritik kecil `< 16 px` (`cecak`, `wulu`, `pepet`) pada gambar resolusi sangat rendah dapat tertukar atau terlewat oleh detektor; Dekoder Beam Konservatif memulihkan kasus jelas seperti `wêda -> wédang` tanpa menyentuh kata kamus yang sudah sah.
