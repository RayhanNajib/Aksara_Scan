# AksaraScan — Master Engineering, Dataset, & Anti-Slop UI/UX Brief (Panduan Prompt Kompleks Jangka Panjang untuk Claude)

> **Tujuan Dokumen Ini:** Memberikan konteks historis yang jujur, inventaris progres yang sudah dicapai, akar masalah teknis (*root-cause bugs*), peta perluasan dataset & model pencocokan berbasis referensi, rencana fitur tambahan, serta spesifikasi arsitektur **UI/UX Anti-Slop** (berbasis **ScanRead.ai** + skill desain Hermes **`impeccable`**, **`claude-design`**, **`popular-web-designs`**, **`design-md`**, **`humanizer`**, dan **`frontend-design`**) untuk menghasilkan prompt rekayasa jangka panjang yang presisi, modular, dan siap eksekusi.

---

## 1. Ringkasan Identitas & Batasan Arsitektur Proyek

- **Nama Proyek:** AksaraScan (`RayhanNajib/Aksara_Scan`)
- **Kategori:** Pemindai, Transliterator, dan Penerjemah Aksara Jawa **100% Client-Side di Peramban** (*Zero External Network Inference*, PWA offline-ready).
- **Tim:** Kelompok 4 — Muhammad Rayhan Najib (`240533600012`) & M. Ikhsan Ridho A. (`240533609185`), Dosen Pengampu: Mahfud Jiono, S.Pd., M.Eng., Universitas Negeri Malang.
- **Deployment Target:** Vercel (`vercel.json` menyajikan `app/web_camscanner`), tanpa backend inference berbayar.
- **Kepatuhan Lisensi Ketat:** Kode aplikasi **MIT**; data & font latih wajib bebas font proprieter (`javatext.ttf` dilarang) serta jalur latih jangka panjang memisahkan ketergantungan **AGPL-3.0** (`ultralytics` digantikan **YOLOX** Apache-2.0 / **RT-DETRv2** / **CRNN-CTC**).

---

## 2. Kronologi Progres & Revisi yang Sudah Diselesaikan

| Fase / Versi | Commit | Capaian Teknis Nyata di Repositori |
|---|---|---|
| **v1.0 MVP** | `b97a02a` | Web client-only klasik: Grayscale → Otsu → Proyeksi Horizontal/Vertikal → Crop `96×96` → Linear SVM (`model.json`, 20 kelas *Nglegena*) menggunakan dataset `hanacaraka` CC0 (`1.579` citra, split 70/15/15). Akurasi uji bersih 68,4%, terdegradasi berat (53,2%) pada foto nyata/miring. |
| **v2.0 Dual-Engine** | `50e33f3` | Menambahkan detektor **YOLOv8-Nano ONNX** 52 kelas (*Nglegena + Pasangan + Sandhangan*) via `onnxruntime-web` WASM + rekonstruktor spasial `orthography.js` (memasang *pasangan* subskrip/kanan & *sandhangan*). |
| **v2.5 Multi-Font** | `4221452` | Ekspansi ke **55 kelas** (+3 *Pada*: `pada_adeg`, `pada_lingsa`, `pada_lungsi`), *Line-Adaptive Tiling*, dukungan terjemahan dasar, dan konvergensi sintetis awal (`mAP50 98,81%` hanya di domain bersih, tetapi rapuh di foto buram/kertas nyata). |
| **v3.0 Penalaran & Data Terbuka** | `66d070a` | Audit teknis & lisensi (`docs/AUDIT.md`); integrasi `Jawa.zip` (CC BY 4.0), `setilanaji/aksara-jawa-ocr` (Apache-2.0), dan korpus teks (`IndoNLP/nusax`, `nusa-writes`, `nsulistiyawan/sastra-jawa`, `afrizalha/Gatra-1-Javanese`, `jvwiki`); validator blok Unicode `U+A980..U+A9DF` (`orthography.js`); Dekoder Beam konservatif & mesin LM Jawa (`javanese_lm_engine.js`, 10.589 unigram + 4.500 entri kamus); modul arsitektur bebas AGPL (`training/yolox_model.py`, `training/train_yolox.py`, `training/train_crnn_track_b.py`); Google Colab GPU Runbook (`training/AksaraScan_v3_Colab_GPU.ipynb`). |
| **UI Overhaul Awal & Vercel** | `31b6cfd`–`a202f0d` | Penghapusan lencana debug mentah (*Ortografi Valid / Mentah detektor*), pemindahan kredit tim ke tab `Tentang`, pemendekan footer, penyelarasan token tema Gelap/Terang, dan uji E2E Chromium (`eval/test_e2e_assertions.mjs`). |
| **v3.1 Blur & Ekspansi Font** | `e87c088` | Ekspansi **11 font SIL OFL/Apache-2.0**; ekstraksi silabus bersandhangan `Jawa.zip` & baris buku `setilanaji`; penambahan normalisasi pencahayaan + *3×3 Unsharp Mask* dasar (`prepareYoloInputCanvas` di `app.js`); *fine-tuning* 3 epoch CPU/lokal → `aksara_yolo.onnx` v3.1.0 (`12.11 MB`, val `92,18% mAP50`, Precision `91,72%`, Recall `85,74%` pada 796 citra campuran). |
| **Revisi Favicon Asli** | `eb19337` | Penggantian ikon generatif/braket ramai dengan **1 glif vektor murni aksara Jawa `ꦲ` (*ha*, `U+A9B2`)** hasil ekstraksi langsung dari `NotoSansJavanese-Bold.ttf` (`favicon.svg/png/ico`, `icon-192/512.png`, `apple-touch-icon.png`) + bump Service Worker cache ke `aksarascan-v12`. |

---

## 3. Daftar Bug & Akar Masalah Saat Ini (Fokus Perbaikan Jangka Panjang)

Meskipun metrik validasi v3.1 mencapai `92,18% mAP50` pada split sintetis+subset lokal, uji dunia nyata dan keluhan pengguna menunjukkan **5 akar masalah fundamental** yang belum terpecahkan:

### Bug 1 — Sensitivitas Tinggi terhadap Variasi Lengkungan Goresan Tulisan Tangan
- **Gejala:** Ketika lengkungan aksara sedikit berbeda dari font baku atau ditulis dengan gaya tangan acak (*cursive slant*, bukaan *taling/wulu* berbeda, kaki *na/ka/da/sa/pa/ya/ba/nya/ga/ma* agak membulat atau meruncing), detektor langsung salah menebak kelas atau memecah 1 aksara lebar menjadi 2 aksara sempit (`ha` → `ra + ...`).
- **Akar Masalah Teknis:**
  1. Model saat ini hanya `YOLOv8n` beresolusi masukan **`416×416`** (`yolo_meta.json`) dengan *smallest stride* **P3 (`stride 8`, fitur 52×52)**. Ciri pembeda lengkungan kecil (ekor `da` vs `sa`, lekuk dalam `ka` vs `na`, lubang `tha` vs `ca`) terkompresi hilang.
  2. Generator sintetis v3 (`generate_v3_blur_hw.py`) hanya mengaplikasikan rotasi `±4°` dan shear affine kaku `2°`, **belum menerapkan deformasi elastis non-rigid (Thin-Plate Spline / TPS), variasi ketebalan kuas (*morphological stroke erosion/dilation*), ataupun variasi gaya penulis acak per karakter**.
  3. Tidak ada **normalisasi struktur/skeleton** maupun cabang penalaran invariansi gaya.

### Bug 2 — Penanganan Citra Blur Sangat Terbatas (Belum Ada Pipeline *Deblur & Adaptive Sharpening* Sejati)
- **Gejala:** Foto yang sedikit buram gerak (*motion blur*), kurang fokus kamera HP (*defocus blur*), atau tangkapan layar terkompresi JPEG sering gagal dideteksi atau diacuhkan oleh praproses.
- **Akar Masalah Teknis (Audit `app/web_camscanner/app.js:580-719`):**
  1. Fungsi `prepareYoloInputCanvas` melakukan *early bailout* (`if (!needIllumNorm && avgLap >= 21 && minInk <= 65) return srcEl;`), sehingga gambar dengan blur ringan-sedang tetap dibiarkan buram jika tinta utamanya cukup gelap.
  2. Penajaman hanya memakai **1 pass *Unsharp Mask* kernel 3×3 radius-1px** lalu diikuti kurva *power gamma* (`Math.pow(..., 1.32)`) yang malah menipiskan atau memutus goresan lengkung tipis saat blur.
  3. Tidak ada **estimasi tingkat blur (*Laplacian variance / edge spread*) yang mengaktifkan *Multi-Scale Unsharp Mask* + *CLAHE (Contrast Limited Adaptive Histogram Equalization)* / *Wiener-Richardson-Lucy lightweight deconvolution***, dan **belum ada kontrol UI interaktif** (slider/toggle ketajaman + pratinjau langsung *before/after*) di kanvas pemindai.

### Bug 3 — Pencocokan Sekali Tembak ("Asal Mirip"), Belum Berbasis Bank Referensi & Verifikasi Metrik
- **Gejala:** Keputusan label sepenuhnya bergantung pada *argmax softmax* mentah kotak YOLO (satu jaringan deteksi 12 MB yang dilatih singkat). Jika skor *confidence* rendah atau dua kelas mirip bersaing (`0.38` vs `0.35`), sistem tidak melakukan verifikasi silang ke bank referensi visual.
- **Akar Masalah Teknis:**
  1. Pada `app.js:900-945`, meskipun `topK` sudah diekstrak hingga 5 kandidat, `orthography.js` hanya memanfaatkannya secara pasif di level string kata, bukan memverifikasi ulang *patch citra* kotak yang meragukan.
  2. Tidak ada **Stage-2 Reference Prototype / Metric Verifier** (misalnya *lightweight glyph embedding ONNX* 1–2 MB berbasis MobileNetV3/ConvNeXt-Atto + *Cosine/ArcFace Prototype Matching* atau deskriptor bentuk *HOG + Zoning + Chamfer/Skeleton Similarity*) yang membandingkan potongan ROI kandidat terhadap **bank prototipe multi-penulis & multi-font terkurasi** sebelum memutuskan label.
  3. Ketidakhadiran *Prototype Similarity Visualizer* membuat pengguna maupun pengembang sulit melihat "deteksi ini mirip dengan referensi dataset yang mana".

### Bug 4 — Cakupan Dataset & Ragam Sumber Masih Kurang Dalam untuk Fine-Tuning Akurasi Tinggi
- **Kondisi Saat Ini:** Pelatihan v3.1 (`training/finetune_v3_blur.py`) masih didominasi data sintetis 11 font + `hanacaraka` (1.453 kotak) + `Jawa.zip` (sebagian besar huruf tunggal terisolasi) + `setilanaji` (hanya 67 baris yang terpakai di fine-tuning Track A) dengan hanya **3 epoch** di CPU (`yolo_meta.json`).
- **Keterbatasan Kelas:** Belum mencakup aksara yang kerap muncul pada papan nama/naskah/tulisan acak seperti **Pa Cerek (`ꦉ`), Nga Lelet (`ꦊ`), Aksara Murda (8 huruf + pasangan), Aksara Swara mandiri (`ꦄ ꦆ ꦈ ꦌ ꦎ`), Rekan (`꦳`), & Angka Jawa (`꧐–꧙`)**.

### Bug 5 — Kelemahan Desain UI/UX & Konsistensi Interaksi Saat Ini
- Layout panel kiri saat ini (`app/web_camscanner/index.html:96-220`) masih menumpuk dropzone, 4 kartu sampel statis, bar pengalih mesin, accordion opsi, dan tombol aksi dalam **satu kolom kiri yang padat** sebelum pengguna selesai mengunggah gambar (*cognitive overload*).
- Elemen `wayang-corner` di kanvas (`index.html:130-133`) terkesan dekoratif/artifisial (*border/corner ornamentation*).
- Diperlukan arsitektur UI modern yang bersih seperti **ScanRead.ai** (landing dropzone langsung fokus → transisi mulus ke *Split Workbench* dokumen + teks hasil + toolbar aksi cepat) yang dipadukan ketat dengan aturan **Anti-Slop UI** dari skill Hermes.

---

## 4. Strategi Perluasan Referensi & Dataset Multi-Sumber (Untuk Fine-Tuning v4.0)

Untuk menjawab *"pencocokannya harus banyak referensi gk asal mirip aja dan perbanyak datasetnya"*, arsitektur prompt untuk Claude harus merancang **pipeline kurasi, normalisasi skema, deduplikasi pHash/dHash, dan pemisahan *writer-disjoint*** dari seluruh sumber berikut:

### A. Sumber Terbuka yang Sudah Diverifikasi di Repo (Maksimalkan Pemanfaatan Penuh)
1. **`dataset/hanacaraka`** (CC0-1.0, `1.579` citra tulisan tangan 20 *Nglegena*).
2. **`Jawa.zip` Mendeley** (`doi:10.17632/vfj32bpjsf.1`, CC BY 4.0, `13.256` citra — manfaatkan **seluruh** `variations` dan `2.196` silabus `E/I/O/U/È`, bukan hanya subset `Vokal A`).
3. **`setilanaji/aksara-jawa-ocr`** (Hugging Face, Apache-2.0 — perluas penggunaan **seluruh ~2.205 baris berpoligon/beranotasi** baik lewat *forced-alignment pseudo-labeling* untuk Track A maupun pelatihan penuh Track B CRNN-CTC/PARSeq).
4. **11 Font SIL OFL / Apache-2.0** di `dataset/external/fonts/` (`NotoSansJavanese`, `TuladhaJejegOT`, `Sehulbari`, `Bharatayuddha`, `Chandrasa`, `Righma_Ciddhi`, `TUNJUNG_BIRU`, `Tunjung_putih`, `Warastra`, `YBG_Kramawirya`).

### B. Sumber Dataset & Referensi Baru yang Harus Diintegrasikan
1. **Roboflow Universe (Deteksi Objek Kotak Langsung, Cek Lisensi CC BY 4.0 per Proyek):**
   - `pa-relate/aksara-jawa` (object detection bounding box Aksara Jawa).
   - `febgabut/aksara-jawa-classification` (variasi *sandhangan* bertingkat).
   - `Fawwaz Zaini Ahmad — Javanese Handwriting Object Detection` (`1.014 train / 284 val / 166 test` kotak tulisan tangan nyata).
   - `fasel/aksarajawasandangan-2` (`4.137` citra anotasi *pasangan* dan *sandhangan*).
2. **Kaggle & GitHub Multi-Writer:**
   - `hermansugiharto/aksara-jawa-yolo-v5-dataset` + repo MIT [`hermansh-id/Aksara-Jawa-OCR`](https://github.com/hermansh-id/Aksara-Jawa-OCR).
   - Kaggle *Handwriting Aksara Jawa on Paper* & *Javanese Script Augmented* (untuk ekstraksi tekstur tinta pulpen/spidol di kertas bertekstur nyata).
   - GitHub [`IqbalLx/Hanacaraka-AI`](https://github.com/IqbalLx/Hanacaraka-AI) (referensi augmentasi tulisan tangan).
3. **Naskah Cetak, Manuskrip Historis, & Aksara Murda:**
   - Hugging Face **`NusaAksara/NusaAksara`** (ACL 2025 benchmark — ~100 halaman cetak Jawa + 6.265 baris OCR + transliterasi; konfirmasi izin penggunaan untuk evaluasi *held-out real print*).
   - Hugging Face **`RayFox52/Aksara-jawa-carakan-murda`** (referensi 27 kelas termasuk 7 *Aksara Murda* bila ruang kelas diperluas dari 55 → 75+ kelas).
   - Mendeley **`doi:10.17632/ms6sjjpzwr.1`** (Suherli et al., BINUS, YOLOv8 + TrOCR arsip historis & British Library EAP — cek lisensi EAP).
4. **Generator Sintetis v4 (*Elastic Handwritten Stroke Synthesis*):**
   - Tambahkan **Thin-Plate Spline (TPS) Elastic Warp** + **Stroke Width Modulation** (erosi/dilasi acak radius 1–2 px) + **Motion/Defocus PSF Blur** + **JPEG compression artifacts** + **Paper fold/shadow gradients** agar model kebal terhadap variasi lengkungan kecil dan gambar buram.
   - Generate **Negative Samples** terfokus: teks Latin berdekatan, garis buku catatan, watermark, dan lipatan kertas.

### C. Arsitektur Deteksi + Pencocokan Berbasis Referensi (2-Stage Hybrid)
Minta Claude merancang arsitektur jangka panjang yang siap di-training di GPU (Colab / Kaggle / Lightning.ai, ekspor ONNX FP16 `< 20 MB`):
1. **Track A1 (Detektor Resolusi Tinggi Bebas-AGPL):** **YOLOX-Tiny / YOLOX-S atau RT-DETRv2 dengan *P2 High-Res Neck* (`stride 4`, input `640`–`960 px`)** agar *sandhangan* kecil (`< 14 px`) dan lekuk tipis tidak hilang.
2. **Track A2 (Stage-2 Reference Prototype Matcher / Metric Verifier):**
   - Backbone ringan (`MobileNetV3-Small` / `EfficientNet-B0` *ArcFace* atau *Cosine Metric Head*, ekspor `aksara_embed.onnx` `<= 3 MB`).
   - Dijalankan **hanya pada ROI kotak ambigu** (*confidence* `0.18..0.62` atau selisih top-1 vs top-2 `< 0.15`), mencocokkan vektor embedding citra potongan terhadap **bank centroid multi-referensi** (*tulisan tangan real + 11 font + variasi lengkungan*) (`reference_prototypes.json`), sekaligus menampilkan *"Cocok dengan Referensi X (skor %)"* di inspektor UI!
3. **Track B (Line-Level CRNN-BiLSTM / SVTR-CTC Verifier):** Pengenal baris *segmentation-free* (`aksara_crnn_ctc.onnx`) untuk memvalidasi konteks kalimat saat karakter tersambung/lengkung meragukan.
4. **Pra-pemroses Gambar Di-Browser (`image_enhancer.js` + opsional WebGL/WASM):**
   - Detektor skor blur otomatis (Laplacian variance).
   - **Adaptive Deblur & Sharpen Engine:** Sauvola local illumination flattening → CLAHE → **Multi-Scale Unsharp Mask / Laplacian-Weighted Sharpening** (atau opsional *Richardson-Lucy 3-step deblur* ringan untuk patch terpilih) **tanpa merusak lengkungan stroke tipis** + kontrol slider real-time *Ketajaman & Kontras* dengan mode *Split Compare*.

---

## 5. Rencana Penambahan Fitur Terarah (Scope Terkendali, Bernilai Tinggi)

1. **Toolbar Peningkat Citra (*Pre-Scan Image Enhancer*):** Toggle/slider *Auto-Sharpen Blur*, *Hapus Bayangan Kertas*, *Rotasi/Deskew 90°*, dan *Before/After Split Slider* langsung di atas kanvas dokumen sebelum/sesudah pindai.
2. **Inspektor Referensi & Disambiguasi Glif (*Stroke Reference Popover*):** Saat pengguna mengklik kotak deteksi atau kata bergaris putus, tampilkan **top-3 kandidat kelas**, **gambar referensi aksara kanonik + variasi tulisan tangan** ("Mengapa terdeteksi *ha* bukan *la*"), dan tombol koreksi 1-klik yang tersimpan di riwayat koreksi lokal (`localStorage`).
3. **Mode Antrean / Batch Multi-Halaman (Terinspirasi ScanRead.ai `/batch`):** Mengunggah atau menempel beberapa gambar/foto naskah sekaligus lalu mengekspor gabungan hasil ke **`.txt`, `.md`, atau `.json`**, serta tombol **"Salin Semua"**.
4. **Riwayat Pindai Lokal (*Recent Scans*):** Penyimpanan riwayat pindai terbaru di `IndexedDB` browser (100% lokal, bisa dibersihkan 1 klik) agar pengguna tidak kehilangan hasil saat *refresh*.

---

## 6. Referensi UI/UX: ScanRead.ai + Anti-Slop Design System & Teori Skill Hermes

Seluruh perombakan antarmuka wajib mematuhi **2 pilar referensi** ini agar hasilnya profesional, editorial, cepat dipakai, dan bebas dari "AI-generated UI slop":

### Pilar A — Anatomi Layout & Navbar dari **ScanRead.ai** (`https://scanread.ai/`)
Hasil inspeksi langsung struktur DOM & CSS token ScanRead.ai memperlihatkan pola produk OCR modern yang jernih:
1. **Top Navbar Minimalis & Lengket (`h-16`, `max-width: 1280px`):**
   - Glassmorphic halus (`backdrop-blur-md`, border bawah tipis `1px` hampir transparan).
   - **Kiri:** Logo glif vektor `ꦲ` + wordmark serif italic (`Aksara` + aksen `Scan`).
   - **Tengah-Kanan (Hanya 3–4 menu fungsional):** `Pemindai` · `Batch` · `Lab & Referensi` · `Tentang` *(atau taruh Kamus di dalam Lab & Referensi agar nav tetap ramping)*.
   - **Aksi Kanan (Tanpa berisik):** Status ringkas engine (pill kecil tanpa animasi berlebih) + tombol pill primer (`Pindai Baru`) + pengalih tema Gelap/Terang yang ringkas.
2. **Zero-Friction Upload Stage → Split Workbench (Progressive Disclosure):**
   - **Saat Belum Ada Gambar:** Halaman fokus penuh pada **Hero Ringkas + Dropzone Terpusat yang Lapang** (mendukung *Drag & Drop*, *Pilih Berkas*, *Kamera*, *Paste `Ctrl+V`*, dan chip format `PNG · JPG · WebP · Foto Buram` + 4 contoh uji kecil di bawah dropzone — **jangan tampilkan panel pengaturan rumit atau kartu kosong di kanan saat belum ada gambar!**).
   - **Setelah Gambar Masuk:** Transisi mulus ke **Split Workbench Asimetris 2-Kolom** (`52% / 48%` di desktop):
     - **Kiri (Kanvas Dokumen + Micro-Toolbar):** Gambar asli dengan overlay bounding box bersih, slider *Zoom/Sharpen/Biner* sejajar tipis di atas/bawah kanvas, dan tombol pindai ulang area (*ROI*).
     - **Kanan (Dokumen Hasil Terekstrak):** Output **Aksara Jawa Unicode** berukuran baca nyaman, diikuti tab/blok **Bacaan Latin** + **Arti Bahasa Indonesia** dengan tombol aksi cepat (`Salin Unicode`, `Salin Latin`, `Unduh .txt/.md/.json`).
   - **Bagian Bawah (Ringkas & Jujur):** 3 langkah cara kerja (`Unggah → Tajamkan & Deteksi → Baca & Koreksi`) + **Footer super ringkas 1 baris** (identitas & tautan tab `Tentang`).

### Pilar B — Prinsip & Teori **Anti-Slop UI** dari Skill Hermes yang Sudah Terpasang
Skill berikut telah dipasang di `C:\Users\WA\.agents\skills\` (aktif lintas sesi Antigravity) untuk dipelajari dan dijadikan acuan saat membuat prompt maupun mengimplementasikan UI:

1. **`impeccable` (`craft-floor.md`, `critique.md`, `layout.md`, `typeset.md`, `colorize.md`, `distill.md`, `polish.md`, `operate.md`):**
   - **Visitor Mode:** Aplikasi utama harus memakai mode **`Operate`** (keterbacaan, kecepatan tugas, stabilitas kepadatan, tanpa dekorasi mengganggu); halaman dokumentasi/tentang memakai **`Read`**.
   - **Larangan Tegas (*Refuse List* / Anti-Slop):**
     - **DILARANG** *kicker / eyebrow label* serba huruf kapital di atas judul utama.
     - **DILARANG** menumpuk grid kartu berukuran seragam (`card` di dalam `card`, sudut bertumpuk, border kiri berwarna tebal `> 1px`).
     - **DILARANG** teks gradasi pada judul, *glow/halo neon blur* dekoratif, penomoran seksi artifisial (`01 / 02 / 03` jika bukan urutan proses nyata), dan penggunaan emoji/karakter acak sebagai pengganti ikon SVG standar.
     - **DILARANG** ornamen sudut palsu (*fake decorative corner brackets* seperti `.wayang-corner`) di atas kanvas foto dokumen pengguna — biarkan kanvas foto bersih!
   - **Tipografi & Permukaan Browser:**
     - Pertahankan pasangan font editorial berkarakter (**`Newsreader` / `Fraunces`** untuk aksen judul merek, **`Plus Jakarta Sans` / `Manrope`** untuk UI fungsional dengan batas panjang baris `65–75ch`, **`JetBrains Mono`** hanya untuk metrik/skor/data teknis, dan **`Noto Sans Javanese`** eksklusif untuk render aksara).
     - Tema juga wajib mewarnai **permukaan bawaan browser**: `::selection`, custom scrollbar, `:focus-visible` ring, dan `tabular-nums` pada angka skor/latensi.
     - Rasio kontras WCAG minimum **`4.5:1`** untuk teks isi & placeholder pada **kedua tema** (*Gelap* maupun *Terang*); teks sekunder harus ber-tint warna latar/ink, **bukan abu-abu mati**.
2. **`claude-design` & `frontend-design`:**
   - Rancang hierarki visual lewat **kedekatan kelompok (*proximity*) dan ritme jarak** (skala kelipatan 4px, jarak atas heading > jarak bawah), bukan membungkus setiap blok dengan kotak kartu berbayangan.
   - Satu momen gerak terarah (*authored state transition* saat unggah/ekstraksi selesai), bukan animasi *fade-up* berulang di setiap elemen.
3. **`popular-web-designs` & `design-md`:**
   - Dokumentasikan semua token final ke `DESIGN.md` format Google Spec (warna semantik `--bg`, `--surface`, `--ink`, `--accent`, skala tipografi, radius, status kontras) agar konsisten di seluruh tab tanpa warna teks tertutup background.
4. **`humanizer` (Anti-Slop Copy):**
   - Gunakan salinan UI bahasa Indonesia yang lugas, ringkas, menyebut aksi nyata kontrol (*"Tajamkan Foto"*, *"Pindai Area Pilihan"*), dan pesan error menjelaskan masalah + cara pulih tanpa retorika marketing AI yang klise.
5. **`dogfood` & `adversarial-ux-test`:**
   - Wajib verifikasi visual multi-viewport (desktop `1440px` & mobile `390px`), uji tema Terang & Gelap secara menyeluruh, cek *zero console error*, dan uji kasus ekstrem (foto sangat buram, gambar besar, kanvas kosong, kata OOV panjang).

---

## 7. Checklist Struktur Prompt yang Harus Dihasilkan Claude

Saat Anda memberikan dokumen ini ke Claude untuk membuat **prompt kompleks & panjang**, arahkan Claude agar membagi prompt final menjadi **5 Fase Rencana Eksekusi Berurutan** lengkap dengan kriteria kelulusan (*acceptance criteria*):

1. **Fase 1 — Ekspansi Dataset & Augmentasi Lengkungan/Blur (`tools/` & `training/`):**
   - Integrasi multi-sumber dataset (Roboflow + Kaggle + `Jawa.zip` penuh + `setilanaji` penuh + 11 font + perluasan kelas *Pa Cerek / Nga Lelet / Murda / Swara / Angka* opsional).
   - Pipeline augmentasi **TPS Elastic Curve + Morph Stroke Width + Motion/Defocus Blur + Hard Negatives** + *writer-disjoint split*.
2. **Fase 2 — Arsitektur Detektor Resolusi Tinggi + Stage-2 Reference Prototype Matcher (`training/` & ONNX):**
   - Pelatihan GPU **YOLOX-S P2 (`stride 4`, 960px)** atau **RT-DETRv2 Apache-2.0** + pelatihan **Stage-2 Metric Embedding Network** (`aksara_embed.onnx`) + ekspor **`reference_prototypes.json`** + fusi Track B CRNN-CTC.
3. **Fase 3 — Engine Praproses Deblur/Sharpen Adaptif & Rekonstruktor Hibrid (`app/web_camscanner/`):**
   - Implementasi modul `image_enhancer.js` (deteksi blur otomatis, *Sauvola illumination normalization*, *CLAHE*, *Multi-Scale Sharpen / Lightweight Deblur* non-destruktif) + integrasi verifikasi *top-K* terhadap prototipe referensi di browser tanpa panggilan jaringan eksternal.
4. **Fase 4 — Rombak Total UI/UX ScanRead-Inspired + Anti-Slop (`DESIGN.md`, `index.html`, `styles.css`, `app.js`):**
   - Terapkan topbar `h-16` glassmorphic, *Centered Dropzone Hero* → *Split Workbench*, *Pre-Scan Sharpen/Compare Toolbar*, *Reference Prototype Inspection Popover*, *Batch Export*, tema Gelap/Terang konsisten `>= 4.5:1`, nol ornamen dekoratif artifisial, dan nol kartu bertumpuk.
5. **Fase 5 — Audit Otomatis, Benchmark Jujur, & Regresi (`eval/`):**
   - Benchmark sebelum vs sesudah pada set uji *held-out* multi-penulis & foto blur nyata (CER, Word Accuracy, F1 Grapheme, Latency WASM) + `eval/test_e2e_assertions.mjs` pada desktop & mobile di tema Terang & Gelap.
