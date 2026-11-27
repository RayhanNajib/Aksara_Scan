# Inventaris Referensi Riset, Dataset, Leksikon, Font, & Arsitektur — AksaraScan v3.0

Dokumen ini merangkum seluruh sumber dataset citra, korpus teks/kamus Jawa-Indonesia, font berlisensi terbuka, repositori kode komunitas/blog/skripsi, serta pustaka model deteksi & OCR untuk mendukung perluasan Fase 1–4 AksaraScan v3.0. Status lisensi dicatat apa adanya sesuai verifikasi 25 November 2026; entri tanpa lisensi eksplisit ditandai `PERLU_KONFIRMASI` dan tidak dimasukkan ke repo sebelum dikonfirmasi.

---

## 1. Dataset Citra Aksara Jawa (Tulisan Tangan, Objek, & OCR Baris)

| ID | Sumber & URL | Isi & Format | Peran di v3.0 | Lisensi Terverifikasi & Status |
|---|---|---|---|---|
| `hanacaraka_vzr` | GitHub [vzrenggamani/aksarajawa-hanacaraka](https://github.com/vzrenggamani/aksarajawa-hanacaraka) & [Kaggle Mirror](https://www.kaggle.com/datasets/vzrenggamani/hanacaraka) | 1.579 citra tulisan tangan digital 20 nglegena | Sudah terpakai di `dataset/hanacaraka/` (split 70/15/15 seed 42) | **CC0-1.0** · `OK` |
| `mendeley_ind_scripts` | Mendeley Data [DOI 10.17632/vfj32bpjsf.1](https://doi.org/10.17632/vfj32bpjsf.1) (Aditya Firman Ihsan) | Arsip `Jawa.zip` lokal: **13.256 citra** (`Vokal A` 9.994, `E/I/O/U/È` 2.196, `variations` 1.066) | Pra-latih glif + penambangan goresan tulisan tangan vokal/nglegena & split bebas-kebocoran | **CC BY 4.0** · `OK` (wajib atribusi) |
| `hf_setilanaji_ocr` | Hugging Face [setilanaji/aksara-jawa-ocr](https://huggingface.co/datasets/setilanaji/aksara-jawa-ocr) | Format PPOCRLabel (`Label.txt` berisi poligon titik + transkripsi Unicode kalimat Jawa lengkap berpasangan & bersandhangan, ~1K–10K citra) | Pelatihan & validasi pengenal baris Track B (CRNN-CTC / TrOCR) & *scriptio continua* | **Apache-2.0** · `OK` |
| `rf_fawwaz_det` | Roboflow Universe *Javanese Handwriting Object Detection-Aksara Jawa* (Fawwaz Zaini Ahmad) | 20 kelas nglegena tulisan tangan dengan anotasi kotak (1.014 latih, 284 val, 166 uji) | Tambahan tulisan tangan berkotak nglegena | **CC BY 4.0** (tertera di sumber; butuh API key) · `PERLU_LOGIN` |
| `rf_fasel_sandhangan` | Roboflow Universe *aksarajawasandangan-2* (fasel) | Deteksi pasangan dan sandhangan (4.137 citra beranotasi) | Menguatkan kelas sandhangan & pasangan lemah (butuh cek pemetaan 55 kelas) | **CC BY 4.0** (butuh API key) · `PERLU_LOGIN` |
| `rf_danukingg` / `rf_baksara` / `rf_parelate` / `rf_febgabut` | Roboflow Universe (`danukingg/aksara-jawa-iglbh-nzoqx`, *baksara* ITS, *PA RELATE*, *FebGabut*) | Variasi dataset deteksi objek & klasifikasi sandhangan Aksara Jawa komunitas | Eksplorasi kotak tambahan bila lisensi CC BY 4.0/MIT terverifikasi di workspace masing-masing | Cek kartu proyek · `PERLU_KONFIRMASI` / `PERLU_LOGIN` |
| `kg_hermansugiharto` | [Kaggle Aksara Jawa YOLO v5](https://www.kaggle.com/datasets/hermansugiharto/aksara-jawa-yolo-v5-dataset) (juga [hermansh-id/Aksara-Jawa-OCR](https://github.com/hermansh-id/Aksara-Jawa-OCR)) | Dataset berformat YOLO untuk 20 aksara dasar (dirujuk riset YOLOv5/v9) | Tambahan kotak deteksi nglegena (kode repo pendukung berlisensi MIT; dataset Kaggle perlu login & cek lisensi) | Kode **MIT**, Dataset Kaggle · `PERLU_LOGIN` |
| `kg_phiard_paper` | Kaggle (*Handwriting Aksara Jawa on Paper* / *Javanese Script Augmented*) | Potongan karakter tulisan tangan di kertas fisik & varian augmentasi | Ekstraksi tekstur goresan pena kertas nyata (tidak cocok langsung untuk deteksi tanpa kotak) | Cek kartu Kaggle · `PERLU_LOGIN` |
| `hf_nusaaksara` | Hugging Face [NusaAksara/NusaAksara](https://huggingface.co/datasets/NusaAksara/NusaAksara) (ACL 2025) | Multimodal segmentasi (7.516 contoh lintas aksara, subset Jawa ~100 halaman), OCR baris (6.265 contoh), transliterasi, dan terjemahan | Sumber pindaian buku cetak nyata, evaluasi transliterasi, dan Track B | Tidak mencantumkan field lisensi pada `README.md`/HF card · `PERLU_KONFIRMASI` |
| `mendeley_old_docs` | Mendeley Data [DOI 10.17632/ms6sjjpzwr.1](https://doi.org/10.17632/ms6sjjpzwr.1) (Suherli et al., BINUS, Des 2025) | Gabungan anotasi YOLOv8 + TrOCR dokumen historis & arsip British Library EAP | Ketahanan naskah kuno/dokumen lawas & Track B | **CC BY 4.0** (cek ketentuan EAP sebelum redistribusi) · `PERLU_KONFIRMASI` |
| `hf_rayfox_murda` | Hugging Face [RayFox52/Aksara-jawa-carakan-murda](https://huggingface.co/datasets/RayFox52/Aksara-jawa-carakan-murda) | Klasifikasi 27 kelas (`01_Ha..20_Nga` + 7 huruf Murda) | Referensi tambahan bila memperluas cakupan ke *Aksara Murda* | Tanpa tag lisensi pada HF card · `PERLU_KONFIRMASI` |
| `bl_eap` | British Library Endangered Archives Programme (`eap.bl.uk`) | Citra naskah historis manuskrip Jawa | Uji ketahanan domain naskah kuno (*held-out* stres) | Berbeda per koleksi · `PERLU_KONFIRMASI` |

---

## 2. Korpus Teks Jawa, Leksikon Dwibahasa, & Terjemahan

| ID | Sumber & URL | Isi & Format | Peran di v3.0 | Lisensi Terverifikasi & Status |
|---|---|---|---|---|
| `nusax_mt` | GitHub [IndoNLP/nusax](https://github.com/IndoNLP/nusax) & HF [indonlp/NusaX-MT](https://huggingface.co/datasets/indonlp/NusaX-MT) | Korpus paralel 12 bahasa termasuk Jawa (`jav`) ↔ Indonesia (`ind`) + kamus dwibahasa | Teks sintetis kalimat alami, kamus Jawa-ID, *word alignment*, evaluasi MT | Kode **Apache-2.0**, Data **CC-BY-SA-4.0** · `OK` |
| `nusawrites_mt` | GitHub [IndoNLP/nusa-writes](https://github.com/IndoNLP/nusa-writes) & HF [indonlp/nusatranslation_mt](https://huggingface.co/datasets/indonlp/nusatranslation_mt) | 72.444 kalimat paralel (*NusaTranslation*) + 57.409 paragraf (*NusaParagraph*) termasuk Jawa (`jav`) | Sumber kalimat panjang sintetis v2, korpus model bahasa karakter/kata, & pasangan terjemahan | **Apache-2.0** · `OK` |
| `sastra_jawa` | GitHub [nsulistiyawan/sastra-jawa](https://github.com/nsulistiyawan/sastra-jawa) | Data CSV Kamus Bahasa Jawa, *Dasanama*, *Bebasan lan Saloka* (peribahasa Jawa) | Memperluas `JAVANESE_DICT` dan daftar paribasan untuk generator kalimat | **MIT** · `OK` |
| `gatra1_jv` | Hugging Face [afrizalha/Gatra-1-Javanese](https://huggingface.co/datasets/afrizalha/Gatra-1-Javanese) | 560 contoh kalimat ragam *Jawa Krama* terstruktur | Menambah kosakata *Jawa Krama* pada leksikon dan korpus sintetis | **CC-BY-SA-4.0** · `OK` |
| `jv_wiki_dump` | Dump resmi Wikipedia Bahasa Jawa (`dumps.wikimedia.org/jvwiki/latest/`) | ~100 ribu artikel berbahasa Jawa (butuh filter ketat teks non-Jawa/markup) | Melatih model bahasa n-gram karakter & unigram/bigram kata Jawa | **CC BY-SA 4.0** · `OK` |
| `korpus_nusantara` | HF [SEACrowd/korpus_nusantara](https://huggingface.co/datasets/SEACrowd/korpus_nusantara) | Kalimat paralel Indonesia-Jawa dari studi Universitas Tanjungpura | Tambahan leksikon (jika dikonfirmasi oleh penulis) | Kartu dataset menulis `Unknown` · `PERLU_KONFIRMASI` |

---

## 3. Font OpenType & Perkakas Transliterasi Jawa

| ID | Sumber & URL | Isi & Kegunaan | Lisensi Terverifikasi & Status |
|---|---|---|---|
| `font_noto_jv` | GitHub [notofonts/javanese](https://github.com/notofonts/javanese) (`fonts/NotoSansJavanese.ttf`) | Render OpenType modern berstandar Unicode penuh (`U+A980..U+A9DF`) | **SIL OFL 1.1** · `OK` |
| `font_tuladha_ot` | GitHub [akufadhl/Tuladha-Jejeg-OT](https://github.com/akufadhl/Tuladha-Jejeg-OT) (`fonts/TuladhaJejegOT-Regular.ttf/.otf`) | Gaya cetak tradisional berbasis desain R.S. Wihananto (di-port ke OpenType AFDKO) | **SIL OFL 1.1** (terverifikasi di `OFL.txt` & header font) · `OK` |
| `font_javatext_win` | `C:/Windows/Fonts/javatext.ttf` (Microsoft Windows *Javanese Text*) | Saat ini dipakai di `generate_yolo_dataset.py` v2.5 | **Proprieter Microsoft** — hindari di data sintetis publik Fase 2 |
| `carakanjs` | GitHub [masnormen/carakanjs](https://github.com/masnormen/carakanjs) | Pustaka JS transliterasi dua arah Latin ↔ Aksara Jawa (rujukan aturan ortografi) | **MIT** · `OK` |
| `aksara_js` | GitHub [junwatu/aksara.js](https://github.com/junwatu/aksara.js) | Implementasi JS aturan konversi Latin ke Aksara Jawa | **MIT** · `OK` |
| `transliterasijawa` | GitHub [bennylin/transliterasijawa](https://github.com/bennylin/transliterasijawa) | Pedoman *Nulisa Aksara Jawa* (`PanduanNulisaAksaraJawa.pdf`) & rujukan ortografi Unicode | Tanpa file `LICENSE` eksplisit (rujukan tata tulis saja) |

---

## 4. Referensi Literatur, Blog Teknis, & Alternatif Kerangka Model (Anti-AGPL)

### A. Makalah & Implementasi OCR Aksara Jawa (Rujukan Arsitektur Track A & Track B)
1. **Deteksi Objek YOLO Aksara Jawa**:
   - Suparwito (2024), *"Image Detection Analysis for Javanese Character Using YOLOv9 Models"* (penilaian *mAP50* deteksi per karakter).
   - Evaluasi **YOLOv8-S vs EfficientDet vs ResNet-50** pada tulisan tangan + diakritik Aksara Jawa (menunjukkan pentingnya kapasitas fitur lebih dalam dibanding Nano maupun resolusi masukan lebih tinggi untuk sandhangan kecil).
   - Repo [hermansh-id/Aksara-Jawa-OCR](https://github.com/hermansh-id/Aksara-Jawa-OCR) (MIT) & model [hermanshid/yolo-aksara-jawa](https://huggingface.co/hermanshid/yolo-aksara-jawa) sebagai pembanding skema label & ambang NMS.
2. **Pengenal Baris Tanpa Segmentasi (Track B — CRNN-CTC & TrOCR)**:
   - Implementasi **CNN + BiLSTM + CTC Loss** untuk baris kata bersambung Aksara Jawa (*segmentation-free* menghindari pemotongan salah pada pasangan/sandhangan bersusun).
   - Panduan implementasi & pelatihan custom CTC/TrOCR: PyLessons / [`pythonlessons/mltu`](https://github.com/pythonlessons/mltu) (CRNN-CTC) serta LearnOpenCV TrOCR Fine-Tuning (`microsoft/trocr-base-printed` / `trocr-base-handwritten` dan varian kecil yang dapat diekspor ke ONNX untuk peramban).
   - Benchmark **NusaAksara (ACL 2025)** (*"A Multimodal and Multilingual Benchmark for Preserving Indonesian Indigenous Scripts"*) untuk evaluasi terstandardisasi OCR + transliterasi + terjemahan Aksara Jawa.

### B. Pilihan Pustaka Detektor Bebas Konflik Lisensi (Alternatif Ultralytics `AGPL-3.0`)
- **`ultralytics` (saat ini `8.4.218`)**: berlisensi **AGPL-3.0** — memicu kewajiban *copyleft* AGPL terhadap aplikasi web/model bila didistribusikan di luar lisensi Enterprise.
- **RT-DETRv2 / D-FINE (via PyTorch / Hugging Face `transformers`)**: kode berlisensi **Apache-2.0**, arsitektur *anchor-free & NMS-free* (menghilangkan masalah NMS antara *nglegena* vs *pasangan* yang bertumpuk!), mendukung ekspor langsung ke ONNX.
- **YOLOX (Megvii / Apache-2.0)**: desain *anchor-free decoupled head* berlisensi **Apache-2.0**, bobot ringan (`YOLOX-Nano` / `YOLOX-Tiny` / `YOLOX-S`) kompatibel luas dengan `onnxruntime-web`.

---

## 5. Referensi Teori & Repositori Praproses Lanjutan, Penolak Teks Latin, & Pencocokan Referensi (v4.0)

Untuk menjawab keluhan deteksi aksara pada foto buram, latar garis tembok/buku tulis, tinta pulpen berwarna, variasi lengkungan penulis, serta penyaringan teks Latin/non-aksara, arsitektur v4.0 merujuk teori dan repositori terbuka berikut:

1. **Penghapusan Garis Buku/Tembok & Koreksi Ambang pada Goresan Melengkung**:
   - **Tanaka et al. (ICDAR)** — *"Threshold Correction of Document Image Binarization for Ruled-line Extraction"*: mengoreksi binerisasi lokal (Sauvola) menggunakan fitur kelurusan lokal sehingga garis lurus (buku/tembok) dapat dilemahkan tanpa memutus lengkungan glif.
   - **Gold & Zesch** — *"CNN-Based Ruled Line Removal in Handwritten Documents"* ([catalpa-cl/ruled-line-removal-in-handwritten-documents](https://github.com/catalpa-cl/ruled-line-removal-in-handwritten-documents)).
   - **Morphological Top-Hat / Bottom-Hat + Binerisasi Adaptif Dokumen**: [rixmape/binarize-architectural-drawings](https://github.com/rixmape/binarize-architectural-drawings) & [opensuh/DocumentBinarization](https://github.com/opensuh/DocumentBinarization).
2. **Pemisahan Tinta Multi-Warna (*Colored Pen Separation*)**:
   - Transformasi ruang warna **CIELAB** (`L*, a*, b*`) dan jarak warna kromatik (`ΔE` terhadap estimasi latar kertas dominan) memisahkan tinta pulpen biru/merah/hijau serta pensil tipis dari latar kertas kuning, noda, atau bayangan bergradien lebih andal ketimbang konversi gray-luminance polos.
3. **Pemisahan Skrip Latin vs Jawa & Penolakan Non-Teks**:
   - **Epshtein, Ofek, & Wexler (CVPR 2010)** — *"Detecting Text in Natural Scenes with Stroke Width Transform (SWT)"* ([mypetyak/StrokeWidthTransform](https://github.com/mypetyak/StrokeWidthTransform)): menganalisis konsistensi ketebalan goresan lokal (*stroke width consistency*) untuk membedakan aksara vs retakan tembok/garis pinggir.
   - **Identifikasi Skrip Multi-Aksara & Hard Negative Mining**: merujuk [google-research/nisaba](https://github.com/google-research/nisaba) (keluarga skrip turunan Brāhmī) + pelatihan contoh campuran Latin-Jawa beranotasi selektif & patch penolak `non_aksara_distractors` di Stage-2 Metric Verifier.
4. **Invariansi Lengkungan (*Elastic Deformation*) & Pencocokan Prototipe**:
   - **Thin-Plate Spline (TPS) Spatial Perturbation**: memodelkan variasi lengkungan goresan non-rigid antar penulis saat pelatihan agar detektor dan embedding Stage-2 (*MobileNetV3 + ArcFace / Cosine Prototype Matching*) toleran terhadap pergeseran lekuk huruf tanpa "asal mirip".
