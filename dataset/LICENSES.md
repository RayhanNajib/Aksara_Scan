# Audit Lisensi Sumber Data, Font, & Model — AksaraScan v4.0 (`dataset/LICENSES.md`)

> Mematuhi **Aturan Besi #1**: Setiap sumber yang digunakan dalam pelatihan, evaluasi, maupun prototipe referensi wajib memiliki status lisensi terverifikasi (`DIPAKAI`, `HANYA-EVALUASI`, atau `DIBLOKIR`) beserta bukti URL, tanggal pemeriksaan, dan ketentuan atribusi. Sumber tanpa lisensi jelas atau berisiko melanggar **otomatis `DIBLOKIR`**.

Tanggal verifikasi mutakhir: **2026-11-27**

---

## 1. Dataset Citra Aksara Jawa

| ID Sumber | Nama & Pembuat | Lokasi Lokal / URL | Status | Lisensi Terverifikasi | Bukti & Catatan Kepatuhan |
|---|---|---|---|---|---|
| `hanacaraka_vzr` | Aksara Jawa / Hanacaraka Digital Handwriting (`R. P. Nugroho / vzrenggamani` + Aditya Bayu Perdana, SMA N 1 Wonoayu, Sakura Fukoka Studio) | [dataset/hanacaraka](file:///c:/Users/WA/Downloads/TUGAS%20PKKMB/Citra%20Komputer/Aksara_Scan/dataset/hanacaraka/LICENSE) · https://github.com/vzrenggamani/aksarajawa-hanacaraka | **DIPAKAI** | **CC0-1.0** (*Public Domain Dedication*) | Berkas `LICENSE` & `LICENSE.md` di repo. Aman untuk latih, prototipe, dan distribusi bobot MIT. Split v4 wajib *writer-disjoint* (79 penulis unik dipisah antar split). |
| `mendeley_jawa_zip` | Indonesian Local Script Characters — Subset `Jawa.zip` (`Aditya Firman Ihsan`, Telkom University) | [dataset/indonesian-scripts/DOI.txt](file:///c:/Users/WA/Downloads/TUGAS%20PKKMB/Citra%20Komputer/Aksara_Scan/dataset/indonesian-scripts/DOI.txt) · https://doi.org/10.17632/vfj32bpjsf.1 | **DIPAKAI** | **CC BY 4.0** | Terverifikasi `HTTP 200` pada Mendeley Data v1 (`13.256` citra: *Vokal A*, *variations*, dan suku kata *Vokal E/I/O/U/È*). **Wajib mencantumkan atribusi** di `DATASET_CREDITS.md` dan tab `Tentang`. |
| `hf_setilanaji_ocr` | Aksara Jawa OCR Multi-Line (`setilanaji`) | `dataset/external/setilanaji_aksara_jawa_ocr` · https://huggingface.co/datasets/setilanaji/aksara-jawa-ocr | **DIPAKAI** | **Apache-2.0** | Terverifikasi via Hugging Face API (`tags: ["license:apache-2.0"]`). Memuat `2.209` baris beranotasi Unicode; pemisahan halaman memakai prefix `split/page` agar nol tumpang tindih. |
| `hf_nusaaksara` | NusaAksara: Indonesian Indigenous Scripts Benchmark (`NusaAksara`, ACL 2025) | `dataset/external/eval_only_nusaaksara` · https://huggingface.co/datasets/NusaAksara/NusaAksara · https://aclanthology.org/ (ACL 2025) | **HANYA-EVALUASI** | **CC BY-SA 4.0** (tercantum pada publikasi ACL 2025; field card HF kosong) | Dipakai **secara ketat hanya sebagai set evaluasi eksternal & bagian `eval/heldout_real`** — **tidak pernah** dimasukkan ke pelatihan bobot MIT agar tidak memicu kewajiban *ShareAlike* pada bobot model. |
| `rf_fawwaz_det` | Javanese Handwriting Object Detection (`Fawwaz Zaini Ahmad`) | Roboflow Universe (perlu API key) | **DIBLOKIR** *(menunggu unduhan manual)* | `CC BY 4.0` (klaim katalog; `HTTP 403` tanpa login/key) | Tidak dapat diakses otomatis tanpa login/API key (`HTTP 403`). Skrip `tools/datasets/ingest_roboflow.py` disiapkan jika pengguna mengekspor ZIP & menyertakan tangkapan bukti lisensi. |
| `rf_fasel_sandhangan` | `aksarajawasandangan-2` (`fasel`) | Roboflow Universe (`4.137` citra) | **DIBLOKIR** *(menunggu unduhan manual)* | `CC BY 4.0` (perlu API key & audit label kelas) | Sama seperti di atas (`HTTP 403`). |
| `rf_parelate` / `rf_febgabut` | `pa-relate/aksara-jawa` & `febgabut/aksara-jawa-classification` | https://universe.roboflow.com/pa-relate/aksara-jawa · https://universe.roboflow.com/febgabut/aksara-jawa-classification | **DIBLOKIR** *(menunggu unduhan manual)* | Perlu konfirmasi per proyek (`HTTP 403` headless) | Diblokir hingga unduhan resmi + bukti kartu lisensi tersedia. |
| `kg_hermansugiharto` / `kg_phiard` | Kaggle Aksara Jawa YOLO v5 & Handwriting on Paper | https://www.kaggle.com/datasets/hermansugiharto/aksara-jawa-yolo-v5-dataset | **DIBLOKIR** *(menunggu API Kaggle)* | Perlu login Kaggle + verifikasi kolom License | Skrip `tools/datasets/ingest_kaggle.py` tersedia apabila arsip ditaruh di `dataset/external/kg_*` dengan bukti lisensi sah. |
| `hf_rayfox_murda` | `RayFox52/Aksara-jawa-carakan-murda` (27 kelas termasuk 7 Murda) | https://huggingface.co/datasets/RayFox52/Aksara-jawa-carakan-murda | **DIBLOKIR** | **Tidak Ada (`None`)** pada metadata HF & README | Terverifikasi `cardData.license == None` (2026-11-27). Dilarang keras dilatih demi kepatuhan MIT. |
| `gh_iqballx_hanacaraka` | `IqbalLx/Hanacaraka-AI` | https://github.com/IqbalLx/Hanacaraka-AI | **DIBLOKIR** | **Tidak Ada (`None`)** via GitHub API | Tidak memiliki berkas `LICENSE` (`spdx_id: None`, 2026-11-27). |
| `mendeley_old_docs` | Digitization of Old Indonesian Documents (Suherli et al., BINUS) | https://doi.org/10.17632/ms6sjjpzwr.1 | **DIBLOKIR** | Mendeley `CC BY 4.0` + citra British Library EAP | Ketentuan ulang distribusi tiap koleksi British Library EAP perlu klarifikasi sebelum dipakai latih. |

---

## 2. Font Terbuka untuk Generator Sintetis v4 (Jawa & Distraktor Latin)

Seluruh berkas font diuji langsung header SFNT `name` (`nameID 13/14`) via `fontTools.ttLib.TTFont`:

| Kelompok | Berkas / Pembuat | Lokasi Lokal | Status | Lisensi Terverifikasi |
|---|---|---|---|---|
| **Aksara Jawa (11 OFL + 1 Apache-2.0)** | `NotoSansJavanese.ttf`, `NotoSansJavanese-Bold.ttf` (`notofonts/javanese`), `TuladhaJejegOT-Regular.{ttf,otf}` (`akufadhl`), `Bharatayuddha.ttf`, `Chandrasa.ttf`, `Righma_Ciddhi.ttf`, `TUNJUNG_BIRU.ttf`, `Tunjung_putih.ttf`, `Warastra.ttf`, `YBG_Kramawirya_Yayat_Suhiryatna.ttf`, `Sehulbari_Arif_Budiarto.ttf` | [dataset/external/fonts](file:///c:/Users/WA/Downloads/TUGAS%20PKKMB/Citra%20Komputer/Aksara_Scan/dataset/external/fonts/OFL.txt) | **DIPAKAI** | **11 SIL OFL 1.1** + **1 Apache-2.0** (`Sehulbari_Arif_Budiarto.ttf`) |
| **Distraktor Latin (3 font tulisan tangan/sans/serif)** | `Caveat-Regular.ttf`, `PlusJakartaSans-Regular.ttf`, `Newsreader-Regular.ttf` (`google/fonts`) | [dataset/external/fonts_latin_ofl](file:///c:/Users/WA/Downloads/TUGAS%20PKKMB/Citra%20Komputer/Aksara_Scan/dataset/external/fonts_latin_ofl/OFL_google_fonts.txt) | **DIPAKAI** | **SIL OFL 1.1** (terverifikasi header font & `OFL.txt`) |
| **Font Proprieter Windows** | `javatext.ttf` (Microsoft *Javanese Text*) | *(tidak ada di repo)* | **DIBLOKIR** | **Proprieter Microsoft** — dilarang keras dalam sintesis maupun bundel |

---

## 3. Korpus Bahasa Jawa & Leksikon

| ID | Sumber & URL | Status | Lisensi Terverifikasi |
|---|---|---|---|
| `nusax` | https://github.com/IndoNLP/nusax | **DIPAKAI** | Data **CC-BY-SA-4.0** / Kode **Apache-2.0** |
| `nusa_writes` | https://github.com/IndoNLP/nusa-writes | **DIPAKAI** | **Apache-2.0** |
| `sastra_jawa` | https://github.com/nsulistiyawan/sastra-jawa | **DIPAKAI** | **MIT** |
| `gatra1_javanese` | https://huggingface.co/datasets/afrizalha/Gatra-1-Javanese | **DIPAKAI** | **CC-BY-SA-4.0** |
| `javanese_conceptnet` | https://huggingface.co/datasets/DGurgurov/javanese_conceptnet | **DIPAKAI** | **MIT** |
| `jvwiki` | https://dumps.wikimedia.org/jvwiki/latest/ | **DIPAKAI** | **CC BY-SA 4.0** |
| `public_javanese` | https://huggingface.co/datasets/thesimonharms/public-javanese-dataset | **DIPAKAI** | **CC-BY-SA-3.0 / MIT** |
| `seacrowd_korpus_nusantara` | `SEACrowd/korpus_nusantara` | **DIBLOKIR** | Metadata lisensi `Unknown` |

---

## 4. Pustaka & Bobot Model Referensi

| Komponen | Lokasi / URL | Status | Lisensi | Keterangan |
|---|---|---|---|---|
| **Jalur Latih Baru v4** | **YOLOX** (Megvii), **MobileNetV3 / EfficientNet** (`torchvision`), **CRNN-CTC** | **DIPAKAI** | **Apache-2.0 / BSD-3-Clause / MIT** | Kompatibel penuh dengan repositori **MIT**. |
| `hermansh-id/Aksara-Jawa-OCR` | https://github.com/hermansh-id/Aksara-Jawa-OCR | **DIPAKAI** *(rujukan arsitektur/kode)* | **MIT** | Repositori hanya berisi kode/skrip OCR (tanpa berkas dataset terbundel). |
| `hermanshid/yolo-aksara-jawa` | https://huggingface.co/hermanshid/yolo-aksara-jawa | **DIBLOKIR** | `None` di HF card | Tidak dipakai sebagai bobot rilis karena kartu model tidak mencantumkan lisensi. |
| `ultralytics` (`YOLOv8` v2.5/v3.1) | `app/web_camscanner/aksara_yolo.onnx` (baseline lama) | **DIHENTIKAN DI V4** | **AGPL-3.0** | Dipertahankan hanya sebagai baseline pembanding `C0`; model v4 wajib dilatih dari kode **Apache-2.0 / MIT** bersih. |
| Pipeline Peningkatan Citra & CV | Tanaka et al. (ICDAR) *Ruled-line Threshold*; Gold & Zesch (`catalpa-cl/ruled-line-removal-in-handwritten-documents`); Epshtein et al. (CVPR 2010) *SWT* | **RUJUKAN TEORI & DESAIN** | Akademik / Open Source | Referensi desain praproses pemisah tinta berwarna & penghapus garis tembok/buku pada Fase 1 & 3. |
