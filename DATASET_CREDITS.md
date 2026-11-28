# Dataset Credits & Licenses

AksaraScan uses **third-party datasets**. This project does not own the
handwriting data. Below is who made what, under which license, and how the
data is divided into segments for training.

## 1. Primary training dataset — Hanacaraka handwritten glyphs

| | |
|---|---|
| Name | Aksara Jawa / Hanacaraka (digital handwriting) |
| Author / Owner | **R. P. Nugroho, GitHub **vzrenggamani**** and community contributors |
| Repository | https://github.com/vzrenggamani/aksarajawa-hanacaraka |
| Kaggle mirror | https://www.kaggle.com/datasets/vzrenggamani/hanacaraka |
| License | **CC0 1.0 Universal** (public domain dedication) — plus PDDL for the database |
| Contents used | 1,579 images, 20 nglegena classes (~79 per class) |
| Location here | `dataset/hanacaraka/` (unmodified copy; original LICENSE, LICENSE.md and README kept inside) |
| Used for | Training/validation/test of the SVM model in `training/` |

Original contributors credited in the upstream README:

> 1. **Aditya Bayu Perdana** (@by.abay) — first contributor, manual handwriting
> 2. **Warga SMA Negeri 1 Wonoayu** — helped collect the data
> 3. **Sakura Fukoka Studio Data Branch** — data cleanup
> 4. All internet participants who submitted digital handwriting via the online questionnaire

If you use `dataset/hanacaraka/`, please cite/credit vzrenggamani and the
contributors above. CC0 imposes no legal requirement, but crediting the
people who drew 1,500+ glyphs by hand is the decent thing to do.

## 2. Auxiliary dataset — Indonesian Local Script Characters

| | |
|---|---|
| Name | Indonesian Local Script Characters |
| Author / Owner | **Aditya Firman Ihsan** |
| DOI | https://doi.org/10.17632/vfj32bpjsf (Mendeley Data, v1, 2024) |
| License | **CC BY 4.0** — attribution is legally required |
| Contents | ~13,400 images incl. Javanese with sandhangan (subsets) |
| Location here | `dataset/indonesian-scripts/` (`DOI.txt` & `Jawa.zip` lokal 13.256 citra) |
| Used for | Eksperimen ketahanan & data latih tambahan (11.060 Nglegena atomik `Vokal A` + `variations` terpisah split; 2.196 citra suku kata ber-sandhangan untuk pengenalan gabungan/Track B) |

Citation:
> Ihsan, Aditya Firman (2024), 'Indonesian Local Script Characters', Mendeley Data, v1, doi:10.17632/vfj32bpjsf.1

## 2b. Additional Verified Open Datasets & Corpora (v3.0 Expansion)

Seluruh sumber berikut diverifikasi lisensinya pada **25 November 2026** (daftar mesin di `tools/datasets/datasets.yaml`, tanda terima di `dataset/external/_download_receipts.json`):

| Sumber | Penulis / Pemilik | URL | Lisensi | Tanggal Unduh | Jumlah & Perubahan |
|---|---|---|---|---|---|
| **Aksara Jawa OCR Multi-Line** | `setilanaji` | https://huggingface.co/datasets/setilanaji/aksara-jawa-ocr | **Apache-2.0** | 2026-11-25 | 1.158 berkas asal → dipotong & dide-duplikasi menjadi **2.205 baris gambar** (2.059 train / 150 val) beranotasi Unicode |
| **NusaX & NusaX-MT** | IndoNLP | https://github.com/IndoNLP/nusax | **CC-BY-SA-4.0** (data) / **Apache-2.0** (kode) | 2026-11-25 | `datasets/lexicon/javanese.csv` + MT parallel train/valid/test; difilter ke kalimat & kamus Jawa bersih |
| **NusaWrites (`nusa_kalimat` & `nusa_alinea`)** | IndoNLP | https://github.com/IndoNLP/nusa-writes | **Apache-2.0** | 2026-11-25 | 6 berkas CSV bahasa Jawa (`jav`); dipecah menjadi kalimat tervalidasi blok Unicode |
| **Sastra Jawa Dictionary & Dasanama** | `nsulistiyawan` | https://github.com/nsulistiyawan/sastra-jawa | **MIT** | 2026-11-25 | `dictionary.csv` + `dasanama.csv`; digabungkan menjadi **10.588 entri** leksikon dwibahasa |
| **Gatra-1 Javanese** | `afrizalha` | https://huggingface.co/datasets/afrizalha/Gatra-1-Javanese | **CC-BY-SA-4.0** | 2026-11-25 | 560 contoh ragam Jawa Krama (Parquet) |
| **Wikipedia Bahasa Jawa (`jvwiki`)** | Wikimedia & Kontributor | https://dumps.wikimedia.org/jvwiki/latest/ | **CC BY-SA 4.0** | 2026-11-25 | Dump XML 58,1 MB; disaring markup wiki & angka hingga menghasilkan bagian dari **71.228 kalimat** Jawa tervalidasi |
| **Noto Sans Javanese (Regular & Bold) & Tuladha Jejeg OT** | Google Noto / R.S. Wihananto & Fadhl Haqq | https://github.com/notofonts/javanese & https://github.com/akufadhl/Tuladha-Jejeg-OT | **SIL OFL 1.1** | 2026-11-25 | Font OpenType (`Regular`, `Bold`, `TuladhaJejegOT`) untuk generator sintetis v2 & v3 (`OFL.txt` terverifikasi; `javatext.ttf` proprieter tidak dipakai) |
| **Font Aksara Jawa Terbuka Komunitas (`bennylin.github.io`)** | Aditya Bayu Perdana, Arif Budiarto, R.S. Wihananto, Yayat Suhiryatna, Bennylin | https://github.com/bennylin/bennylin.github.io/tree/master/keyboards/fonts | **SIL OFL 1.1** (`Bharatayuddha`, `Chandrasa`, `Righma_Ciddhi`, `TUNJUNG_BIRU`, `Tunjung_putih`, `Warastra`, `YBG_Kramawirya`) & **Apache-2.0** (`Sehulbari_Arif_Budiarto`) | 2026-11-25 | 8 font Unicode Jawa tambahan (metadata tabel `name` TTF teraudit; font CC-BY-NC-ND & All-Rights-Reserved dikecualikan) |
| **Javanese ConceptNet (`DGurgurov/javanese_conceptnet`)** | D. Gurgurov | https://huggingface.co/datasets/DGurgurov/javanese_conceptnet | **MIT** | 2026-11-25 | `train_cn_jv.csv` (171 KB); referensi kosakata & grafem Jawa |
| **Public-Domain Javanese Primer Index (`blajar-maca-jawa-1996`)** | S. Harms / Wikisource | https://huggingface.co/datasets/thesimonharms/public-javanese-dataset | **CC-BY-SA 3.0** / **MIT** | 2026-11-25 | `sources.yaml` & `WS_lesson_index.clean.json` untuk referensi tata baca |

## 2c. Perluasan v4.0 (Split Writer-Disjoint, NusaAksara Eval-Only, & Distraktor)

- Seluruh pembagian split v4 dikendalikan oleh `dataset/external/manifests_v4/` dan lolos pemeriksaan zero-leakage (`tools/check_leakage.py`, `21.457` rekaman, `0` tumpang tindih).
- **`NusaAksara` (ACL 2025, CC BY-SA 4.0)**: `48` cuplikan (`18` di set terkunci `eval/heldout_real/`) digunakan secara ketat sebagai **HANYA-EVALUASI** tanpa ikut melatih bobot/prototipe model MIT.
- **`synth_v4` & `non_aksara_distractors_v4`**: `1.050` citra sintetis deforma TPS + `1.050` patch distraktor Latin/garis tembok menggunakan **11 font Jawa SIL OFL 1.1 + 1 Apache-2.0** dan **3 font Latin SIL OFL 1.1** (`Caveat`, `Plus Jakarta Sans`, `Newsreader`). Detail lengkap tersimpan di [dataset/LICENSES.md](dataset/LICENSES.md) dan [docs/DATASET_V4.md](docs/DATASET_V4.md).

## 3. Segmentation of the data (legacy v1 & v4 summary)

The 1,579 hanacaraka images are deterministically split per class
(`training/split_dataset.py`, seed 42):

| Segment | Images | Share | Purpose |
|---|---|---|---|
| `dataset/split/train/` | 1,100 | 70% | Model fitting (+9x augmentation each during training) |
| `dataset/split/validation/` | 240 | 15% | Hyperparameter selection (C, augmentation blend) |
| `dataset/split/test/` | 239 | 15% | Final honest evaluation (never seen in training) |

Every file is traceable: `dataset/manifest.csv` lists
`path, class, split, source_license`.

Held-out test metrics of the shipped model (`app/web_camscanner/model.json`):
**68.4%** on clean binarized test segments, **53.2%** on photo-simulated
variants (blur/noise/rotation/illumination).

## 4. Fonts & Runtime Bundle (v4.0 Self-Hosted Offline)

- **Self-Hosted UI & Aksara Fonts (`app/web_camscanner/vendor/fonts/`, `994 KB`)**:
  - `NotoSansJavanese.ttf` & `NotoSansJavanese-Bold.ttf` — **SIL Open Font License 1.1** (`notofonts/javanese`)
  - `PlusJakartaSans-Regular.ttf` — **SIL Open Font License 1.1**
  - `Newsreader-Regular.ttf` — **SIL Open Font License 1.1**
- Seluruh font dibundel langsung secara lokal di repositori dan di-precache oleh Service Worker untuk operasi PWA offline penuh (**0 permintaan CDN eksternal** saat muat maupun inferensi). Font proprieter `javatext.ttf` tidak digunakan di mana pun pada jalur produksi maupun pelatihan v4.

## 5. What this repository owns

All code (`app/`, `training/`, `docs/`) and the trained model weights
(`model.json` / `model.js`) are original work by **Muhammad Rayhan Najib**
(RayhanNajib), Universitas Negeri Malang, released under the MIT license in
`LICENSE` — except the third-party items listed above, which keep their own
licenses.
