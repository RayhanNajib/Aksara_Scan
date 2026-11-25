# AksaraScan v2.5

**Pemeriksa & penerjemah kalimat Aksara Jawa (*Nglegena + Pasangan + Sandhangan + Tanda Baca Pada*) langsung dari kamera / gambar — 100% di browser, tanpa server.**

Web app client-only bermesin ganda (*Dual-Engine*):
1. **Engine v2.5 — YOLOv8-Nano ONNX (55 Kelas Atomik · 3 Gaya Font)** via `onnxruntime-web` + **Rekonstruktor Ortografi (`orthography.js`)** + **Pemisah Kata DP & Penerjemah Bahasa Indonesia (`javanese_lexicon.js`)**: mengenali 20 Aksara Nglegena, 20 Pasangan, 12 Sandhangan/Pangkon, dan 3 Tanda Baca *Pada* (`꧋`, `꧈`, `꧉`), lalu merangkainya otomatis menjadi Unicode, kalimat Latin tergabung (*scriptio continua* → kata), dan terjemahan kalimat Bahasa Indonesia.
2. **Engine v1.0 — Pipeline Pengolahan Citra Klasik + SVM Linear**: grayscale BT.601 → ambang adaptif Otsu → median 3×3 → morfologi → segmentasi CCL → normalisasi 96×96 → **SVM Linear One-vs-Rest** dari bobot float16 Base64.

Privasi penuh: gambar tidak pernah meninggalkan perangkat.

> Tugas Akhir — Muhammad Rayhan Najib (240533600012), Teknik Informasi, Universitas Negeri Malang.

##  Fitur

- **Mesin Ganda 1-Klik**: beralih bebas antara **YOLOv8-Nano ONNX (Full: Nglegena + Pasangan + Sandhangan)** dan **Pipeline Klasik PCD + SVM (20 Nglegena)**
- Kamera web/ponsel (HTML5 MediaDevices) + auto-rata 6 frame biar hasil konsisten
- **Auto-deteksi**: upload / tempel **Ctrl+V** / **seret gambar** ke halaman → langsung dipindai
- **Seleksi area ala Google Lens**: seret kotak pada satu aksara → *Pindai Area Pilihan* (ambang lokal lebih akurat untuk glif kecil)
- **Koreksi manual (human-in-the-loop)**: klik kotak deteksi → ganti aksara (20 kelas di mode SVM, 52 kelas di mode YOLOv8)
- **Lab Pipeline 8 tahap**: histogram Otsu, biner, median, morfologi, segmentasi, normalisasi, montase — semua bisa diaudit visual
- Kamus aksara Jawa lengkap + ekspor hasil JSON + contoh uji `sample_4.png` (Pasangan & Sandhangan)

##  Menjalankan

```bash
# pilih salah satu
python -m http.server 8899      # lalu buka http://127.0.0.1:8899/app/web_camscanner/
# atau jalankan app/web_camscanner/run.bat (Windows, klik dua kali)
```

Butuh: browser modern. Tanpa Node, tanpa backend, tanpa instalasi lain.

##  Struktur

```
app/web_camscanner/          # aplikasi web (index.html, app.js, orthography.js, aksara_yolo.onnx, model.json, PWA)
training/                    # trainer YOLOv8n + SVM + pembagian segmen dataset
  generate_yolo_dataset.py   # generator hibrid 1.050 halaman (Jawa.zip + OpenType javatext.ttf RAQM)
  train_yolo.py              # pelatih YOLOv8n (52 kelas) -> app/web_camscanner/aksara_yolo.onnx
  split_dataset.py           # -> dataset/split/{train,validation,test} (70/15/15, seed 42)
  train_svm_v2.py            # augmentasi + Gaussian blur + LinearSVC -> model.json/js
dataset/
  hanacaraka/                # 1.579 citra tulisan tangan (20 kelas) — CC0, milik orang lain ⚠️
  split/                     # hasil pembagian train/validation/test
  manifest.csv               # 1 baris per citra: path, kelas, split, lisensi sumber
docs/                        # PRD, arsitektur, spesifikasi dataset/model, rencana uji
```

##  Model

| Engine | Detail & Evaluasi |
|---|---|
| **YOLOv8-Nano ONNX (v2.0)** | 52 kelas atomik (`nglegena` + `pasangan` + `sandhangan`), input `416×416` (`12.11 MB` di `aksara_yolo.onnx`), **mAP50 `98.4%`** (Precision `97.08%` · Recall `95.76%`) |
| **LinearSVC SVM (v1.0)** | 20 Nglegena, fitur kanvas biner 96×96 Gaussian-blur (σ 2.5), float16 Base64 (`~484 KB` di `model.json`), **68.4%** bersih · **53.2%** simulasi foto |

##  Lisensi & Kredit dataset (PENTING)

Kode & model repo ini: **MIT** (Muhammad Rayhan Najib). **Dataset BUKAN milik proyek ini** dan tetap di bawah lisensi pembuatnya:

| Dataset | Pemilik | Lisensi |
|---|---|---|
| `dataset/hanacaraka/` (data latih utama) | **vzrenggamani** + kontributor (Aditya Bayu Perdana, SMA Negeri 1 Wonoayu, Sakura Fukoka Studio, para partisipan kuesioner) | [CC0-1.0](https://github.com/vzrenggamani/aksarajawa-hanacaraka) |
| Dataset Mendeley *Indonesian Local Script Characters* (eksperimen) | **Aditya Firman Ihsan** | [CC BY 4.0](https://doi.org/10.17632/vfj32bpjsf.1) |
| Font Noto Sans Javanese (dimuat runtime, tidak didistribusikan) | Google | OFL 1.1 |

Rincian lengkap, format sitasi, dan cara pembagian segmen: **[DATASET_CREDITS.md](DATASET_CREDITS.md)**.
