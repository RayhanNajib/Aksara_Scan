# AksaraScan 📜✨

**Pemeriksa aksara Jawa (hanacaraka nglegena) langsung dari kamera / gambar — 100% di browser, tanpa server.**

Web app client-only: foto/unggah tulisan tangan → pipeline Computer Vision (grayscale BT.601 → ambang adaptif → median → morfologi → segmentasi CCL → normalisasi 96×96) → **SVM Linear One-vs-Rest** yang berjalan penuh di sisi klien dari bobot float16 Base64. Privasi penuh: gambar tidak pernah meninggalkan perangkat.

> Tugas Akhir — Muhammad Rayhan Najib (240533600012), Teknik Informasi, Universitas Negeri Malang.

## ✨ Fitur

- 📷 Kamera web/ponsel (HTML5 MediaDevices) + auto-rata 6 frame biar hasil konsisten
- 🖱️ **Auto-deteksi**: upload / tempel **Ctrl+V** / **seret gambar** ke halaman → langsung dipindai
- 🔎 **Seleksi area ala Google Lens**: seret kotak pada satu aksara → *Pindai Area Pilihan* (ambang lokal lebih akurat untuk glif kecil)
- 🧑‍⚖️ **Koreksi manual (human-in-the-loop)**: klik kotak deteksi → ganti aksara, akurasi akhir milik pengguna
- 🧪 **Lab Pipeline 8 tahap**: histogram Otsu, biner, median, morfologi, segmentasi, normalisasi, montase SVM — semua bisa diaudit visual
- 📖 Kamus aksara Jawa lengkap + ekspor hasil JSON

## 🚀 Menjalankan

```bash
# pilih salah satu
python -m http.server 8899      # lalu buka http://127.0.0.1:8899/app/web_camscanner/
# atau jalankan app/web_camscanner/run.bat (Windows, klik dua kali)
```

Butuh: browser modern. Tanpa Node, tanpa backend, tanpa instalasi lain.

## 📁 Struktur

```
app/web_camscanner/   # aplikasi web (index.html, app.js, styles.css, model.json, PWA)
training/             # trainer SVM + pembagian segmen dataset
  split_dataset.py    # -> dataset/split/{train,validation,test} (70/15/15, seed 42)
  train_svm_v2.py     # augmentasi + Gaussian blur + LinearSVC -> model.json/js
dataset/
  hanacaraka/         # 1.579 citra tulisan tangan (20 kelas) — CC0, milik orang lain ⚠️
  split/              # hasil pembagian train/validation/test
  manifest.csv        # 1 baris per citra: path, kelas, split, lisensi sumber
docs/                 # PRD, arsitektur, spesifikasi dataset/model, rencana uji
```

## 🧬 Model

| | |
|---|---|
| Arsitektur | LinearSVC one-vs-rest, fitur = canvas biner 96×96 di-Gaussian-blur (σ 2.5) |
| Bobot | float16 → Base64 dalam `model.json` (~484 KB), inference murni JS |
| Data latih | 1.100 citra train × 10 varian augmentasi, holdout uji 239 citra asli **tidak pernah dilihat model** |
| Akurasi | **68.4%** bersih · **53.2%** simulasi foto |

## ⚖️ Lisensi & Kredit dataset (PENTING)

Kode & model repo ini: **MIT** (Muhammad Rayhan Najib). **Dataset BUKAN milik proyek ini** dan tetap di bawah lisensi pembuatnya:

| Dataset | Pemilik | Lisensi |
|---|---|---|
| `dataset/hanacaraka/` (data latih utama) | **vzrenggamani** + kontributor (Aditya Bayu Perdana, SMA Negeri 1 Wonoayu, Sakura Fukoka Studio, para partisipan kuesioner) | [CC0-1.0](https://github.com/vzrenggamani/aksarajawa-hanacaraka) |
| Dataset Mendeley *Indonesian Local Script Characters* (eksperimen) | **Aditya Firman Ihsan** | [CC BY 4.0](https://doi.org/10.17632/vfj32bpjsf.1) |
| Font Noto Sans Javanese (dimuat runtime, tidak didistribusikan) | Google | OFL 1.1 |

Rincian lengkap, format sitasi, dan cara pembagian segmen: **[DATASET_CREDITS.md](DATASET_CREDITS.md)**.
