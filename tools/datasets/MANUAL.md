# Panduan Akuisisi Manual Dataset (Login / API Key / Konfirmasi Lisensi)

Sesuai **Aturan Keras #4** AksaraScan: agen tidak boleh mengakali dinding login atau menebak lisensi. Gunakan petunjuk langkah demi langkah berikut bila Anda ingin menambahkan sumber berstatus `PERLU_LOGIN` atau `PERLU_KONFIRMASI`.

---

## 1. Sumber Roboflow Universe (`PERLU_LOGIN`)

### A. *Javanese Handwriting Object Detection - Aksara Jawa* (`Fawwaz Zaini Ahmad`, 1.464 citra, CC BY 4.0)
### B. *aksarajawasandangan-2* (`fasel`, 4.137 citra, CC BY 4.0)

1. Buka [Roboflow Universe](https://universe.roboflow.com/) dan masuk dengan akun Anda.
2. Cari proyek *"Javanese Handwriting Object Detection-Aksara Jawa"* (Fawwaz Zaini Ahmad) dan *"aksarajawasandangan-2"* (fasel).
3. Di tab **License**, pastikan tertera **`CC BY 4.0`**.
4. Klik **Download Dataset** $\rightarrow$ pilih format **YOLOv8** $\rightarrow$ **Download zip to computer** (atau salin snippet `roboflow` Python API dengan key Anda).
5. Ekstrak masing-masing ke:
   - `dataset/external/rf_fawwaz/`
   - `dataset/external/rf_fasel_sandhangan/`
6. Pastikan skrip konversi memeriksa skema `data.yaml` mereka terlebih dahulu sebelum memetakan ke 55 kelas AksaraScan.

---

## 2. Sumber Kaggle (`PERLU_LOGIN`)

### A. *Aksara Jawa YOLO v5 Dataset* (`hermansugiharto/aksara-jawa-yolo-v5-dataset`)
### B. *Aksara Jawa Custom / Handwriting on Paper* (`Phiard`)

1. Masuk ke [Kaggle](https://www.kaggle.com/) dan buka halaman dataset masing-masing.
2. Periksa kotak **License** di bagian kanan halaman dataset sebelum mengunduh (pastikan bukan *Unknown* atau *Non-Commercial* tanpa persetujuan).
3. Pasang API token (`~/.kaggle/kaggle.json`) lalu jalankan:
   ```bash
   kaggle datasets download -d hermansugiharto/aksara-jawa-yolo-v5-dataset -p dataset/external/kg_hermansugiharto --unzip
   ```
4. Untuk dataset potongan per folder (`Phiard`), ekstrak ke `dataset/external/kg_phiard/` (dipakai sebagai sumber tekstur goresan tulisan tangan untuk generator sintetis v2, bukan kotak deteksi langsung).

---

## 3. Sumber Hugging Face / Mendeley Berstatus `PERLU_KONFIRMASI`

1. **`NusaAksara/NusaAksara`** ([HF Card](https://huggingface.co/datasets/NusaAksara/NusaAksara)):
   - Pada `README.md` HF per 25 November 2026 tidak tercantum field `license:`. Hubungi penulis / cek publikasi ACL 2025 resmi apakah lisensinya mengizinkan distribusi bobot komersial atau riset terbuka (`CC BY-SA` / `CC BY-NC`).
2. **`Digitization of Old Indonesian Documents using YOLO and OCR`** (`DOI: 10.17632/ms6sjjpzwr.1`):
   - Berlisensi `CC BY 4.0` di Mendeley Data, namun memuat citra dari British Library Endangered Archives Programme (*EAP*) yang aturan koleksi aslinya bisa bervariasi (*non-commercial research* vs terbuka). Unduh manual bila sudah memastikan ketentuan koleksi Jawa terkait.
3. **`RayFox52/Aksara-jawa-carakan-murda`** & **`SEACrowd/korpus_nusantara`**:
   - Kedua kartu dataset belum mencantumkan lisensi eksplisit (`Unknown` / kosong), sehingga **dikecualikan** dari pelatihan v3.0 demi kepatuhan lisensi repo MIT.
