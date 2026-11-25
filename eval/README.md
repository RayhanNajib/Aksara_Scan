# Protokol Set Uji & Validasi Nyata — AksaraScan

Dokumen ini mendefinisikan protokol wajib pengumpulan, anotasi, pembagian data, dan penguncian manifes untuk set data tulisan tangan dunia nyata (`real_train`, `real_val`, `real_test`), agar angka evaluasi AksaraScan v3.0 mencerminkan performa lapangan pada foto buku tulis dan bukan artefak generator sintetis.

---

## 1. Perbedaan Tegas Antar-Set

| Direktori / Manifes | Fungsi | Boleh Dipakai untuk Tuning? | Boleh Dipakai untuk Klaim Akhir? |
|---|---|---|---|
| `eval/failures/` | **Set diagnostik kecil awal** (foto gagal laporan user & sampel uji internal) | Ya (untuk analisis akar masalah) | **TIDAK** (ukuran terlalu kecil dan terbias laporan kegagalan) |
| `eval/real_train/` | Fine-tuning tahap 2 & penambangan *hard negative* | Ya | **TIDAK** |
| `eval/real_val/` | Seleksi checkpoint, ablasi augmentasi, penyetelan bobot dekoder beam & ambang conf | Ya (hanya untuk seleksi) | **TIDAK** |
| `eval/real_test/` | Evaluasi akhir Gerbang Rilis Fase 5 | **TIDAK PERNAH** | **YA** (dikunci SHA-256 hingga rilis) |

---

## 2. Target Ukuran, Penulis, dan Komposisi Konten

- **Ukuran Minimum**: **150 halaman atau baris**, target ideal **300+ halaman/baris**.
- **Jumlah Penulis Minimum**: **10 penulis berbeda**, target ideal **20+ penulis berbeda** (mahasiswa, siswa, guru bahasa Jawa).
- **Komposisi Teks Wajib**:
  1. **50%** — Kalimat bebas dengan kombinasi lengkap *nglegena + pasangan + sandhangan*.
  2. **20%** — Kata tunggal berisi kasus tersulit (*taling* `ꦺ` vs *pepet* `ꦼ`, tumpukan *pasangan*, *pasangan + suku/wulu*, kata berakhir *cecak* `ꦁ` / *layar* `ꦂ` / *wignyan* `ꦃ` / *pangkon* `꧀`).
  3. **30%** — Kalimat panjang *scriptio continua* (minimal **8 kata tanpa spasi**, lengkap dengan *pada adeg-adeg* `꧋`, *pada lingsa* `꧈`, *pada lungsi* `꧉`).

---

## 3. Variasi Lingkungan Wajib

Setiap kelompok penulis/sesi wajib mencakup variasi fisik berikut:
- **Alat tulis**: pulpen tinta hitam, pulpen tinta biru, dan pensil (ketebalan bervariasi).
- **Media kertas**: kertas HVS polos, buku tulis bergaris, dan kertas berpetak (*grid*).
- **Pencahayaan**: cahaya siang tidak langsung, lampu ruangan kuning/hangat, dan bayangan sebagian (mis. bayangan jari atau ponsel).
- **Perangkat & Geometri**: minimal **2 model kamera ponsel berbeda**, variasi jarak, dan kemiringan foto alami hingga ±10°.

---

## 4. Aturan Ground Truth & Anotasi

1. **String Kebenaran Dasar (Wajib Manual)**:
   - Setiap gambar wajib memiliki transkripsi Unicode Blok Jawa (`U+A980..U+A9DF`) dan bacaan Latin yang **diketik manual** oleh penulis atau peninjau yang menguasai tata tulis Aksara Jawa.
   - **Dilarang keras** menyalin mentah keluaran model lalu dianggap sebagai ground truth tanpa koreksi manusia.
2. **Kotak Pembatas (*Bounding Box*)**:
   - Anotasi kotak 55 kelas atomik dilakukan di Label Studio menggunakan templat `eval/label_studio_template.xml`.
   - Pra-anotasi model boleh dipakai mempercepat peletakan awal, namun setiap kotak wajib diperiksa dan diperbaiki manusia (terutama komponen kecil *cecak*, *wulu*, *pepet*, *taling*, dan *pasangan*).
3. **Skor & Verifikasi**:
   - Cantumkan metadata `writer_id`, `session_id`, `paper_type`, `pen_type`, `lighting`, `device_id`.

---

## 5. Pembagian Per Penulis & Penguncian Hash

Pembagian **HARUS** dilakukan **per `writer_id` (dan tidak boleh memecah satu `session_id` antar-split)** — bukan acak per gambar — untuk mencegah kebocoran gaya tulisan tangan penulis atau tekstur meja yang sama ke set validasi/uji.

Rasio pembagian:
- **60%** $\rightarrow$ `real_train`
- **20%** $\rightarrow$ `real_val`
- **20%** $\rightarrow$ `real_test`

Jalankan skrip `eval/split_real.py`:
```bash
python eval/split_real.py --input-manifest path/to/raw_real_manifest.jsonl --out-dir eval/splits --seed 42
```
Skrip otomatis menghasilkan:
- `real_train_manifest.jsonl`, `real_val_manifest.jsonl`, `real_test_manifest.jsonl`
- `real_test.lock.sha256` (mengunci integritas berkas uji; verifikasi sebelum rilis Fase 5).

---

## 6. Etika & Privasi Kontributor

- Setiap penulis mengisi persetujuan singkat penggunaan gambar tulisan tangan untuk riset akademis/model open-source (`DATASET_CREDITS.md`).
- Identitas seluruh penulis dianonimkan menggunakan kode **`W01`, `W02`, dst.**
- Pastikan foto tidak memuat nama lengkap, NIM, wajah, tanda tangan asli, atau data pribadi kontributor.
