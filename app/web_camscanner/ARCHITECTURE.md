# AksaraScan Web — Arsitektur & MVP

## Scope MVP (v1)
**Masuk:**
- Input citra: unggah berkas + kamera ponsel (capture environment)
- Praproses citra: grayscale → binerisasi Otsu → segmentasi baris & aksara (proyeksi)
- Normalisasi glif ke kanvas 96×96 (sama dengan pelatihan model)
- Klasifikasi SVM linear, 20 kelas Nglegena (bobot dari model.json)
- Tampilkan hasil teks + keyakinan per aksara + citra biner
- Respon: desktop + layar ponsel
- Semua pemrosesan di browser, tanpa server, tanpa API key

**Keluar (v2+):**
- Deteksi tepi & perbaikan perspektif (warp CamScanner-style)
- Web Worker / off-main-thread
- Filter spasial lanjutan di luar Otsu
- Autentikasi, sinkronisasi cloud, akun pengguna

## Arsitektur (client-only SPA)
```
Browser
├── index.html   — susunan UI (responsif)
├── styles.css   — token warna, tata letak, breakpoint
└── app.js       — orok-orko pipeline + pemancah UI

Pipeline (semua di main thread):
Ambil citra → batasi ukuran 1500 px → grayscale (0.299/0.587/0.114)
→ Otsu ambang → biner → proyeksi horizontal (baris)
→ proyeksi vertikal (aksara per baris) → normCrop 96×96
→ SVM linear (vektor 9.216, bobot float16) → hasil + keyakinan
```

## Aturan implementasi
- Tanpa build step, tanpa dependency baru (stdlib browser saja)
- Satu halaman, vanilla JS
- Fungsi ≤ 50 baris, setiap baris menjawab kebutuhan pengguna
- Tidak ada `any`, setiap async di-wrap try-catch, input divalidasi
- Upload maksimal 15 MB; model dimuat sekali saat init
