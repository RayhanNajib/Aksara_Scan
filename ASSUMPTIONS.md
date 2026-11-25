# Asumsi dan Keputusan Sementara — AksaraScan (Fase 0)

Dokumen ini mencatat asumsi sederhana yang diambil agar audit dan pembangunan *harness* evaluasi Fase 0 berjalan tanpa memblokir hal yang masih memerlukan konfirmasi pengguna.

1. **Lingkungan Perangkat Keras & Pelatihan Lokal**:
   - Audit saat ini mendeteksi tidak ada GPU CUDA (`torch 2.13.0+cpu`, `CUDA available: False`). Semua pengukuran inferensi baseline di Fase 0 dijalankan di CPU / WASM. Pelatihan tahap selanjutnya (`imgsz 960/1280` atau YOLOv8s / RT-DETR) kemungkinan membutuhkan GPU (Google Colab, Kaggle Notebooks, atau GPU lokal pengguna).
2. **Lokasi Data Mentah `Jawa.zip` dan `dataset_yolo`**:
   - Di repo git (`Aksara_Scan`), hanya `dataset/indonesian-scripts/DOI.txt` yang tercatat (mengingat batas 100 MB GitHub), sedangkan arsip fisik `Jawa.zip` (~135.3 MB) dan hasil render v2.5 `dataset_yolo/` (1.400 train + 240 val) berada di direktori saudara `../Projek_Akhir/` pada mesin lokal pengguna. Harness Fase 0 membaca jalur tersebut secara opsional via argumen `--yolo-dir` tanpa mengubah atau menyalin aset raksasa ke dalam git.
3. **Harness Evaluasi JavaScript (`eval/run_eval.mjs`)**:
   - Untuk menjamin 100% paritas dengan peramban pengguna, `eval/run_eval.mjs` memakai `onnxruntime-web@1.20.1` dan `playwright-core` (memanfaatkan biner Chromium lokal yang sudah terpasang) untuk mengeksekusi langsung `app/web_camscanner/app.js`, `orthography.js`, dan `javanese_lexicon.js` dalam mode *headless* tanpa permintaan jaringan eksternal, sekaligus mendukung mode Node langsung (`onnxruntime-web` WASM + `sharp`) untuk menguji paritas tensor mentah terhadap `onnxruntime` Python.
4. **Set Diagnostik Awal (`eval/failures/`)**:
   - Karena set foto tulisan tangan nyata (`eval/real_val` / `eval/real_test`) belum terkumpul, Fase 0 memakai 8 berkas citra uji yang sudah tersedia secara lokal (gambar kalimat pengguna *Wédang bubuk...*, `sample_4.png`, `test_shadow.png`, `test_walllines.png`, `test_small.png`, `sample_1_paper.png`, `sample_2.png`, `sample_3.png`) ke dalam `eval/failures/` sebagai **set diagnostik kecil awal** dan secara eksplisit **tidak** diklaim sebagai bukti performa dunia nyata (*real_test*).
5. **Konvensi Kecil & Pelabelan Diagnostik**:
   - Ukuran "objek kecil (< 16 px)" pada evaluasi kotak mengacu pada `max(w, h) < 16 px` pada resolusi input inferensi `416×416` (dan turut dilaporkan pada koordinat asli gambar sebelum *resize* agar transparan).
   - Keputusan besar terkait **konflik lisensi MIT vs Ultralytics AGPL-3.0** tidak diputuskan sepihak di Fase 0 dan ditandai eksplisit untuk disetujui pengguna sebelum Fase 1–3.
