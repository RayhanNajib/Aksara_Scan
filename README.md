# AksaraScan
Pemindai & Penerjemah Aksara Jawa (v3.0 Offline Browser Engine)

**AksaraScan** adalah aplikasi web progresif untuk mendeteksi, memetakan ortografi **55 kelas** Aksara Jawa, mentransliterasi secara kanonik, dan menerjemahkannya ke Bahasa Indonesia secara langsung di peramban tanpa mengunggah citra pengguna ke peladen mana pun.

## Arsitektur Utama v3.0
1. **Detektor Objek Atomik 55 Kelas (`app/web_camscanner/aksara_yolo.onnx`, `12.11 MB`)**:
   - **20 Nglegena**, **20 Pasangan** (`p_ha`–`p_nga`), **12 Sandhangan & Pangkon**, serta **3 Tanda Baca Pada** (`pada_adeg`, `pada_lingsa`, `pada_lungsi`).
   - Tersedia arsitektur pelatihan bersih **Apache-2.0 / MIT** di `training/yolox_model.py` dan `training/train_yolox.py` (**YOLOX** + opsional *P2* stride-4 head untuk glif kecil `< 16 px`, bebas `ultralytics` AGPL-3.0) serta jalur penantang **Track B CRNN + BiLSTM + CTC Loss** (`training/train_crnn_track_b.py` -> `aksara_crnn_ctc.onnx`).
2. **Rekonstruktor Spasial & Validator Ortografi (`app/web_camscanner/orthography.js`)**:
   - Perakitan gugus pasangan bawah & kanan (`p_ha`, `p_sa`, `p_pa`), pengaitan sandhangan relatif terhadap `fullAnchor`, dan validasi kanonik blok Unicode `U+A980..U+A9DF`.
3. **Dekoder Beam Konservatif & Mesin Bahasa Jawa (`app/web_camscanner/javanese_lm_data.js`, `app/web_camscanner/javanese_lm_engine.js`)**:
   - Menggabungkan kamus kurasi + `AKSARA_LM_DATA` (`10.589` unigram, `4.500` entri kamus, `560` konteks char-trigram dari `866.150` token korpus Jawa) serta morfologi afiks produktif untuk memperbaiki salah-baca diakritik kecil pada kata OOV tanpa menghalusinasi kata yang sudah sah.
4. **Antarmuka Transparan & Umpan Balik Lokal (`app/web_camscanner/v3_ui_enhancer.js`)**:
   - Menampilkan transkripsi Unicode mentah detektor di samping hasil perbaikan, menandai kata belum dikenal dengan garis bawah putus-putus untuk tukar kandidat *top-K*, ekspor koreksi lokal JSON, serta *toggle* terjemahan daring yang **mati secara baku**.

## Struktur Dokumentasi, Pelatihan, & Evaluasi
- [Audit & Roadmap (`docs/AUDIT.md`)](docs/AUDIT.md)
- [Katalog Referensi & Lisensi (`docs/REFERENCES.md`)](docs/REFERENCES.md)
- [Kartu Model (`MODEL_CARD.md`)](MODEL_CARD.md) & [Kredit Dataset (`DATASET_CREDITS.md`)](DATASET_CREDITS.md)
- [Laporan Data Sintetis v2 (`dataset/external/reports/PHASE2_SYNTH_REPORT.md`)](dataset/external/reports/PHASE2_SYNTH_REPORT.md) (`14.998` sampel SIL OFL 1.1, `727.495` kotak tervalidasi)
- [Runbook Pelatihan GPU Penuh (`training/COLAB_GPU_RUNBOOK.md`)](training/COLAB_GPU_RUNBOOK.md)
- [Laporan Evaluasi Baseline & v3.0 (`eval/reports/v3.0.md`)](eval/reports/v3.0.md)
- Uji Otomatis Browser: `node eval/test_e2e_assertions.mjs` & `python eval/run_eval.py --parity eval/parity_20.jsonl --failures eval/failures/manifest.jsonl`

## Menjalankan Secara Lokal
Buka folder `app/web_camscanner/` melalui *local web server* (live preview, `python -m http.server`, atau `npx serve`):

```bash
cd app/web_camscanner
npx serve .
```
