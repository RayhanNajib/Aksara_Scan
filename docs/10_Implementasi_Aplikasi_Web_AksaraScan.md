# Panduan Implementasi dan Pengujian Aplikasi Web AksaraScan
## Pemindai Aksara Jawa Berbasis Web (Client-Only Single Page Application)

| Parameter | Keterangan |
| :--- | :--- |
| **Mata Kuliah** | PTIN236021 Pengolahan Citra dan Komputer Vision (Semester V) |
| **Dosen Pengampu** | Mahfud Jiono, S.Pd., M.Eng. |
| **Kelompok / Tim** | **Kelompok 4** |
| **Penyusun** | 1. Muhammad Rayhan Najib (`240533600012`)<br>2. M. Ikhsan Ridho A. (`240533609185`) |
| **Institusi** | S1 Pendidikan Teknik Informatika, Departemen Teknik Elektro dan Informatika, Universitas Negeri Malang |
| **Direktori Aplikasi** | `C:\Users\WA\Downloads\TUGAS PKKMB\Citra Komputer\Projek_Akhir\web_camscanner` |
| **Versi Aplikasi** | 1.0.1 (Hotfix Null Style & Touch Interaction) |
| **Tanggal Pembaruan** | 20 November 2026 |

---

## 1. Ringkasan Implementasi Sistem

Aplikasi web **AksaraScan** dibangun secara mandiri di sisi klien (*client-only*) pada direktori `Projek_Akhir\web_camscanner\`:

```
web_camscanner/
├── index.html            # Antarmuka SPA lengkap bertema "Manuskrip Modern"
├── styles.css            # Sistem token warna (Prada, Wulung, Sogan, Dluwang) & Anti-Slop UI
├── app.js                # Engine pengolahan citra 8 tahap murni + SVM Float16 Decoder
├── model.js              # Model SVM terintegrasi bebas blokir CORS (file:// friendly)
├── model.json            # Model SVM Linear 20 kelas Nglegena (184.320 bobot float16)
├── manifest.webmanifest  # Metadata PWA (Add to Home Screen & Layar Penuh)
├── sw.js                 # Service Worker Cache (Dukungan Offline 100%)
├── sample_1.png          # Sampel Uji 1: Naskah 1 Baris (1600×1028 px)
├── sample_2.png          # Sampel Uji 2: Kisi 20 Aksara Buku Tulis (1400×900 px)
└── sample_3.png          # Sampel Uji 3: Tulisan Tangan Pena Gel (1073×1102 px)
```

---

## 2. Catatan Pembaruan & Perbaikan Bug (v1.0.1 Hotfix)

1. **Resolusi Error `Cannot read properties of null (reading 'style')`**:
   - Di versi sebelumnya, kontainer meter keyakinan di `index.html` belum memiliki elemen anak ber-ID `meterFill`, sementara fungsi `renderOutputs()` dan `resetResults()` di `app.js` mencoba memperbarui `$('#meterFill').style.width`.
   - **Solusi**: Elemen `<div class="confidence-meter__fill" id="meterFill"></div>` dan styling bar telah ditambahkan, serta diberikan perlindungan *defensive null-check* di seluruh kode JavaScript.
2. **Audit 40 Elemen DOM**:
   - Seluruh 40 pemanggilan selector ID (`$('#...')`) di `app.js` telah diaudit otomatis dan terbukti cocok $100\%$ dengan elemen di `index.html`.
3. **Penyertaan `model.js` Mandiri**:
   - Menghindari pemblokiran CORS saat file `index.html` diklik ganda langsung di Windows Explorer (`file:///`).

---

## 3. Cara Menjalankan Aplikasi Web

### Opsi A: Klik Ganda Langsung
Cukup buka berkas `index.html` langsung dari Windows Explorer.

### Opsi B: Menggunakan Server Lokal Python
```powershell
cd "C:\Users\WA\Downloads\TUGAS PKKMB\Citra Komputer\Projek_Akhir\web_camscanner"
python -m http.server 8080
```
Buka peramban: `http://localhost:8080`
