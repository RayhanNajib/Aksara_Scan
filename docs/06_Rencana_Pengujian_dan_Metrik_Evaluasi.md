# Rencana Pengujian dan Metrik Evaluasi
## AksaraScan: Pemindai Aksara Jawa Berbasis Web

| Parameter | Keterangan |
| :--- | :--- |
| **Mata Kuliah** | PTIN236021 Pengolahan Citra dan Komputer Vision (Semester V) |
| **Dosen Pengampu** | Mahfud Jiono, S.Pd., M.Eng. |
| **Kelompok / Tim** | **Kelompok 4** |
| **Penyusun** | 1. Muhammad Rayhan Najib (`240533600012`)<br>2. M. Ikhsan Ridho A. (`240533609185`) |
| **Institusi** | S1 Pendidikan Teknik Informatika, Departemen Teknik Elektro dan Informatika, Universitas Negeri Malang |
| **Versi Dokumen** | 1.0 (Draf Resmi) |
| **Tanggal** | 19 November 2026 |
| **Status Dokumen** | **Keluaran #6: Siap Tinjau** |

---

## 1. Landasan Metrik Evaluasi Ilmiah

Pengukuran efektivitas sistem AksaraScan dibagi menjadi tiga domain evaluasi: **Kinerja Pengenalan Pola (*Pattern Recognition*)**, **Efisiensi Komputasi (*Computational Latency*)**, dan **Kelayakan Pengalaman Pengguna (*Usability*)**.

```
                           DOMAIN EVALUASI SISTEM
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
[ KINERJA PENGENALAN ]     [ LATENSI KOMPUTASI ]        [ PENGALAMAN PENGGUNA ]
• Akurasi Karakter         • Latensi End-to-End         • System Usability Scale (SUS)
• Macro F1-Score           • Profil Waktu per Tahap     • Task Completion Rate (TCR)
• Character Error Rate     • Waktu Blokir UI (Jank)     • Zero Egress Network Audit
```

### 1.1 Metrik Kinerja Pengenalan Karakter

1. **Akurasi Klasifikasi Karakter (*Character Accuracy*)**:
   Diukur pada subset uji mandiri penulis (*Writer-Independent Test Set*):
   $$\text{Akurasi} = \frac{\sum_{c=0}^{19} TP_c}{N_{\text{total\_uji}}} \times 100\% \quad (\text{Target Awal: } \ge 85.0\%)$$
2. **Presisi, Recall, dan Macro F1-Score**:
   Menghitung keseimbangan performa model pada seluruh 20 kelas aksara Nglegena:
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$
   $$F1_c = 2 \times \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}, \quad \text{Macro } F1 = \frac{1}{20} \sum_{c=0}^{19} F1_c$$
3. **Character Error Rate (CER)**:
   Mengukur akurasi transliterasi rangkaian kalimat utuh pada baris tulisan tangan menggunakan algoritma jarak Levenshtein (*edit distance*):
   $$\text{CER} = \frac{S + D + I}{N} = \frac{\operatorname{Levenshtein}(T_{\text{ground\_truth}}, T_{\text{prediksi}})}{\operatorname{Length}(T_{\text{ground\_truth}})} \times 100\% \quad (\text{Target: } \le 15.0\%)$$
   *Keterangan: $S$ = Substitusi (salah tebak huruf), $D$ = Deplesi (karakter terlewat), $I$ = Insersi (derau dideteksi sebagai karakter), $N$ = Total karakter asli.*

---

### 1.2 Metrik Kinerja Komputasi & Efisiensi Sistem

1. **Latensi Pemrosesan Penuh (*End-to-End Latency*)**:
   Waktu total dari tombol "Proses Pindai" ditekan hingga bounding box dan transliterasi tampil di layar (diukur via `performance.now()`). Target:
   $$\text{Median } (p50) \le 3.0\text{ detik}, \quad \text{Percentile } (p95) \le 4.5\text{ detik}$$
2. **Ketiadaan Jeda Antarmuka (*Zero UI Jank*)**:
   Waktu eksekusi tugas terpanjang (*Long Task Duration*) pada thread utama UI wajib $< 50\text{ ms}$ untuk menjaga animasi rendering pada 60 FPS.
3. **Ukuran Bundel Jaringan (*Network Footprint*)**:
   - Total ukuran transfer aset web $\le 2.5\text{ MB}$.
   - Permintaan jaringan keluar selama inferensi: **Tepat 0 byte (*Zero Egress*)**.

---

### 1.3 Metrik Kegunaan (*Usability Metrics*)

1. **System Usability Scale (SUS)**: Kuesioner standar 10 pertanyaan skala Likert (1–5) dengan target skor rata-rata:
   $$\overline{\text{SUS}} \ge 68.0 \quad (\text{Kategori: Di atas rata-rata industri / Good})$$
2. **Tingkat Keberhasilan Tugas Mandiri (*Task Completion Rate*)**:
   Persentase responden yang berhasil menyelesaikan tugas memotret, memangkas, dan menyalin hasil tanpa intervensi penguji: Target $\ge 80.0\%$.

---

## 2. Piramida & Strategi Pengujian Sistem

Sistem diuji secara berlapis dari unit matematika dasar hingga interaksi perangkat keras nyata:

```
                      / \
                     /   \
                    / E2E \       Level 4: Uji Lapangan 30 Foto Nyata & 3 Perangkat
                   /-------\
                  / Integr. \     Level 3: Web Worker Transferable & PWA Offline
                 /-----------\
                / Golden Img. \   Level 2: Regresi 10 Citra Patokan (Golden Fixtures)
               /---------------\
              /  Unit Testing   \ Level 1: Uji Mandiri Fungsi Matematika Citra (Vitest)
             /-------------------\
```

### 2.1 Level 1: Pengujian Unit Modul Citra (Vitest)
Setiap modul di dalam direktori `src/pipeline/` memiliki berkas uji unit independen dengan masukan matriks sintetis terkontrol:

| Modul Uji | Skenario Masukan Uji | Kriteria Keberhasilan (*Assertion*) |
| :--- | :--- | :--- |
| `grayscale.test.ts` | Matriks RGB buatan $[(255,0,0), (0,255,0), (0,0,255)]$ | Nilai luminansi tepat sesuai koefisien $[76, 150, 29]$ toleransi $\pm 1$. |
| `otsu.test.ts` | Histogram bimodal sintetis (dua puncak di intensitas 50 dan 200) | Nilai ambang optimal $T^*$ jatuh di antara lembah ($T^* \approx 125$). |
| `median.test.ts` | Matriks $5 \times 5$ berisi derau bintik putih terisolasi $(x=2, y=2)$ | Piksel derau bernilai 1 tereliminasi menjadi 0 oleh jendela $3 \times 3$. |
| `morphology.test.ts` | Citra sintetis garis horizontal tipis $1 \times 30\text{ px}$ memotong garis vertikal | Garis horizontal terhapus tanpa memutus kontinuitas garis vertikal. |
| `projection.test.ts` | Citra biner 3 baris blok hitam terpisah spasi putih | Proyeksi horizontal mendeteksi tepat 3 interval baris teks valid. |
| `normalize.test.ts` | Glif persegi panjang berukuran $30 \times 70\text{ px}$ | Terpusat di kanvas $96 \times 96\text{ px}$ dengan margin aman $\ge 4\text{ px}$. |
| `svm.test.ts` | Vektor fitur tiruan yang cocok persis dengan hyperplane kelas Ha | Output probabilitas kelas Ha menghasilkan nilai tertinggi $> 0.80$. |

---

### 2.2 Level 2: Pengujian Citra Patokan (*Golden Fixtures Regression*)
Disiapkan **10 citra golden** yang disimpan permanen di repositori (`tests/fixtures/golden/`):
1. `golden_01_hvs_clear.png`: Tulisan sempurna di kertas polos.
2. `golden_02_lined_notebook.png`: Tulisan di buku bergaris SiDU dengan tinta gel hitam.
3. `golden_03_thin_ink.png`: Tulisan dengan pulpen ballpoint tipis $0.38\text{ mm}$.
4. `golden_04_skew_5deg.png`: Citra dengan kemiringan rotasi $5^\circ$.
5. `golden_05_gradient_shadow.png`: Citra dengan gradien bayangan ponsel di sudut kanan bawah.
6. `golden_06_ra_ka_pair.png`: Khusus pengujian pasangan mirip Ra dan Ka berdampingan.
7. `golden_07_pa_ya_pair.png`: Khusus pengujian pasangan mirip Pa dan Ya berdampingan.
8. `golden_08_multiline.png`: Halaman penuh berisi 4 baris teks berurutan.
9. `golden_09_dense_chars.png`: Jarak antarkarakter relatif rapat ($\approx 3\text{ px}$).
10. `golden_10_noisy_paper.png`: Kertas bernoda bintik debu halus.

*Pengujian otomatis memverifikasi bahwa pembaruan kode tidak menurunkan jumlah bounding box yang terdeteksi atau mengubah nilai hash citra antara pada Lab Pipeline.*

---

### 2.3 Level 3: Pengujian Integrasi & Jaringan
1. **Verifikasi Web Worker**: Memastikan objek `ImageBitmap` dikirim melalui mekanisme *Transferable Objects* tanpa duplikasi memori di heap utama.
2. **Audit Zero Egress (Playwright Network Interception)**:
   ```typescript
   // tests/network-privacy.spec.ts
   test('Tidak ada request keluar selama proses pindaian citra', async ({ page }) => {
     const externalRequests: string[] = [];
     page.on('request', req => {
       const url = new URL(req.url());
       if (url.origin !== 'http://localhost:5173') {
         externalRequests.push(req.url());
       }
     });
     // Jalankan pindaian foto penuh...
     expect(externalRequests).toHaveLength(0);
   });
   ```
3. **Pengujian PWA Mode Pesawat (*Offline Resilience*)**:
   - Buka aplikasi saat terkoneksi internet (Service Worker aktif).
   - Putuskan seluruh sambungan jaringan (*Offline / Airplane Mode*).
   - Muat ulang (*hard reload*) aplikasi: halaman harus terbuka penuh.
   - Eksekusi alur kamera dan pengenalan citra: seluruh pipeline harus sukses $100\%$.

---

### 2.4 Level 4: Pengujian End-to-End pada Perangkat Fisik Nyata

Pengujian komputasi dilakukan secara empiris pada 3 profil perangkat fisik yang merefleksikan pengguna sasaran:

| Kode Uji | Kategori Perangkat | Spesifikasi Perangkat Keras | Lingkungan Peramban | Peran Pengujian |
| :---: | :--- | :--- | :--- | :--- |
| **DEV-01** | Smartphone Mid-Range (Utama) | Helio G99 / Snapdragon 680, RAM 4 GB | Google Chrome Android v120+ | Patokan target latensi $\le 3.0\text{ detik}$ |
| **DEV-02** | Smartphone Entry-Level | Unisoc T606 / Helio G35, RAM 3 GB | Chrome / Samsung Internet | Uji batas bawah memori & ketahanan RAM |
| **DEV-03** | Laptop Mahasiswa / Dosen | Intel Core i5 / AMD Ryzen 5, RAM 8 GB | Chrome / Firefox Desktop | Uji Lab Pipeline di layar proyektor kelas |

---

## 3. Protokol Uji Kegunaan Pengguna (SUS & Skenario Lapangan)

Pengujian dilakukan terhadap **10 responden** yang merefleksikan 4 persona PRD:
- $5$ Siswa SMP/SMA (Persona Nadia)
- $2$ Guru Bahasa Jawa (Persona Bu Sri)
- $3$ Mahasiswa PTI / Rekan Sejawat (Persona Dimas)

### 3.1 Tiga Skenario Tugas Pengujian Terstruktur
1. **Skenario 1 (Pindaian Standar)**: Membuka aplikasi, memotret satu baris tulisan tangan di buku tulis menggunakan kamera ponsel, memangkas area teks, dan melihat hasil transliterasi.
2. **Skenario 2 (Koreksi Interaktif)**: Mengetuk satu kotak aksara yang salah tebak pada tampilan hasil, memilih aksara yang benar dari panel 20 Nglegena, dan menyalin teks Latin yang telah diperbarui.
3. **Skenario 3 (Eksplorasi Lab Pipeline)**: Membuka tab "Lab Pipeline", mengganti ukuran kernel filter median, dan mengamati perubahan visual pada histogram intensitas citra.

### 3.2 Kuesioner System Usability Scale (SUS)
Setiap responden mengisi 10 instrumen baku SUS dengan skala 1 (*Sangat Tidak Setuju*) sampai 5 (*Sangat Setuju*):
1. Saya berpikir akan sering menggunakan aplikasi AksaraScan ini.
2. Saya merasa aplikasi ini terlalu rumit, padahal dapat dibuat lebih sederhana.
3. Saya merasa aplikasi ini mudah digunakan.
4. Saya rasa saya membutuhkan bantuan orang teknis untuk dapat menggunakan aplikasi ini.
5. Saya menemukan fungsi-fungsi dalam aplikasi ini terintegrasi dengan sangat baik.
6. Saya merasa ada terlalu banyak hal yang tidak konsisten pada aplikasi ini.
7. Saya yakin orang lain akan belajar menggunakan aplikasi ini dengan sangat cepat.
8. Saya merasa aplikasi ini sangat membingungkan saat digunakan.
9. Saya merasa sangat percaya diri menggunakan aplikasi ini.
10. Saya harus belajar banyak hal terlebih dahulu sebelum menguasai aplikasi ini.

---

## 4. Format Template Pelaporan Hasil Pengujian (Untuk Bab IV Laporan TA)

Tabel-tabel berikut disiapkan dengan tanda `[ISI SETELAH EVALUASI]` dan akan diisi data numerik riil saat pengujian lapangan selesai dijalankan:

### 4.1 Tabel Kinerja Latensi Komputasi per Tahap Pipeline
*(Diuji pada citra resolusi kerja $1600 \times 1200\text{ px}$)*

| No | Tahapan Pipeline Citra | DEV-01 (Ponsel Mid) | DEV-02 (Ponsel Entry) | DEV-03 (Laptop Core i5) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | Pemuatan Citra & Resampling | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 2 | Konversi Grayscale ITU-R BT.601 | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 3 | Binerisasi Ambang Batas Otsu | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 4 | Filter Spasial Median $3 \times 3$ | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 5 | Morfologi (Garis Buku, Closing) | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 6 | Segmentasi Baris & Karakter | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 7 | Normalisasi Glif $96 \times 96$ | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| 8 | Inferensi SVM Linear (20 Glif) | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` | `[ISI SETELAH EVALUASI] ms` |
| **-** | **TOTAL WAKTU END-TO-END** | **`[ISI] s` (Target $\le 3\text{s}$)** | **`[ISI] s`** | **`[ISI] s`** |

---

### 4.2 Tabel Hasil Pengujian End-to-End pada 30 Foto Nyata

| Kategori Kondisi Foto | Jumlah Dokumen | Rata-rata CER (%) | Akurasi Glif (%) | Keterangan Evaluasi |
| :--- | :---: | :---: | :---: | :--- |
| **Kertas HVS Polos (Ideal)** | 10 foto | `[ISI] %` | `[ISI] %` | Kondisi pencahayaan merata |
| **Buku Bergaris SiDU (Standar)** | 10 foto | `[ISI] %` | `[ISI] %` | Efektivitas eliminasi garis buku |
| **Kondisi Menantang (Miring/Bayang)** | 10 foto | `[ISI] %` | `[ISI] %` | Kemiringan $\approx 5\text{–}10^\circ$, bayangan HP |
| **RATA-RATA KESELURUHAN** | **30 foto** | **`[ISI] %` (Target $\le 15\%$)** | **`[ISI] %` (Target $\ge 85\%$)** | **Evaluasi Ujung-ke-Ujung Bab IV** |

---

### 4.3 Tabel Hasil Skor Usability (SUS)

| No Responden | Peran / Persona | Skor SUS Individual | Tingkat Keberhasilan Tugas |
| :---: | :--- | :---: | :---: |
| R01 | Siswa SMP 1 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R02 | Siswa SMP 2 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R03 | Siswa SMA 1 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R04 | Siswa SMA 2 | `[ISI SETELAH EVALUASI]` | Butuh 1 Bantuan (80%) |
| R05 | Siswa SMA 3 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R06 | Guru Bahasa Jawa 1 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R07 | Guru Bahasa Jawa 2 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R08 | Mahasiswa PTI 1 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R09 | Mahasiswa PTI 2 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| R10 | Mahasiswa PTI 3 | `[ISI SETELAH EVALUASI]` | Berhasil Mandiri (100%) |
| **RATA-RATA** | **10 Responden** | **`[ISI]` (Target $\ge 68.0$)** | **`[ISI] %` (Target $\ge 80\%$)** |

---

> [!NOTE]
> **Status Kelulusan Tahap 6**: Rencana Pengujian dan Metrik Evaluasi telah selesai dirinci. Siap dilanjutkan ke **Keluaran #7: Outline Laporan Tugas Akhir (Bab I–V)** setelah konfirmasi dari Anda.
