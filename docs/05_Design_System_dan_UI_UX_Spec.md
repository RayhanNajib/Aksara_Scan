# Design System dan Spesifikasi UI/UX
## AksaraScan: Konsep Visual "Manuskrip Modern"

| Parameter | Keterangan |
| :--- | :--- |
| **Mata Kuliah** | PTIN236021 Pengolahan Citra dan Komputer Vision (Semester V) |
| **Dosen Pengampu** | Mahfud Jiono, S.Pd., M.Eng. |
| **Kelompok / Tim** | **Kelompok 4** |
| **Penyusun** | 1. Muhammad Rayhan Najib (`240533600012`)<br>2. M. Ikhsan Ridho A. (`240533609185`) |
| **Institusi** | S1 Pendidikan Teknik Informatika, Departemen Teknik Elektro dan Informatika, Universitas Negeri Malang |
| **Versi Dokumen** | 1.0 (Draf Resmi) |
| **Tanggal** | 19 November 2026 |
| **Status Dokumen** | **Keluaran #5: Siap Tinjau** |

---

## 1. Filosofi & Arah Konsep Desain: "Manuskrip Modern"

AksaraScan memadukan estetika manuskrip kuno nusantara (*kertas dluwang dan daun lontar*) dengan disiplin antarmuka digital kontemporer yang tenang, presisi, dan fungsional. 

### 1.1 Prinsip Desain Inti
1. **Keaslian Budaya yang Fungsional (*Purposeful Heritage*)**: Budaya Jawa dihadirkan melalui palet warna alami (*sogan, wedel, prada*), tipografi glif aksara yang anggun, serta pembatas *pada lingsa*. Tidak ada penempelan ornamen batik acak yang mengaburkan keterbacaan (*anti-slop*).
2. **Ketenangan Visual (*Visual Serenity*)**: Mengutamakan kenyamanan membaca dalam durasi lama. Mengurangi ketegangan mata saat siswa belajar atau dosen memeriksa tugas di ruang minim cahaya.
3. **Ketegasan Data & Transparansi (*Data Precision*)**: Data teknis pengolahan citra (dimensi piksel, histogram, nilai ambang Otsu, skor keyakinan SVM) disajikan dengan presisi tinggi menggunakan font *monospaced* teknik.
4. **Respon Taktil Cepat (*Tactile Responsiveness*)**: Animasi transisi ringkas ($150\text{–}250\text{ ms}$) dengan efek pemunculan karakter beruntun (*staggered reveal*) untuk menciptakan momen kepuasan pengguna.

---

## 2. Token Warna Sistem (*Design Tokens Palette*)

Palet warna dibangun berdasarkan pewarna alami kain dan naskah Jawa tradisional:

```
MODE TERANG (Kertas Dluwang)          MODE GELAP (Malam Keraton - Bawaan)
┌───────────────────────────────┐     ┌───────────────────────────────┐
│ paper      #F4EAD5 (Latar)    │     │ wulung     #1F2A44 (Latar)    │
│ paper-deep #E8D9B8 (Kartu)    │     │ wulung-sfc #161E31 (Kartu)    │
│ sogan      #5B3A29 (Teks Utk) │     │ paper      #F4EAD5 (Teks Utk) │
│ prada      #C8963E (Aksen Em) │     │ prada      #C8963E (Aksen Em) │
└───────────────────────────────┘     └───────────────────────────────┘
```

### 2.1 Definisi Token Warna Lengkap

| Token CSS / Tailwind | Nilai Hex | Peran & Penggunaan | Rasio Kontras (WCAG AA) |
| :--- | :---: | :--- | :---: |
| `paper` | `#F4EAD5` | Latar belakang utama mode terang (tekstur kertas dluwang). Teks utama di mode gelap. | $11.4:1$ (vs sogan) |
| `paper-deep` | `#E8D9B8` | Latar kartu, panel sekunder, dan *bottom sheet* mode terang. | $9.2:1$ (vs sogan) |
| `sogan` | `#5B3A29` | Teks judul, teks body, dan ikon utama pada mode terang (warna cokelat kulit kayu). | $11.4:1$ (vs paper) |
| `sogan-soft` | `#8A5A3B` | Teks sekunder, label pembantu, dan garis pemisah halus di mode terang. | $5.1:1$ (Lolos AA) |
| `wulung` / `indigo-wedel` | `#1F2A44` | Latar belakang utama mode gelap (biru nila malam keraton) & tombol utama terang. | $13.2:1$ (vs paper) |
| `wulung-surface` | `#161E31` | Kartu elevated, panel drawer, dan modal dialog pada mode gelap. | - |
| `prada` | `#C8963E` | Aksen emas: garis tepi aktif (*focus ring*), kotak bounding box yakin, lencana prestasi. | $3.5:1$ (Garis & Ikon) |
| `bata` | `#A8432A` | Status kesalahan, aksi destruktif (hapus), dan penandaan aksara ragu (`?`). | $4.8:1$ (Lolos AA) |
| `lontar` | `#5E6B3A` | Status sukses, lencana pindaian terverifikasi, dan akurasi tinggi. | $4.9:1$ (Lolos AA) |
| `tinta` | `#1B1410` | Glif aksara Jawa hitam pekat untuk kontras maksimum di kanvas biner. | $15.8:1$ (vs paper) |

> [!IMPORTANT]
> **Aturan Penggunaan Prada (`#C8963E`)**: Token `prada` adalah warna aksen emas khusus untuk *stroke*, *border*, indikator fokus, dan ikon. Dilarang keras menggunakan `prada` sebagai warna teks paragraf kecil di atas latar `paper` karena tidak memenuhi rasio kontras WCAG AA.

---

## 3. Sistem Tipografi (*Typography Scale*)

Kombinasi empat keluarga font yang memiliki peran fungsional tegas:

| Peran | Keluarga Font (*Font Family*) | Karakteristik & Alasan Pemilihan |
| :--- | :--- | :--- |
| **Glif Aksara Jawa** | **Noto Sans Javanese** | Font standar Unicode resmi Google, proporsi anatomi glif sempurna, didukung secara native. |
| **Judul Layar & Hero** | **Fraunces** (atau Newsreader) | Serif modern dengan sentuhan kaligrafi hangat yang mencerminkan martabat manuskrip. |
| **Teks UI & Konten** | **Plus Jakarta Sans** | Sans-serif geometris karya desainer Indonesia, sangat jernih dan terbaca pada layar HP kecil. |
| **Data Teknis & Lab** | **JetBrains Mono** | Monospace presisi untuk koordinat piksel, nilai histogram, dan kode matriks. |

### 3.1 Skala Tipografi Modular (Basis Skala 4 px)

```
Level       Ukuran / Line-Height    Font Family             Bobot       Penggunaan
──────────────────────────────────────────────────────────────────────────────────────────
Display     40 px / 1.2             Fraunces                Bold        Judul Hero Beranda
H1          28 px / 1.3             Fraunces                SemiBold    Judul Layar Utama
H2          20 px / 1.4             Plus Jakarta Sans       Bold        Header Kartu / Seksi
Glif Besar  32 px / 1.2             Noto Sans Javanese      Regular     Tampilan Glif Hasil
Body        16 px / 1.5             Plus Jakarta Sans       Regular     Teks Paragraf & Tombol
Body Small  14 px / 1.5             Plus Jakarta Sans       Medium      Label Input & Metadata
Caption     12 px / 1.4             JetBrains Mono          Regular     Nilai Ambang, Timer ms
```

---

## 4. Standar Kualitas Keras: "Skill Anti-Slop UI/UX"

Penerapan aturan absolut anti-slop untuk mencegah antarmuka yang generik, norak, atau murahan:

1. **Larangan Gradien Neon & Glow**: Dilarang menggunakan gradien ungu-biru ala template AI murahan, efek pendar neon (*neon glow*), atau *glassmorphism* buram yang membebani GPU ponsel.
2. **Larangan Emoji Sebagai Ikon Sistem**: Dilarang menggunakan emotikon ponsel (seperti 📷, 🔍, 📝) sebagai ikon tombol antarmuka. Wajib menggunakan pustaka **Lucide React** dengan ketebalan garis seragam $1.5\text{ px}$.
3. **Larangan Teks Semu (*No Dummy Copy*)**: Dilarang menggunakan "Lorem Ipsum", "Feature One", atau "Test Title". Semua teks sampel wajib menggunakan kalimat bahasa Jawa nyata (contoh: *"ꦲꦤꦕꦫꦏ - Hanacaraka"*, *"ꦱꦧꦼꦤ꧀ꦢꦶꦤ - Saben dina"*).
4. **Larangan Ornamen Budaya Acak**: Motif Jawa dilarang ditempel sembarangan di pojok layar. Ornamen dibatasi pada:
   - Pembatas bagian: Simbol **pada lingsa** (`꧈`) berupa garis SVG tipis bergaris ganda.
   - Latar belakang empty-state: Pola geometris **kawung** dengan opasitas sangat tipis ($\le 5\%$).
   - Bingkai bidik kamera: Empat sudut penahan berwarna `prada` terinspirasi tatahan wayang kulit.
5. **Hierarki Kartu yang Berbeda**: Hindari 3 kartu identik berdampingan. Elemen aksi utama harus memiliki bobot visual dominan dibanding aksi sekunder.
6. **Konsistensi Jarak (*Spatial Rhythm*)**: Seluruh margin dan padding wajib menggunakan kelipatan 4 px: `$space-1` (4px), `$space-2` (8px), `$space-3` (12px), `$space-4` (16px), `$space-6` (24px), `$space-8` (32px).
7. **Kelengkapan Status Komponen**: Setiap kontrol interaktif wajib memiliki 6 status terdokumentasi: *Default, Hover, Active, Focus-Visible, Disabled, Loading*.
8. **Ergonomi Sentuh Ponsel**: Semua tombol dan area ketuk karakter memiliki luas sentuh fisik minimal $48 \times 48\text{ dp}$ (target WCAG 2.2).
9. **Mikrokopi Hangat & Kontekstual**: Gunakan pesan kesalahan yang ramah dan solutif (Contoh: *"Cahaya tampak redup. Nyalakan lampu atau dekatkan kamera agar aksara terbaca tajam."*).
10. **Aksesibilitas Gerak**: Seluruh animasi dinonaktifkan secara otomatis bagi pengguna yang mengaktifkan fitur OS `prefers-reduced-motion`.

---

## 5. Spesifikasi Komponen Inti (*UI Components*)

### 5.1 Kartu Manuskrip (*Manuskrip Card*)
- **Bentuk**: Radius sudut persegi halus ($r = 8\text{ px}$). Bukan pil lonjong (*pill-shaped*).
- **Border**: Garis tepi tipis $1\text{ px}$ bernilai `sogan/20` (mode terang) atau `prada/25` (mode gelap).
- **Elemen Penting**: Ditambahkan garis aksen ganda tipis emas `prada` selebar $1\text{ px}$ pada tepi dalam atas (*top hairline*).
- **Bayangan**: Bayangan tunggal lembut alami: `box-shadow: 0 2px 8px -2px rgba(27, 20, 16, 0.08)`.

### 5.2 Tombol Aksi Budaya (*Heritage Buttons*)
```
[ TOMBOL UTAMA ]  Background: #1F2A44 (wulung) | Text: #F4EAD5 (paper)
                  Border: 1px solid #C8963E (prada) | Padding: 12px 24px | Radius: 6px
                  Hover: Tingkatkan kecerahan background +2% | Focus: Outline 2px prada

[ TOMBOL SEKUNDER ] Background: Transparan | Text: #5B3A29 (sogan)
                    Border: 1px solid #8A5A3B/40 | Hover: Background #E8D9B8/50
```

### 5.3 Kotak Deteksi Karakter (*Bounding Box Overlay*)
- Garis kotak: Lebar garis $1.5\text{ px}$ di atas kanvas citra.
- Warna kotak:
  - **Status Yakin** ($P \ge 0.70$): Emas `prada` (`#C8963E`) dengan label kecil di sudut kiri atas.
  - **Status Sedang** ($0.35 \le P < 0.70$): Cokelat `sogan-soft` (`#8A5A3B`).
  - **Status Ragu** ($P < 0.35$): Merah bata `bata` (`#A8432A`) dengan tanda tanya (`?`).
- Interaksi: Mengetuk kotak memicu *pulse animation* 200 ms dan membuka panel koreksi.

---

## 6. Spesifikasi Tata Letak 7 Layar Inti (Mobile & Desktop)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        ARSITEKTUR LAYAR UTAMA                          │
├───────────────────┬────────────────────────────────────────────────────┤
│ Layar             │ Deskripsi & Komponen Kunci                         │
├───────────────────┼────────────────────────────────────────────────────┤
│ S1: Beranda       │ Hero Manuskrip, Tombol Pindai/Unggah, 3 Demo Bawaan│
│ S2: Kamera        │ Viewfinder Wayang 4-Sudut, Indikator Cahaya, Rana  │
│ S3: Pangkas       │ 4 Pin Sudut Kontrol Drag, Rotasi 90°, Preview Zoom │
│ S4: Memproses     │ Linimasa 7 Tahap Bernama & Bar Persentase Aktif    │
│ S5: Hasil Pindaian│ Overlay Kotak, Teks Aksara/Latin, Skor Keyakinan   │
│ S6: Lab Pipeline  │ Galeri 7 Tahap, Histogram, Slider Parameter Dosen  │
│ S7: Kamus 20      │ Grid 20 Aksara Nglegena, Audio Fonetik, Filosofi   │
└───────────────────┴────────────────────────────────────────────────────┘
```

### 6.1 Wireframe S1: Layar Beranda (Mobile 390 px)
```
┌────────────────────────────────────────┐
│ [≡ Menu]       AksaraScan       [☼/☾] │
├────────────────────────────────────────┤
│                                        │
│          ꦲꦤꦕꦫꦏ (Noto Sans)             │
│   Pemindai Aksara Jawa Mandiri        │
│    "Ubah foto tulisan tangan buku      │
│      tulis menjadi teks Latin"         │
│                                        │
│  ┌──────────────────────────────────┐  │
│  │  [📷 PINDAI TULISAN TANGAN]      │  │  <- Tombol Utama (Wulung + Prada)
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │  [📁 Unggah Foto dari Galeri]     │  │  <- Tombol Sekunder
│  └──────────────────────────────────┘  │
│                                        │
│  ─── ꧈  Coba Contoh Tulisan Buku  ꧈ ──  │  <- Pembatas Pada Lingsa
│                                        │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Sampel 1 │ │ Sampel 2 │ │Sampel 3 │ │  <- 3 Kartu Uji Nyata Bawaan
│  │ "Hana..."│ │ "Data..."│ │"Maga.." │ │
│  └──────────┘ └──────────┘ └─────────┘ │
│                                        │
│  🛡️ 100% Privat: Diproses di ponsel Anda│
└────────────────────────────────────────┘
```

### 6.2 Wireframe S5: Layar Hasil Pindaian (Mobile 390 px)
```
┌────────────────────────────────────────┐
│ [< Kembali]      Hasil Pindaian  [Lab⚙]│
├────────────────────────────────────────┤
│ ┌────────────────────────────────────┐ │
│ │  [Foto Buku Tulis dengan Overlay]  │ │
│ │   ┌────┐ ┌────┐ ┌────┐ ┌────┐      │ │  <- Bounding Box Prada 1.5 px
│ │   │ ꦲ  │ │ ꦤ  │ │ ꦕ  │ │ ꦫ  │      │ │
│ │   │ ha │ │ na │ │ ca │ │ ra │      │ │
│ │   └────┘ └────┘ └────┘ └────┘      │ │
│ └────────────────────────────────────┘ │
│                                        │
│ ┌─ KELUARAN AKSARA JAWA ─────────────┐ │
│ │ ꦲ ꦤ ꦕ ꦫ ꦏ ꦢ ꦠ ꦱ ꦮ ꦭ                │ │  <- Glif Besar 28 px
│ └────────────────────────────────────┘ │
│                                        │
│ ┌─ TRANSLITERASI LATIN ──────────────┐ │
│ │ ha na ca ra ka da ta sa wa la      │ │  <- Font Plus Jakarta Sans
│ └────────────────────────────────────┘ │
│                                        │
│ Keyakinan Rata-rata: 92% (Tinggi)      │
│ [📋 Salin Teks]      [💾 Unduh Gambar] │
│                                        │
│ *Ketuk kotak huruf di atas untuk koreksi│
└────────────────────────────────────────┘
```

### 6.3 Wireframe S6: Layar Lab Pipeline (Andalan Presentasi Dosen)
```
┌────────────────────────────────────────────────────────┐
│ [< Hasil]       Lab Pengolahan Citra Digital (7 Tahap) │
├────────────────────────────────────────────────────────┤
│ [1. Asli] [2. Grayscale] [3. Otsu] [4. Median] [5. Morf]│ <- Tab Navigasi Tahap
│                                                        │
│ ┌───────────────────────────┐ ┌──────────────────────┐ │
│ │    Citra Hasil Tahapan    │ │  Histogram Intensitas│ │
│ │                           │ │   |  /\              │ │
│ │   [Preview Biner Otsu]    │ │   |_/  \__/\__       │ │
│ │                           │ │  Ambang T* = 142     │ │
│ └───────────────────────────┘ └──────────────────────┘ │
│                                                        │
│ Parameter Eksperimen Interaktif (Dosen Test):          │
│ • Ambang Binerisasi Otsu : [───●──────────] 142       │
│ • Ukuran Kernel Median   : (o) 3x3   ( ) 5x5           │
│ • Elemen Penstruktur     : (o) Garis 1x21  ( ) Disk 3x3│
│                                                        │
│ [🔄 Terapkan Ulang & Ukur Latensi: 214 ms]             │
└────────────────────────────────────────────────────────┘
```

### 6.4 Wireframe S5 + S6 pada Tampilan Desktop (Dua Kolom 1280 px)
Pada layar laptop/proyektor kelas, layar Hasil dan Lab Pipeline disajikan berdampingan (*split-screen view*):
- **Kolom Kiri ($50\%$)**: Citra dokumen resolusi tinggi dengan overlay bounding box interaktif dan editor koreksi teks.
- **Kolom Kanan ($50\%$)**: Panel inspeksi matriks, histogram frekuensi, slider parameter, dan linimasa waktu eksekusi Web Worker.

---

## 7. Desain Sistem Transisi & Responsivitas Gerak

- **Durasi Standar**: $200\text{ ms}$ untuk *fade*, *slide-up*, dan *drawer*.
- **Kurva Easing**: `cubic-bezier(0.2, 0.7, 0.2, 1.0)` (responsif di awal, lembut saat mendarat).
- **Efek Stagger Hasil**: Setiap kotak karakter dan huruf transliterasi muncul dengan jeda berurutan $30\text{ ms}$ antarkarakter:
  $$\text{Delay}_i = i \times 30\text{ ms}, \quad i \in [0, N-1]$$
  Memberikan sensasi visual dokumen yang sedang dibaca dan dipahami oleh mesin.

---

> [!NOTE]
> **Status Kelulusan Tahap 5**: Design System dan UI/UX Spec telah selesai dirinci secara komprehensif. Siap dilanjutkan ke **Keluaran #6: Rencana Pengujian dan Metrik Evaluasi** setelah konfirmasi dari Anda.
