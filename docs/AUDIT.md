# Audit Teknis & Kepatuhan Lisensi — AksaraScan v2.5 (Fase 0)

> Audit menyeluruh terhadap repositori `RayhanNajib/Aksara_Scan` pada kondisi **v2.5** (commit dasar audit `492b7bc`) sebelum masuk ke akuisisi dataset Fase 1, sintesis domain-acak Fase 2, pelatihan bertahap Fase 3, dan dekoder berpenalaran Fase 4.

---

## 1. Arsitektur Aktual & Alur Data

AksaraScan berjalan **100% di sisi klien (browser)** tanpa backend API inferensi, memakai arsitektur mesin ganda (*Dual-Engine*) dengan *fallback* PWA offline (`sw.js`, cache `aksarascan-v8`, deploy Vercel dari `app/web_camscanner` sesuai `vercel.json`):

```
[Input: Kamera 6-frame avg / Upload File / Paste / Drop / Seleksi ROI]
        │  resize maxDim = 1500 px  (app.js:331-334)
        ▼
[Praproses Citra Klasik 4 Tahap (app.js:753-795)]
  1. Grayscale ITU-R BT.601: round(0.299R + 0.587G + 0.114B)
  2. Otsu sentral (margin 12%) + adaptiveBinary: min(T*, localMean * 0.88)
     + clearBorderNoise (hapus komponen tepi > 1500 px atau > 2% area)
  3. Median 3×3 preservasi goresan tipis (ink>=1 dipertahankan; bg>=7 diisi)
  4. Morfologi hapus garis horizontal (kernel 1×13..31) + closing 3×3 -> morphBin
        │
        ├───[Mesin Utama: YOLOv8-Nano ONNX (default saat HTTP + ort siap)]───┐
        │    app.js:497-720 (memakai RGB sourceImage + masker morphBin)       │
        │      - Cari bbox tinta & lineBands horizontal dari morphBin         │
        │      - Line-Adaptive Tiling (maks 12 petak; fallback grid 480 px)    │
        │      - Letterbox RGB 416×416 pad=(114,114,114) ke Float32 [0..1]    │
        │      - ONNX Runtime Web WASM 1.20.1 (`aksara_yolo.onnx`, 12.11 MB)  │
        │      - Argmax tunggal (conf > 0.25) + buang tepi petak (<= 4 px)     │
        │      - NMS sadar-kategori (same/excl IoU=0.42, sandhangan=0.72)     │
        │      ▼                                                              │
        │    orthography.js:48-347 (reconstructJavaneseText)                  │
        │      - Klastering baris -> pasang Pasangan & Sandhangan via         │
        │        heuristik rasio jarak -> rakit string Unicode & suku kata    │
        │      ▼                                                              │
        │    javanese_lexicon.js:149-277 (translateJavaneseTokens)            │
        │      - DP pemisah kata + peluluhan `h` + kamus 439 entri -> ID     │
        │                                                                     │
        └───[Mesin Klasik: PCD + SVM Linear (20 Nglegena, v1.0)]──────────────┘
             app.js:804-844
               - CCL 8-arah + klaster baris + gabung stroke rapat
               - normCrop 96×96 (inner 76) -> Gaussian blur sigma=2.5
               - LinearSVC OvR bobot float16 (`model.json` / `model.js`)
```

---

## 2. Daftar 55 Kelas Saat Ini (`yolo_meta.json`)

Model `app/web_camscanner/aksara_yolo.onnx` mengeluarkan tensor `(1, 59, 3549)` (4 koordinat `cx, cy, w, h` + 55 probabilitas kelas):

| ID | Kelompok | Daftar Kelas | Posisi Spasial | Jumlah |
|---|---|---|---|---:|
| `0..19` | `nglegena` | `ha`, `na`, `ca`, `ra`, `ka`, `da`, `ta`, `sa`, `wa`, `la`, `pa`, `dha`, `ja`, `ya`, `nya`, `ma`, `ga`, `ba`, `tha`, `nga` | `base` | 20 |
| `20..39` | `pasangan` | `p_ha`, `p_na`, `p_ca`, `p_ra`, `p_ka`, `p_da`, `p_ta`, `p_sa`, `p_wa`, `p_la`, `p_pa`, `p_dha`, `p_ja`, `p_ya`, `p_nya`, `p_ma`, `p_ga`, `p_ba`, `p_tha`, `p_nga` | `pasangan` (subskrip / kanan utk `ha/sa/pa`) | 20 |
| `40..51` | `sandhangan` | `wulu` (`ꦶ`, above), `suku` (`ꦸ`, below), `pepet` (`ꦼ`, above), `taling` (`ꦺ`, left), `tarung` (`ꦴ`, right), `layar` (`ꦂ`, above), `cecak` (`ꦁ`, above), `wignyan` (`ꦃ`, right), `pangkon` (`꧀`, right), `cakra` (`ꦿ`, below), `keret` (`ꦽ`, below), `pengkal` (`ꦾ`, right) | `above/below/left/right` | 12 |
| `52..54` | `pada` | `pada_adeg` (`꧋`), `pada_lingsa` (`꧈`), `pada_lungsi` (`꧉`) | `pada` | 3 |
| **Total** | | | | **55** |

### Kelas Blok Unicode Javanese (`U+A980..U+A9DF`) yang BELUM Tercakup

1. **Aksara Murda (+ Pasangannya)**:
   - Konsonan kapital kehormatan: `ꦑ` (Ka Murda `U+A991`), `ꦓ` (Ga Murda `U+A993`), `ꦖ` (Ca Murda `U+A996`), `ꦘ` (Nya Murda `U+A998`), `ꦟ` (Na Murda `U+A99F`), `ꦡ` (Ta Murda `U+A9A1`), `ꦦ` (Pa Murda `U+A9A6`), `ꦨ` (Ba Murda `U+A9A8`), `ꦯ` (Sa Murda `U+A9AF`), beserta *pasangan*-nya.
2. **Aksara Swara & Karakter Vokal Wajib Khusus**:
   - **Sangat kritis (sering muncul di kalimat biasa)**: **`ꦉ` (*Pa Cerek*, `rê`, `U+A989`)** dan **`ꦊ` (*Nga Lelet*, `lê`, `U+A98A`)**, karena dalam ejaan baku tidak sah menulis `ra+pepet` atau `la+pepet`.
   - Vokal mandiri: `ꦄ` (A `U+A984`), `ꦅ` (I Kawi `U+A985`), `ꦆ` (I `U+A986`), `ꦇ` (Ii `U+A987`), `ꦈ` (U `U+A988`), `ꦋ` (Nga Lelet Raswadi `U+A98B`), `ꦌ` (E `U+A98C`), `ꦍ` (Ai `U+A98D`), `ꦎ` (O `U+A98E`).
3. **Aksara Rekan & Tanda Diakritik Serapan**:
   - **`꦳` (*cecak telu*, `U+A9B3`)** untuk memodifikasi bunyi serapan Arab/asing (*kha*, *dza*, *fa/va*, *za*, *gha*), serta konsonan tambahan (`ꦐ` Ka Sasak, `ꦬ` Ra Agung, `ꦜ` Tha Mahaprana, `ꦞ` Dha Mahaprana, `ꦰ` Sa Mahaprana).
4. **Sandhangan Tambahan & Panjang**:
   - `ꦷ` (*wulu melik*, `U+A9B7`), `ꦹ` (*suku mendut*, `U+A9B9`), `ꦻ` (*dirga mure*, `U+A9BB`), `ꦵ` (*toleng*, `U+A9B5`), `ꦀ` (*panyangga*, `U+A980`).
5. **Angka Jawa (`U+A9D0..U+A9D9`)**:
   - `꧐, ꧑, ꧒, ꧓, ꧔, ꧕, ꧖, ꧗, ꧘, ꧙` (`0..9`).
6. **Tanda Baca *Pada* Lainnya**:
   - **`꧇` (*pada pangkat*, `U+A9C7`)** pembatas angka Jawa, serta `꧃ ꧄ ꧅` (*pada andhap/madya/luhur*), `꧊` (*pada guru*), `꧌` (*pada pancak*), `ꧏ` (*pada rangkep* untuk reduplikasi kata), `꧞` (*tirta tumetes*), `꧟` (*isèn-isèn*).

---

## 3. Parameter Tiling, Ambang Konfidensi, & Titik Kehilangan Informasi

Seluruh titik berikut diaudit langsung dari `app/web_camscanner/app.js`, `orthography.js`, `javanese_lexicon.js`, dan `yolo_meta.json`:

| No | Titik Pipeline | Konfigurasi Aktual | Informasi yang Terbuang / Risiko Kegagalan | Lokasi Kode |
|---:|---|---|---|---|
| 1 | Resampling input | `maxDim = 1500` px | Foto dokumen multi-baris beresolusi > 1500 px menyusutkan *cecak* & *wulu* menjadi < 6–8 px sebelum di-crop. | `app.js:331` |
| 2 | Binerisasi menggerbang crop YOLO | `lineBands` & bbox tinta dibaca dari `morphBin` (Otsu+adaptiveBinary+clearBorderNoise+lineRemoval) | Walaupun YOLO membaca RGB mentah, area pemindaian dipilih oleh biner klasik! Bayangan berat, goresan tipis terputus, atau garis tepi meja membuat `morphBin` kehilangan glif atau memotong baris. | `app.js:503-556`, `app.js:798` |
| 3 | Filter tinggi pita baris | `activeBands = lineBands.filter(b => height >= max(20, maxLH * 0.45))` | Baris pendek atau baris tanpa pasangan tinggi dapat tereliminasi bila ada elemen vertikal tinggi (mis. bingkai/coretan) pada baris lain. | `app.js:560-563` |
| 4 | Batas jumlah petak & padding vertikal | `tiles.length >= 12` berhenti paksa; `padV` dipotong `minTopBound/maxBotBound` | Dokumen panjang > 12 petak **terpotong diam-diam** di ujung kanan/bawah. Tanda tinggi (*cecak/wulu*) berpotensi terpotong `padV` bila baris berdekatan. | `app.js:568-611` |
| 5 | Buang kotak di tepi petak | `(tx0 > baseX0 && x0 <= tx0 + 4) \|\| (tx1 < baseX1 && x1 >= tx1 - 4)` | Glif sempit (*taling*, *tarung*, `pada_adeg`) atau glif besar di irisan petak tetangga bisa terbuang oleh kedua petak sekaligus. | `app.js:668-670` |
| 6 | **Argmax tunggal pra-perakitan** | `bestCls`, `bestScore > confThr (0.25)` | Seluruh probabilitas kelas ke-2..ke-5 dibuang! `orthography.js` tidak pernah tahu bahwa `pepet (0.52)` hampir imbang dengan `taling (0.47)`, atau *cecak* berskor `0.21` terbuang mutlak. | `app.js:652-658`, `yolo_meta.json:4` |
| 7 | Kuantisasi koordinat & inkonsistensi ambang ragu | `Math.round(box)` di `app.js`; ambang ragu `0.45` (`app.js`) vs `0.40` (`orthography.js`) | Hilang presisi sub-piksel untuk tanda mini dan status ragu tidak seragam antar-tahap. | `app.js:685-686`, `orthography.js:229,323` |
| 8 | NMS kategori greedy | `sameCatIou = 0.42` antara `nglegena`, `pasangan`, dan `pada` | *Pasangan* posisi kanan (`p_ha`, `p_sa`, `p_pa`) atau bertumpuk rapat dengan induknya dapat saling menekan bila IoU > 0.42. | `app.js:481-495` |
| 9 | Perakitan heuristik tanpa validasi Unicode | Batas jarak tetap (`bm.h * 2.8`, `bm.w * 1.4`), pengurutan kluster per `avgBaseH * 1.05` | Gagal pada kemiringan halaman/tulisan miring, tidak ada validator aturan Blok Unicode `U+A980..U+A9DF` (duplikasi vokal atau urutan tak sah lolos diam-diam). | `orthography.js:101-310` |
| 10 | Leksikon tertutup & DP tanpa morfologi/alternatif | Kamus `JAVANESE_DICT` kecil (439 entri), tanpa aturan imbuhan, kata OOV diloloskan mentah | Kata berimbuhan (*ngamuk* kecuali terdaftar, *dakjupuk*, *tulisen*) & variasi `a/o` / `é/è/ê` tak bisa dipecah/diterjemahkan, dan kesalahan pengenalan (*Wêda*) tidak memicu pencarian hipotesis alternatif. | `javanese_lexicon.js:8-210` |

---

## 4. Audit Kecocokan Lisensi (Wajib Keputusan User — `ATURAN KERAS #5`)

Audit langsung atas lingkungan terpasang dan berkas repo menemukan **konflik lisensi serius** antara pustaka pelatihan dan lisensi repositori:

1. **Konflik Inti (`MIT` vs `AGPL-3.0`)**:
   - `LICENSE` repositori AksaraScan adalah **MIT License**.
   - `training/train_yolo.py` memakai pustaka **`ultralytics==8.4.218`**, yang terverifikasi pada metadata paket Python berlisensi **`AGPL-3.0`** (*GNU Affero General Public License v3 or later*).
   - Di bawah **AGPL-3.0**, melatih dan mendistribusikan bobot model turunan atau kode aplikasi jaringan yang bergantung pada kode Ultralytics memicu kewajiban copyleft AGPL-3.0 terhadap seluruh aplikasi (kecuali memiliki *Ultralytics Enterprise License*). Ini **tidak kompatibel dengan MIT** pada distribusi komersial/terbuka tanpa pengungkapan AGPL-3.0 atau persetujuan lisensi ulang.
2. **Audit Font Sintetis (`generate_yolo_dataset.py`)**:
   - `NotoSansJavanese.ttf` $\rightarrow$ terverifikasi **SIL Open Font License 1.1** (`notofonts/javanese`). Aman untuk sintesis dan bundling.
   - `TuladhaJejegOT-Regular.ttf/.otf` $\rightarrow$ terverifikasi **SIL Open Font License 1.1** (`akufadhl/Tuladha-Jejeg-OT/OFL.txt`). Aman.
   - `C:/Windows/Fonts/javatext.ttf` (*Javanese Text*, Microsoft) $\rightarrow$ **Font proprieter Windows**. Tidak boleh didistribusikan ulang dan status hukum penggunaannya sebagai data latih model publik ambigu; sebaiknya diganti murni dengan font OFL 1.1 di Fase 2.
3. **Audit Dataset Lokal**:
   - `dataset/hanacaraka/` (1.579 citra): **CC0-1.0** (*Public Domain*), aman.
   - `Jawa.zip` (*Indonesian Local Script Characters*, Mendeley DOI `10.17632/vfj32bpjsf.1`): terverifikasi **CC BY 4.0** (wajib atribusi eksplisit di `DATASET_CREDITS.md` untuk model v2.5+ yang memakainya). Catatan kebocoran data pada v2.5 saat ini: `generate_yolo_dataset.py` mengambil 70 citra/kelas `Vokal A` (1.400 sampel) lalu mencampurnya ke dalam split `train` maupun `val` tanpa pemisahan writer/file, serta belum memanfaatkan **2.196 citra `Vokal E/I/O/U/È`** dan **1.066 `variations`** yang sudah ada di dalam `Jawa.zip`.

### Tiga Opsi Penyelesaian Lisensi Pustaka Pelatihan (Meminta Persetujuan Anda)

- **Opsi A (Direkomendasikan untuk tetap MIT bersih)**: Migrasikan pelatihan detektor Fase 3 ke kerangka berlisensi **Apache-2.0 / MIT**, seperti **RT-DETRv2 / D-FINE** (via Hugging Face `transformers` atau kode resmi Apache-2.0) atau **YOLOX** (Apache-2.0), dan/atau bangun jalur penantang CRNN-CTC/TrOCR di PyTorch (BSD-3). Model `aksara_yolo.onnx` v2.5 yang sudah ada tetap dicatat asal latihnya di `MODEL_CARD.md`, sementara rilis v3.0 bersih di bawah MIT/Apache-2.0.
- **Opsi B (Ubah lisensi repositori ke AGPL-3.0)**: Pertahankan ekosistem **Ultralytics YOLOv8/YOLO11**, namun perbarui `LICENSE` proyek (atau bagian kode + bobot YOLO) menjadi **AGPL-3.0** agar patuh secara hukum, khususnya karena AksaraScan memang *open-source* di GitHub.
- **Opsi C (Akademik non-komersial dengan catatan eksplisit)**: Gunakan skrip latih Ultralytics hanya untuk eksperimen tugas kuliah dengan mencantumkan pemisahan lisensi tegas di `LICENSE` + `MODEL_CARD.md` (kode web app MIT vs skrip/bobot turunan YOLOv8 AGPL-3.0), tetap mengetahui batasannya jika kelak dipublikasikan sebagai layanan komersial.

---

## 5. Ketidakcocokan Klaim Prompt/Docs vs Isi Repo Sebenarnya

Sesuai aturan pelaporan *"jika klaim tidak cocok dengan isi repo, percayai repo dan laporkan selisihnya"*:

1. **Label UI & Dokumentasi Tertinggal pada `v2.0` vs `v2.5`**:
   - `app/web_camscanner/index.html:166` masih menulis `52 Kelas · Nglegena+Sandhangan+Pasangan` (kode & `yolo_meta.json` sebenarnya **55 kelas** dengan 3 *Pada*).
   - `app/web_camscanner/index.html:85,375` masih menulis `v2.0`, dan `index.html:364` menulis `"Pilih salah satu dari 20 aksara Nglegena"` meskipun modal di mode YOLO merender 55 tombol.
   - `app/web_camscanner/app.js:429,444` fallback badge tertulis `'52 Kelas'`.
   - `README.md` mencampur judul `# AksaraScan v2.5` dengan tabel dan daftar berkas versi v2.0 (`52 kelas`, `98.4% mAP50`, `1.050 halaman`), serta belum mencantumkan `orthography.js`, `javanese_lexicon.js`, dan `pada`.
2. **Jalur `Jawa.zip` di Skrip Pelatihan**:
   - `training/generate_yolo_dataset.py:19` mencari `dataset/Jawa.zip`, padahal `.gitignore` dan `DATASET_CREDITS.md` menunjuk `dataset/indonesian-scripts/Jawa.zip`, sedangkan arsip fisiknya ada di direktori lokal saudara `../Projek_Akhir/dataset/Jawa.zip`.
3. **Dataset `DATASET_CREDITS.md` v1 vs Penggunaan Nyata v2.5**:
   - `DATASET_CREDITS.md:41` masih menyatakan `Jawa.zip` *"NOT part of the shipped MVP model"*, padahal v2.0 dan v2.5 (`aksara_yolo.onnx`) sudah memakai 1.400 potongan `Vokal A` dari `Jawa.zip` serta 3 font (`javatext.ttf`, `NotoSansJavanese.ttf`, `TuladhaJejegOT-Regular.ttf`).
4. **Rujukan Sumber Tambahan Terverifikasi (Hasil Audit Web Fase 0)**:
   - Tambahan temuan berlisensi jelas di luar tabel awal prompt:
     - **`setilanaji/aksara-jawa-ocr`** (Hugging Face, format PPOCRLabel baris kalimat Unicode berpasangan & bersandhangan, **`Apache-2.0`** — sangat potensial untuk Track B & evaluasi OCR).
     - **`akufadhl/Tuladha-Jejeg-OT`** (`SIL OFL 1.1` terverifikasi di `OFL.txt` & header TTF/OTF) & **`notofonts/javanese`** (`SIL OFL 1.1`).
     - **`IndoNLP/nusax` & `indonlp/NusaX-MT`** (kode `Apache-2.0`, dataset `CC-BY-SA-4.0`) serta **`IndoNLP/nusa-writes` / `indonlp/nusatranslation_mt`** (`Apache-2.0`).
     - **`nsulistiyawan/sastra-jawa`** (`MIT`), **`masnormen/carakanjs`** (`MIT`), **`junwatu/aksara.js`** (`MIT`), **`hermansh-id/Aksara-Jawa-OCR`** (`MIT`).
     - Perlu konfirmasi lisensi (`PERLU_KONFIRMASI` pada Fase 1): `NusaAksara/NusaAksara` (tidak mencantumkan field lisensi pada kartu Hugging Face), `RayFox52/Aksara-jawa-carakan-murda`, `SEACrowd/korpus_nusantara` (`Unknown`), dan dataset Kaggle/Roboflow yang belum diverifikasi kartu lisensinya secara lokal.
