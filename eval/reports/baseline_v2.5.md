# Laporan Evaluasi Baseline — AksaraScan v2.5 (Fase 0)

- **Waktu Evaluasi**: `2026-11-25 14:52:06 +0700`
- **Commit Hash**: `492b7bc`
- **Seed Generator**: `108`
- **Model ONNX**: `app\web_camscanner\aksara_yolo.onnx` (`12.11 MB`)
- **Galeri 50 Kesalahan Terburuk**: [baseline_worst50.html](./baseline_worst50.html)

---

## 1. Uji Paritas Python (`onnxruntime` CPU) vs JavaScript Browser (`onnxruntime-web` WASM)

Diukur pada **20 gambar emas** ([parity_20.jsonl](../parity_20.jsonl)): 8 set diagnostik, 2 sampel web, dan 10 halaman validasi sintetis.

| Pengujian Paritas | Hasil Terukur | Interpretasi & Toleransi |
|---|---|---|
| Tensor Input Identik → Output Raw ONNX (59×3549) | max `|Δ| = 0.009125` | Toleransi koordinat kotak `[0..416]`: **`0.009125 px`**; probabilitas kelas `[0..1]`: **`5.01e-06`**. Engine ONNX CPU & Browser WASM identik secara numerik. |
| Praproses `cv2` vs `sharp` (416×416) | max `|Δpixel| = 0.5490` | `0.0000` pada gambar 416×416; selisih sub-piksel interpolasi muncul pada goresan tipis resolusi tinggi. |
| Kesepakatan Kotak Raw 416 (IoU ≥ 0.85, kelas sama) | `423/476` | `100%` identik (`354/354`) pada seluruh 10 gambar val 416×416 (`p11..p20`). |
| Kesepakatan Kotak Browser E2E vs Replika Python (IoU ≥ 0.70) | `540/555` | `540/555` (`97.3%`) kotak selaras di seluruh pipeline Chromium HTML5 Canvas + Tiling. |
| Kecocokan String Unicode Penuh Browser vs Python | `13/20` (`mean ΔCER = 0.06784`) | `10/10` identik di val + `3/3` identik di `p01..p03`; selisih hanya terjadi pada goresan pensil 1 px (`p04..p10`) karena filter binerisasi `morphBin` peka perbedaan resampling Canvas vs OpenCV. |
| Privasi Jaringan & Error Konsol saat Inferensi Browser | External reqs: `0`, Console err: `0` | **Zero egress** selama inferensi (`0` permintaan eksternal). |

---

## 2. Baseline End-to-End pada Set Diagnostik (`eval/failures/`, n=8)

> **PERINGATAN METODOLOGIS**: SET DIAGNOSTIK KECIL (n=8) — BUKAN real_test untuk klaim performa akhir!

- **Rata-rata Unicode CER**: **61.49%** (semakin rendah semakin baik)
- **Rata-rata Grapheme F1**: **55.49%**
- **Rata-rata Akurasi Kata (Sentence Latin)**: **32.29%**

| ID | Domain / Skenario | CER ↓ | Grapheme F1 ↑ | Word Acc ↑ | Oracle DP Acc ↑ | Masalah Utama (Ringkas) |
|---|---|---:|---:|---:|---:|---|
| `diag_01` | `printed_lowres_with_latin_caption` | 5.41% | 89.47% | 87.5% | 100.0% | Box=31 · OOV GT: `—` |
| `diag_02` | `printed_lowres_cropped_line` | 5.41% | 89.47% | 87.5% | 100.0% | Box=31 · OOV GT: `—` |
| `diag_03` | `synthetic_lined_paper` | 0.0% | 100.0% | 83.33% | 100.0% | Box=21 · OOV GT: `kothi` |
| `diag_04` | `handwriting_shadow_stress` | 88.89% | 22.22% | 0.0% | 0.0% | Box=10 · OOV GT: `ha, na, ca, ra` |
| `diag_05` | `handwriting_border_lines_stress` | 77.78% | 22.22% | 0.0% | 0.0% | Box=10 · OOV GT: `ha, na, ca, ra` |
| `diag_06` | `handwriting_lowres_600px` | 33.33% | 66.67% | 0.0% | 0.0% | Box=9 · OOV GT: `ha, na, ca, ra` |
| `diag_07` | `handwriting_latin_distractor_border` | 111.11% | 20.0% | 0.0% | 0.0% | Box=17 · OOV GT: `ha, na, ca, ra` |
| `diag_08` | `handwriting_chart_with_latin_labels` | 170.0% | 33.9% | 0.0% | 0.0% | Box=53 · OOV GT: `ha, na, ca, ra` |

### Rincian Prediksi vs Ground Truth Set Diagnostik

- **`diag_01`** (Gambar laporan user 761x131 px; taling terbaca pepet dan cecak terlewat pada kata pertama):
  - Unicode GT: `꧋ꦮꦺꦢꦁꦧꦸꦧꦸꦏ꧀ꦠꦤ꧀ꦥꦒꦸꦭ꧈ꦲꦗꦔꦩꦸꦏ꧀ꦩꦸꦤ꧀ꦝꦏ꧀ꦠꦸꦮ꧉`
  - Unicode Pred: `꧋ꦮꦼꦢꦧꦸꦧꦸꦏ꧀ꦠꦤ꧀ꦥꦒꦸꦭ꧈ꦲꦗꦔꦩꦸꦏ꧀ꦩꦸꦤ꧀ꦝꦏ꧀ꦠꦸꦮ꧉`
  - Latin Pred: `Wêda bubuk tanpa gula, aja ngamuk mundhak tuwa.` → Terjemahan: `Wêda bubuk tanpa gula, jangan marah-marah nanti makin tua.`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.` → `Minuman seduh bubuk tanpa gula, jangan marah-marah nanti makin tua.`
- **`diag_02`** (Potongan baris aksara tanpa teks Latin di atasnya):
  - Unicode GT: `꧋ꦮꦺꦢꦁꦧꦸꦧꦸꦏ꧀ꦠꦤ꧀ꦥꦒꦸꦭ꧈ꦲꦗꦔꦩꦸꦏ꧀ꦩꦸꦤ꧀ꦝꦏ꧀ꦠꦸꦮ꧉`
  - Unicode Pred: `꧋ꦮꦼꦢꦧꦸꦧꦸꦏ꧀ꦠꦤ꧀ꦥꦒꦸꦭ꧈ꦲꦗꦔꦩꦸꦏ꧀ꦩꦸꦤ꧀ꦝꦏ꧀ꦠꦸꦮ꧉`
  - Latin Pred: `Wêda bubuk tanpa gula, aja ngamuk mundhak tuwa.` → Terjemahan: `Wêda bubuk tanpa gula, jangan marah-marah nanti makin tua.`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Wédang bubuk tanpa gula, aja ngamuk mundhak tuwa.` → `Minuman seduh bubuk tanpa gula, jangan marah-marah nanti makin tua.`
- **`diag_03`** (Sampel bawaan app sample_4.png dengan pasangan, pangkon, cakra, taling-tarung, wulu):
  - Unicode GT: `ꦲꦏ꧀ꦱꦫꦗꦮ / ꦩꦔꦤ꧀ꦱꦼꦒ / ꦏꦿꦩꦏꦺꦴꦛꦶ`
  - Unicode Pred: `ꦲꦏ꧀ꦱꦫꦗꦮ / ꦩꦔꦤ꧀ꦱꦼꦒ / ꦏꦿꦩꦏꦺꦴꦛꦶ`
  - Latin Pred: `Aksara jawa mangan sêga krama kothi` → Terjemahan: `Aksara Jawa makan nasi krama/sopan kothi`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Aksara Jawa mangan sega krama kothi.` → `Aksara Jawa makan nasi krama/sopan kothi.`
- **`diag_04`** (Tulisan tangan digital tipis dengan bayangan gradien gelap di tengah):
  - Unicode GT: `ꦲꦤꦕꦫꦏ / ꦢꦱꦮꦭ`
  - Unicode Pred: `ꦫꦕꦫꦫꦫꦫ / ꦫꦫꦗ`
  - Latin Pred: `Racarararararar aja` → Terjemahan: `Racarararararar jangan`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Hanacaraka dasaw ala` → `Hanacaraka dasaw buruk`
- **`diag_05`** (Tulisan tangan tipis dengan garis tepi tebal mengganggu segmentasi):
  - Unicode GT: `ꦲꦤꦕꦫꦏ / ꦢꦱꦮꦭ`
  - Unicode Pred: `ꦫꦫꦕꦫꦫꦫꦫ / ꦫꦗ`
  - Latin Pred: `Raracararararar aja` → Terjemahan: `Raracararararar jangan`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Hanacaraka dasaw ala` → `Hanacaraka dasaw buruk`
- **`diag_06`** (Versi resolusi rendah 600x385 px):
  - Unicode GT: `ꦲꦤꦕꦫꦏ / ꦢꦱꦮꦭ`
  - Unicode Pred: `ꦲꦤꦩꦱꦏ / ꦢꦱꦮꦪ`
  - Latin Pred: `Ana masak adasawaya` → Terjemahan: `Ada memasak adasawaya`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Hanacaraka dasaw ala` → `Hanacaraka dasaw buruk`
- **`diag_07`** (Ada teks header Latin dan tepi gelap Scanner):
  - Unicode GT: `ꦲꦤꦕꦫꦏ / ꦢꦱꦮꦭ`
  - Unicode Pred: `꧋ꦫꦒꦫ / ꦫꦫꦫꦕꦫꦫꦫ`
  - Latin Pred: `Ragararararacararara` → Terjemahan: `Ragararararacararara`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Hanacaraka dasaw ala` → `Hanacaraka dasaw buruk`
- **`diag_08`** (20 Nglegena tulisan tangan berlabel Latin di atas tiap aksara):
  - Unicode GT: `ꦲꦤꦕꦫꦏ / ꦢꦠꦱꦮꦭ / ꦥꦝꦗꦪꦚ / ꦩꦒꦧꦛꦔ`
  - Unicode Pred: `ꦒꦱꦱꦒꦤꦱꦗꦱꦝ꧀ꦱ / ꦤꦤ / ꦲꦝꦫꦠ / ꦕꦤꦤ / ꦫꦢꦫꦫꦫꦮꦥ / ꦤ / ꦥꦝꦱꦗꦁ꧈ꦗꦥꦒ / ꦒꦤ / ꦩꦫꦒ`
  - Latin Pred: `Gasasag ana s aja sadhsan ana hadharatacan ana radarararaw apa n apa dhas aja ng, j apa gag ana maraga` → Terjemahan: `Gasasag ada s jangan sadhsan ada hadharatacan ada radarararaw apa n apa dhas jangan ng, j apa gag ada maraga`
  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `Hanacaraka datasawal apa d aja yanyamagabathanga` → `Hanacaraka datasawal apa d jangan yanyamagabathanga`

---

## 3. Baseline Deteksi pada Validasi Sintetis (`dataset_yolo/val`, n=240)

> **CATATAN**: Validasi sintetis ini berasal dari generator 3 font yang sama dengan data latih (mencakup *pool* campuran `Jawa.zip`), sehingga **BUKAN** representasi kinerja pada foto tulisan tangan buku tulis di dunia nyata.

| Kelompok | Jumlah Kelas | Kotak GT | mAP50 (%) | mAP50-95 (%) |
|---|---:|---:|---:|---:|
| `overall` | 55 | 8494 | **98.88%** | **87.84%** |
| `nglegena` | 20 | 3758 | **98.96%** | **90.92%** |
| `pasangan` | 20 | 1547 | **99.11%** | **86.12%** |
| `sandhangan` | 12 | 2901 | **98.21%** | **86.09%** |
| `pada` | 3 | 288 | **99.5%** | **85.83%** |

- **Titik Operasi Browser (`conf >= 0.25`)**: `TP=8263`, `Salah Kelas (Confusion)=72`, `Terlewat (FN)=159`, `Palsu (FP)=20`
- **Cluster-Level Val Proxy (GT Box Assembly vs Pred Box Assembly)**: Grapheme F1 = **95.46%**, CER = **2.82%**
- **Objek Kecil (`< 16 px` pada `416×416`)**: Total Recall = **89.57%** (`541/604` terdeteksi)

### 3.1 Recall Objek Kecil (`< 16 px`) per Kelas

| Kelas | Kategori | GT < 16 px | TP < 16 px | Recall (%) |
|---|---|---:|---:|---:|
| `na` | nglegena | 1 | 0 | 0.0% |
| `p_sa` | pasangan | 3 | 2 | 66.67% |
| `p_ja` | pasangan | 2 | 2 | 100.0% |
| `wulu` | sandhangan | 135 | 108 | 80.0% |
| `suku` | sandhangan | 1 | 1 | 100.0% |
| `pepet` | sandhangan | 10 | 10 | 100.0% |
| `tarung` | sandhangan | 33 | 32 | 96.97% |
| `layar` | sandhangan | 67 | 64 | 95.52% |
| `cecak` | sandhangan | 285 | 258 | 90.53% |
| `pada_lingsa` | pada | 59 | 56 | 94.92% |
| `pada_lungsi` | pada | 8 | 8 | 100.0% |

### 3.2 Kelas Terlemah (AP50 < 98.5% atau FN terbanyak) & Pasangan Tertukar

| Kelas Terlemah | Kategori | GT | AP50 (%) | AP50-95 (%) |
|---|---|---:|---:|---:|
| `cecak` | sandhangan | 285 | 92.69% | 55.47% |
| `wulu` | sandhangan | 285 | 94.79% | 82.42% |
| `p_la` | pasangan | 78 | 95.92% | 76.95% |
| `suku` | sandhangan | 555 | 97.55% | 79.0% |
| `cakra` | sandhangan | 130 | 97.9% | 93.11% |
| `da` | nglegena | 172 | 98.14% | 89.52% |
| `ra` | nglegena | 196 | 98.4% | 87.97% |
| `nya` | nglegena | 182 | 98.47% | 90.07% |
| `ya` | nglegena | 194 | 98.52% | 90.34% |
| `ma` | nglegena | 197 | 98.55% | 91.36% |
| `pa` | nglegena | 175 | 98.59% | 91.88% |
| `dha` | nglegena | 179 | 98.62% | 91.06% |

**Pasangan Kelas Paling Sering Tertukar pada `val` (IoU ≥ 0.45, `conf >= 0.25`)**:
- `wulu` $\rightarrow$ `cecak`: **10** kali
- `ta` $\rightarrow$ `ba`: **4** kali
- `ca` $\rightarrow$ `sa`: **3** kali
- `nya` $\rightarrow$ `ba`: **3** kali
- `ya` $\rightarrow$ `ha`: **2** kali
- `ya` $\rightarrow$ `ka`: **2** kali
- `ba` $\rightarrow$ `ta`: **2** kali
- `tha` $\rightarrow$ `nga`: **2** kali
- `p_la` $\rightarrow$ `p_ya`: **2** kali
- `p_dha` $\rightarrow$ `p_tha`: **2** kali

**Kelas Paling Sering Terlewat (*False Negative*) pada `val`**:
- `suku` (sandhangan): **31** terlewat dari 555 GT (Recall operasi `94.41%`)
- `cecak` (sandhangan): **26** terlewat dari 285 GT (Recall operasi `90.88%`)
- `wulu` (sandhangan): **21** terlewat dari 285 GT (Recall operasi `92.63%`)
- `p_la` (pasangan): **7** terlewat dari 78 GT (Recall operasi `91.03%`)
- `pepet` (sandhangan): **6** terlewat dari 292 GT (Recall operasi `97.95%`)
- `cakra` (sandhangan): **6** terlewat dari 130 GT (Recall operasi `95.38%`)
- `p_ha` (pasangan): **5** terlewat dari 65 GT (Recall operasi `92.31%`)
- `layar` (sandhangan): **5** terlewat dari 268 GT (Recall operasi `98.13%`)
- `ra` (nglegena): **4** terlewat dari 196 GT (Recall operasi `97.96%`)
- `da` (nglegena): **4** terlewat dari 172 GT (Recall operasi `97.67%`)

---

## 4. Taksonomi Kesalahan (Dasar Prioritas Fase 1–4)

### A. Taksonomi Kesalahan pada Validasi Sintetis (`val`, total 251 kesalahan kotak)

1. **Deteksi Gagal / Terlewat (*False Negative*)**: **63.3%** (159/251 kesalahan; didominasi tanda kecil `cecak` 26, `wulu` 21, `suku` 31, dan pasangan rapat `p_la` 7 / `p_ha` 5).
2. **Klasifikasi Keliru (*Confusion*)**: **28.7%** (72/251 kesalahan; #1 adalah `wulu` tertukar `cecak` sebanyak 10 kali, diikuti kemiripan bentuk *nglegena* `ta`/`ba`, `ca`/`sa`, `nya`/`ba`).
3. **Deteksi Palsu (*False Positive*)**: **8.0%** (20/251 kesalahan; tersering pada `layar` 5 dan `suku` 4).
4. **Celah Perakitan (`val`)**: Meskipun akurasi klasifikasi kotak 97.28% (`8263/8494`), *Cluster Grapheme F1* berada di **95.46%** (`CER = 2.82%`), mengindikasikan ~1.8 poin kehilangan akurasi berasal dari pelekatan kotak ke induk / pengurutan kluster.

### B. Taksonomi Kesalahan pada Set Diagnostik Nyata & Stres (`eval/failures/`, n=8)

Mengisolasi masing-masing tahap pipeline pada 8 kasus diagnostik (termasuk uji terisolasi DP+Leksikon dengan *input* Latin GT bersih):

1. **Klasifikasi & Pergeseran Domain Tulisan Tangan (Prioritas #1 — Tertinggi)**: Mengkontribusi mayoritas kegagalan ekstrem pada `diag_04..diag_08` (CER `33.3%..170.0%`), serta kesalahan `pepet`/`taling` (`wê` vs `wé`, skor `0.930`) pada `diag_01/02` karena distribusi latih hanya bergantung pada 3 font cetak dan potongan terisolasi `Vokal A`.
2. **Deteksi / Lokalisasi, Objek Kecil (< 16 px), & Teks Distraktor Latin**: Tanda `cecak` hilang total pada `diag_01/02` (Wédang $\rightarrow$ Wêda), huruf Latin pada `diag_07` dan `diag_08` memicu puluhan *False Positive* (`53` kotak terdeteksi padahal hanya `20` aksara pada `diag_08`!), dan *binarizer* `morphBin` kehilangan glif pada bayangan berat (`diag_04`).
3. **Perakitan Ortografi (`orthography.js`)**: Akurat (`0% CER`) pada tata letak cetak rapi seperti `diag_03` (`sample_4.png`), namun belum membuang hipotesis tak sah secara ortografis (lolos urutan seperti `sadhsan`, `ng, j`) maupun memanfaatkan skor top-k.
4. **Pemisah Kata DP (`javanese_lexicon.js`)**: Pada 3 kasus kalimat (`diag_01..diag_03`), saat diberi Latin GT bersih (`oracle_dp`), **3/3** kalimat tersegmen 100% benar — namun *rule-set* kamus masih tertutup sehingga kegagalan pengenalan hulu (`wêda`) tidak bisa dipulihkan, dan pada bagan karakter lepas (`diag_04..diag_08`) DP memaksa penggabungan suku kata lepas (`hanacaraka`, `alas`).
5. **Terjemahan & Cakupan Leksikon (`439` entri)**: Kata kalimat umum pada `diag_01/02` sudah tercakup (`0` OOV GT pada kalimat uji), namun kata luar kamus sempit (seperti `kothi` pada `diag_03` serta kata rusak `wêda`) langsung dilewatkan tanpa penandaan kata ragu atau alternatif.
