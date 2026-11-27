# Statistik & Dokumentasi Dataset AksaraScan v4.0 (`docs/DATASET_V4.md`)

Dihasilkan otomatis oleh `tools/datasets/summarize_dataset_v4.py` dari manifes v4 terverifikasi.

## 1. Status Audit Kebocoran Data (`tools/check_leakage.py`)
- **Status:** **`PASS_ZERO_LEAKAGE`**
- **Total rekaman diperiksa:** `18568`
- **Overlap `writer_id`:** `0`
- **Overlap `cluster_id`:** `0`
- **Overlap `SHA-256`:** `0`
- **Duplikat dekat lintas-split (`dHash+pHash+cos>=0.85`):** `0`

## 2. Ringkasan Sumber Data & Split v4 (`manifests_v4/`)

| Manifes / Sumber | Lisensi | Rekaman | Kotak YOLO | Penulis Unik | Cluster Unik | Rincian Split |
|---|---|---:|---:|---:|---:|---|
| `hanacaraka_v4` | CC0-1.0 | 1562 | 1562 | 79 | 78 | `test_heldout=238, val=239, train=1085` |
| `jawa_syllables_v4` | CC BY 4.0 | 1469 | 3361 | 1464 | 1464 | `val=221, train=1248` |
| `jawa_zip_v4` | CC BY 4.0 | 11060 | 11060 | 10721 | 10721 | `val=1634, train=9426` |
| `non_aksara_distractors_v4` | - | 1050 | 0 | 0 | 0 | `train=887, val=163` |
| `nusaaksara_eval_only_v4` | CC BY-SA 4.0 | 48 | 0 | 6 | 48 | `eval_only_external=48` |
| `setilanaji_lines_v4` | Apache-2.0 | 2207 | 0 | 1148 | 2203 | `train=2059, test_heldout=16, val=132` |
| `setilanaji_teacher_v4` | Apache-2.0 | 67 | 993 | 67 | 67 | `train=67` |
| `synth_v4` | SIL OFL 1.1 / CC0 / CC BY 4.0 | 1050 | 22794 | 1050 | 1050 | `train=900, val=150` |

## 3. Set Uji Nyata Terkunci (`eval/heldout_real/`)
- **Total citra:** `42` (dikunci oleh `eval/heldout_real/SHA256SUMS.lock`)
- `nusaaksara_acl2025_eval_only`: `18` citra
- `setilanaji_real_book_val`: `12` citra
- `hanacaraka_12_unseen_human_writers`: `12` citra

## 4. Histogram Kotak Kelas Tier 1 (55 Kelas Aktif)

| ID | Kelas | Kotak Nyata (train+val) | Penulis Nyata | Kotak Sintetis v4 | Total Aktif |
|---:|---|---:|---:|---:|---:|
| 0 | `ha` | 766 | 749 | 1009 | 1775 |
| 1 | `na` | 788 | 760 | 1023 | 1811 |
| 2 | `ca` | 738 | 734 | 432 | 1170 |
| 3 | `ra` | 767 | 731 | 647 | 1414 |
| 4 | `ka` | 849 | 810 | 1299 | 2148 |
| 5 | `da` | 751 | 746 | 604 | 1355 |
| 6 | `ta` | 768 | 760 | 682 | 1450 |
| 7 | `sa` | 794 | 771 | 1011 | 1805 |
| 8 | `wa` | 751 | 748 | 625 | 1376 |
| 9 | `la` | 743 | 738 | 635 | 1378 |
| 10 | `pa` | 772 | 759 | 609 | 1381 |
| 11 | `dha` | 750 | 748 | 393 | 1143 |
| 12 | `ja` | 763 | 744 | 446 | 1209 |
| 13 | `ya` | 841 | 838 | 426 | 1267 |
| 14 | `nya` | 754 | 748 | 394 | 1148 |
| 15 | `ma` | 796 | 780 | 726 | 1522 |
| 16 | `ga` | 777 | 761 | 554 | 1331 |
| 17 | `ba` | 706 | 694 | 581 | 1287 |
| 18 | `tha` | 245 | 245 | 349 | 594 |
| 19 | `nga` | 245 | 245 | 702 | 947 |
| 20 | `p_ha` | 0 | 0 | 16 | 16 |
| 21 | `p_na` | 0 | 0 | 23 | 23 |
| 22 | `p_ca` | 0 | 0 | 14 | 14 |
| 23 | `p_ra` | 0 | 0 | 66 | 66 |
| 24 | `p_ka` | 0 | 0 | 55 | 55 |
| 25 | `p_da` | 0 | 0 | 43 | 43 |
| 26 | `p_ta` | 0 | 0 | 51 | 51 |
| 27 | `p_sa` | 0 | 0 | 61 | 61 |
| 28 | `p_wa` | 0 | 0 | 18 | 18 |
| 29 | `p_la` | 0 | 0 | 26 | 26 |
| 30 | `p_pa` | 0 | 0 | 69 | 69 |
| 31 | `p_dha` | 0 | 0 | 45 | 45 |
| 32 | `p_ja` | 0 | 0 | 13 | 13 |
| 33 | `p_ya` | 0 | 0 | 64 | 64 |
| 34 | `p_nya` | 0 | 0 | 51 | 51 |
| 35 | `p_ma` | 0 | 0 | 20 | 20 |
| 36 | `p_ga` | 0 | 0 | 47 | 47 |
| 37 | `p_ba` | 0 | 0 | 32 | 32 |
| 38 | `p_tha` | 0 | 0 | 39 | 39 |
| 39 | `p_nga` | 0 | 0 | 20 | 20 |
| 40 | `wulu` | 527 | 497 | 1625 | 2152 |
| 41 | `suku` | 182 | 75 | 671 | 853 |
| 42 | `pepet` | 269 | 222 | 1340 | 1609 |
| 43 | `taling` | 859 | 854 | 2028 | 2887 |
| 44 | `tarung` | 423 | 423 | 1293 | 1716 |
| 45 | `layar` | 24 | 23 | 108 | 132 |
| 46 | `cecak` | 24 | 20 | 384 | 408 |
| 47 | `wignyan` | 8 | 8 | 141 | 149 |
| 48 | `pangkon` | 51 | 37 | 135 | 186 |
| 49 | `cakra` | 0 | 0 | 70 | 70 |
| 50 | `keret` | 0 | 0 | 78 | 78 |
| 51 | `pengkal` | 7 | 7 | 52 | 59 |
| 52 | `pada_adeg` | 0 | 0 | 724 | 724 |
| 53 | `pada_lingsa` | 0 | 0 | 204 | 204 |
| 54 | `pada_lungsi` | 0 | 0 | 21 | 21 |

## 5. Kebijakan Kelas Tier 2 & Tier 3 (Ditunda / *Deferred*)
Sesuai aturan skema `dataset/schema/classes_v4.yaml`, kelas Tier 2/3 hanya diaktifkan bila memiliki `>= 100` kotak nyata teranotasi dari `>= 5` penulis/cluster independen.
- Sumber bebas-lisensi saat ini hanya memiliki catatan baris tanpa kotak karakter di `setilanaji`: `ꦣ (U+A9A3)`: `112`
- Dataset Murda `RayFox52` ditandai **`DIBLOKIR`** (tanpa lisensi), dan ekspor Roboflow/Kaggle menunggu kunci API/unduhan manual. Maka **seluruh kelas Tier 2/3 tetap berstatus `deferred`** pada Fase 1 agar tidak melatih kelas hantu.

## 6. Contact Sheet Audit Visual (`dataset/external/reports/`)
- `v4_contact_hanacaraka.jpg`
- `v4_contact_jawa_zip.jpg` & `v4_contact_syllables.jpg`
- `v4_contact_setilanaji.jpg` & `v4_contact_nusaaksara_eval.jpg`
- `v4_contact_synth_tps.jpg` (*TPS Elastic Handwriting*)
- `v4_contact_synth_multiink_wall.jpg` (*Pulpen Multi-Warna, Garis Tembok/Buku, Dark Mode*)
- `v4_contact_synth_hardneg_latin.jpg` (*Teks Latin Campuran & Hard Negatives*)
- `v4_contact_heldout_real.jpg` (*42 Citra Evaluasi Terkunci*)
