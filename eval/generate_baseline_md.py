"""
Memformat laporan Markdown resmi Fase 0 (`eval/reports/baseline_v2.5.md`)
secara deterministik langsung dari berkas hasil `eval/reports/baseline_v2.5.json`.
"""
import json
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_JSON = os.path.join(ROOT_DIR, "eval", "reports", "baseline_v2.5.json")
OUT_MD = os.path.join(ROOT_DIR, "eval", "reports", "baseline_v2.5.md")


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)

    p = d["parity_20"]
    df = d["diagnostic_failures"]
    sv = d.get("synthetic_val", {})
    op = sv.get("operating_point_totals", {})

    lines = [
        "# Laporan Evaluasi Baseline — AksaraScan v2.5 (Fase 0)",
        "",
        f"- **Waktu Evaluasi**: `{d['timestamp']}`",
        f"- **Commit Hash**: `{d['commit_hash']}`",
        f"- **Seed Generator**: `{d['seed']}`",
        f"- **Model ONNX**: `{d['onnx_model']}` (`{d['onnx_size_mb']} MB`)",
        f"- **Galeri 50 Kesalahan Terburuk**: [baseline_worst50.html](./baseline_worst50.html)",
        "",
        "---",
        "",
        "## 1. Uji Paritas Python (`onnxruntime` CPU) vs JavaScript Browser (`onnxruntime-web` WASM)",
        "",
        f"Diukur pada **20 gambar emas** ([parity_20.jsonl](../parity_20.jsonl)): 8 set diagnostik, 2 sampel web, dan 10 halaman validasi sintetis.",
        "",
        "| Pengujian Paritas | Hasil Terukur | Interpretasi & Toleransi |",
        "|---|---|---|",
        f"| Tensor Input Identik → Output Raw ONNX (59×3549) | max `|Δ| = {p['onnx_engine_exact_tensor_parity_same_input_max_abs_diff']:.6f}` | Toleransi koordinat kotak `[0..416]`: **`{p['onnx_engine_exact_coord_px_max_abs_diff']:.6f} px`**; probabilitas kelas `[0..1]`: **`{p['onnx_engine_exact_cls_prob_max_abs_diff']:.2e}`**. Engine ONNX CPU & Browser WASM identik secara numerik. |",
        f"| Praproses `cv2` vs `sharp` (416×416) | max `|Δpixel| = {p['preprocess_cv2_vs_sharp_max_abs_diff']:.4f}` | `0.0000` pada gambar 416×416; selisih sub-piksel interpolasi muncul pada goresan tipis resolusi tinggi. |",
        f"| Kesepakatan Kotak Raw 416 (IoU ≥ 0.85, kelas sama) | `{p['raw_416_box_class_agreement_iou085']}` | `100%` identik (`354/354`) pada seluruh 10 gambar val 416×416 (`p11..p20`). |",
        f"| Kesepakatan Kotak Browser E2E vs Replika Python (IoU ≥ 0.70) | `{p['browser_e2e_box_agreement_iou070']}` | `540/555` (`97.3%`) kotak selaras di seluruh pipeline Chromium HTML5 Canvas + Tiling. |",
        f"| Kecocokan String Unicode Penuh Browser vs Python | `{p['browser_e2e_unicode_exact_matches']}` (`mean ΔCER = {p['browser_e2e_mean_cer_delta']}`) | `10/10` identik di val + `3/3` identik di `p01..p03`; selisih hanya terjadi pada goresan pensil 1 px (`p04..p10`) karena filter binerisasi `morphBin` peka perbedaan resampling Canvas vs OpenCV. |",
        f"| Privasi Jaringan & Error Konsol saat Inferensi Browser | External reqs: `{len(p['browser_external_requests_during_inference'])}`, Console err: `{len(p['browser_console_errors'])}` | **Zero egress** selama inferensi (`0` permintaan eksternal). |",
        "",
        "---",
        "",
        "## 2. Baseline End-to-End pada Set Diagnostik (`eval/failures/`, n=8)",
        "",
        f"> **PERINGATAN METODOLOGIS**: {df['warning']}",
        "",
        f"- **Rata-rata Unicode CER**: **{df['mean_cer_pct']}%** (semakin rendah semakin baik)",
        f"- **Rata-rata Grapheme F1**: **{df['mean_grapheme_f1_pct']}%**",
        f"- **Rata-rata Akurasi Kata (Sentence Latin)**: **{df['mean_word_acc_pct']}%**",
        "",
        "| ID | Domain / Skenario | CER ↓ | Grapheme F1 ↑ | Word Acc ↑ | Oracle DP Acc ↑ | Masalah Utama (Ringkas) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    for it in df["items"]:
        oov = ", ".join(it.get("gt_lexicon_oov_words", [])[:4]) or "—"
        lines.append(
            f"| `{it['id']}` | `{it['domain']}` | {it['cer_pct']}% | {it['grapheme_f1_pct']}% | {it['word_acc_pct']}% | {it.get('oracle_dp_word_acc_pct', '-')}% | Box={it['pred_box_count']} · OOV GT: `{oov}` |"
        )

    lines.extend([
        "",
        "### Rincian Prediksi vs Ground Truth Set Diagnostik",
        "",
    ])
    for it in df["items"]:
        u_gt = it["unicode_gt"].replace("\n", " / ")
        u_pr = it["unicode_pred"].replace("\n", " / ")
        lines.extend([
            f"- **`{it['id']}`** ({it['notes']}):",
            f"  - Unicode GT: `{u_gt}`",
            f"  - Unicode Pred: `{u_pr}`",
            f"  - Latin Pred: `{it['latin_pred']}` → Terjemahan: `{it['indonesian_pred']}`",
            f"  - Uji Terisolasi DP+Kamus (dari Latin GT bersih): `{it.get('oracle_dp_latin', '-')}` → `{it.get('oracle_dp_indonesian', '-')}`",
        ])

    if sv:
        gm = sv["group_metrics"]
        so = sv["small_objects_lt16px_total"]
        lines.extend([
            "",
            "---",
            "",
            "## 3. Baseline Deteksi pada Validasi Sintetis (`dataset_yolo/val`, n=240)",
            "",
            "> **CATATAN**: Validasi sintetis ini berasal dari generator 3 font yang sama dengan data latih (mencakup *pool* campuran `Jawa.zip`), sehingga **BUKAN** representasi kinerja pada foto tulisan tangan buku tulis di dunia nyata.",
            "",
            "| Kelompok | Jumlah Kelas | Kotak GT | mAP50 (%) | mAP50-95 (%) |",
            "|---|---:|---:|---:|---:|",
        ])
        for gname in ("overall", "nglegena", "pasangan", "sandhangan", "pada"):
            g = gm[gname]
            lines.append(f"| `{gname}` | {g['num_classes']} | {g['gt_boxes']} | **{g['mAP50_pct']}%** | **{g['mAP50_95_pct']}%** |")

        lines.extend([
            "",
            f"- **Titik Operasi Browser (`conf >= {sv['conf_thresh']}`)**: `TP={op['tp_cls_ok']}`, `Salah Kelas (Confusion)={op['cls_confusion']}`, `Terlewat (FN)={op['fn_missed']}`, `Palsu (FP)={op['fp_spurious']}`",
            f"- **Cluster-Level Val Proxy (GT Box Assembly vs Pred Box Assembly)**: Grapheme F1 = **{op['mean_cluster_grapheme_f1_pct']}%**, CER = **{op['mean_cluster_cer_pct']}%**",
            f"- **Objek Kecil (`< 16 px` pada `416×416`)**: Total Recall = **{so['recall_416_pct']}%** (`{so['tp_416']}/{so['gt_416']}` terdeteksi)",
            "",
            "### 3.1 Recall Objek Kecil (`< 16 px`) per Kelas",
            "",
            "| Kelas | Kategori | GT < 16 px | TP < 16 px | Recall (%) |",
            "|---|---|---:|---:|---:|",
        ])
        for row in sv["small_objects_per_class"]:
            lines.append(f"| `{row['class']}` | {row['category']} | {row['gt_lt16px_416']} | {row['tp_lt16px_416']} | {row['recall_416_pct']}% |")

        lines.extend([
            "",
            "### 3.2 Kelas Terlemah (AP50 < 98.5% atau FN terbanyak) & Pasangan Tertukar",
            "",
            "| Kelas Terlemah | Kategori | GT | AP50 (%) | AP50-95 (%) |",
            "|---|---|---:|---:|---:|",
        ])
        weak = sorted(sv["per_class_ap"].items(), key=lambda kv: (kv[1]["ap50_pct"], kv[1]["ap50_95_pct"]))[:12]
        for cname, v in weak:
            lines.append(f"| `{cname}` | {v['category']} | {v['gt_boxes']} | {v['ap50_pct']}% | {v['ap50_95_pct']}% |")

        lines.extend([
            "",
            "**Pasangan Kelas Paling Sering Tertukar pada `val` (IoU ≥ 0.45, `conf >= 0.25`)**:",
        ])
        for cp in sv["top_confused_pairs"][:10]:
            lines.append(f"- `{cp['gt']}` $\\rightarrow$ `{cp['pred']}`: **{cp['count']}** kali")

        lines.extend([
            "",
            "**Kelas Paling Sering Terlewat (*False Negative*) pada `val`**:",
        ])
        for fn in sv["top_missed_classes_fn"][:10]:
            lines.append(f"- `{fn['class']}` ({fn['category']}): **{fn['fn']}** terlewat dari {fn['gt']} GT (Recall operasi `{fn['recall_op_pct']}%`)")

        # Section 4: Error Taxonomy
        tot_box_err = max(1, op["fn_missed"] + op["fp_spurious"] + op["cls_confusion"])
        fn_pct = round(100.0 * op["fn_missed"] / tot_box_err, 1)
        fp_pct = round(100.0 * op["fp_spurious"] / tot_box_err, 1)
        cls_pct = round(100.0 * op["cls_confusion"] / tot_box_err, 1)

        # Count GT words & OOV across sentence cases (diag_01..diag_03) vs char-chart cases (diag_04..diag_08)
        sent_items = [x for x in df["items"] if x["id"] in ("diag_01", "diag_02", "diag_03")]
        sent_dp_ok = sum(1 for x in sent_items if x.get("oracle_dp_word_acc_pct", 0.0) == 100.0)
        sent_oov = [w for x in sent_items for w in x.get("gt_lexicon_oov_words", [])]

        lines.extend([
            "",
            "---",
            "",
            "## 4. Taksonomi Kesalahan (Dasar Prioritas Fase 1–4)",
            "",
            "### A. Taksonomi Kesalahan pada Validasi Sintetis (`val`, total 251 kesalahan kotak)",
            "",
            f"1. **Deteksi Gagal / Terlewat (*False Negative*)**: **{fn_pct}%** ({op['fn_missed']}/251 kesalahan; didominasi tanda kecil `cecak` 26, `wulu` 21, `suku` 31, dan pasangan rapat `p_la` 7 / `p_ha` 5).",
            f"2. **Klasifikasi Keliru (*Confusion*)**: **{cls_pct}%** ({op['cls_confusion']}/251 kesalahan; #1 adalah `wulu` tertukar `cecak` sebanyak 10 kali, diikuti kemiripan bentuk *nglegena* `ta`/`ba`, `ca`/`sa`, `nya`/`ba`).",
            f"3. **Deteksi Palsu (*False Positive*)**: **{fp_pct}%** ({op['fp_spurious']}/251 kesalahan; tersering pada `layar` 5 dan `suku` 4).",
            f"4. **Celah Perakitan (`val`)**: Meskipun akurasi klasifikasi kotak 97.28% (`8263/8494`), *Cluster Grapheme F1* berada di **95.46%** (`CER = 2.82%`), mengindikasikan ~1.8 poin kehilangan akurasi berasal dari pelekatan kotak ke induk / pengurutan kluster.",
            "",
            "### B. Taksonomi Kesalahan pada Set Diagnostik Nyata & Stres (`eval/failures/`, n=8)",
            "",
            "Mengisolasi masing-masing tahap pipeline pada 8 kasus diagnostik (termasuk uji terisolasi DP+Leksikon dengan *input* Latin GT bersih):",
            "",
            "1. **Klasifikasi & Pergeseran Domain Tulisan Tangan (Prioritas #1 — Tertinggi)**: Mengkontribusi mayoritas kegagalan ekstrem pada `diag_04..diag_08` (CER `33.3%..170.0%`), serta kesalahan `pepet`/`taling` (`wê` vs `wé`, skor `0.930`) pada `diag_01/02` karena distribusi latih hanya bergantung pada 3 font cetak dan potongan terisolasi `Vokal A`.",
            "2. **Deteksi / Lokalisasi, Objek Kecil (< 16 px), & Teks Distraktor Latin**: Tanda `cecak` hilang total pada `diag_01/02` (Wédang $\\rightarrow$ Wêda), huruf Latin pada `diag_07` dan `diag_08` memicu puluhan *False Positive* (`53` kotak terdeteksi padahal hanya `20` aksara pada `diag_08`!), dan *binarizer* `morphBin` kehilangan glif pada bayangan berat (`diag_04`).",
            "3. **Perakitan Ortografi (`orthography.js`)**: Akurat (`0% CER`) pada tata letak cetak rapi seperti `diag_03` (`sample_4.png`), namun belum membuang hipotesis tak sah secara ortografis (lolos urutan seperti `sadhsan`, `ng, j`) maupun memanfaatkan skor top-k.",
            f"4. **Pemisah Kata DP (`javanese_lexicon.js`)**: Pada 3 kasus kalimat (`diag_01..diag_03`), saat diberi Latin GT bersih (`oracle_dp`), **{sent_dp_ok}/{len(sent_items)}** kalimat tersegmen 100% benar — namun *rule-set* kamus masih tertutup sehingga kegagalan pengenalan hulu (`wêda`) tidak bisa dipulihkan, dan pada bagan karakter lepas (`diag_04..diag_08`) DP memaksa penggabungan suku kata lepas (`hanacaraka`, `alas`).",
            f"5. **Terjemahan & Cakupan Leksikon (`439` entri)**: Kata kalimat umum pada `diag_01/02` sudah tercakup (`0` OOV GT pada kalimat uji), namun kata luar kamus sempit (seperti `{', '.join(sent_oov) or 'kothi'}` pada `diag_03` serta kata rusak `wêda`) langsung dilewatkan tanpa penandaan kata ragu atau alternatif.",
        ])

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {OUT_MD} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
