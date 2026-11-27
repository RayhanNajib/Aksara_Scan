"""
Penghasil Laporan Statistik Dataset v4 (`docs/DATASET_V4.md`) — Dihitung Langsung dari Manifes v4.
Mencakup distribusi per sumber, per split, per kelas Tier 1 (55 aktif) vs Tier 2/3 (ditunda),
jumlah penulis/cluster unik, serta tautan ke audit kebocoran & contact sheet visual.
"""
import glob
import json
import os
from collections import Counter, defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_DIR = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4")
LEAK_JSON = os.path.join(ROOT_DIR, "dataset", "external", "reports", "LEAKAGE_AUDIT_V4.json")
HELDOUT_MANIFEST = os.path.join(ROOT_DIR, "eval", "heldout_real", "manifest.jsonl")
OUT_DOC = os.path.join(ROOT_DIR, "docs", "DATASET_V4.md")
OUT_JSON = os.path.join(ROOT_DIR, "dataset", "external", "reports", "DATASET_V4_STATS.json")

import sys
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
from schema_v4 import CLASSES_55


def main():
    per_source = {}
    class_boxes_real = Counter()
    class_writers_real = defaultdict(set)
    class_boxes_synth = Counter()

    for mf in sorted(glob.glob(os.path.join(MANIFEST_DIR, "*.jsonl"))):
        name = os.path.splitext(os.path.basename(mf))[0]
        rows = [json.loads(l) for l in open(mf, "r", encoding="utf-8") if l.strip()]
        sp_c = Counter(r.get("split", "unknown") for r in rows)
        writers = {r.get("writer_id") for r in rows if r.get("writer_id")}
        clusters = {r.get("cluster_id") for r in rows if r.get("cluster_id")}
        box_cnt = 0
        for r in rows:
            sp = r.get("split", "")
            for b in r.get("boxes", []):
                box_cnt += 1
                cname = b["class"]
                if "synth" in name:
                    if sp in ("train", "val"):
                        class_boxes_synth[cname] += 1
                elif sp in ("train", "val"):
                    class_boxes_real[cname] += 1
                    if r.get("writer_id"):
                        class_writers_real[cname].add(r["writer_id"])
        per_source[name] = {
            "rows": len(rows),
            "splits": dict(sp_c),
            "boxes": box_cnt,
            "unique_writers": len(writers),
            "unique_clusters": len(clusters),
            "license": rows[0].get("license", "-") if rows else "-",
        }

    # OOV / Tier 2 & Tier 3 check pada setilanaji_lines_v4 (tanpa kotak atomik manusia)
    oov_lines_freq = Counter()
    seti_p = os.path.join(MANIFEST_DIR, "setilanaji_lines_v4.jsonl")
    if os.path.exists(seti_p):
        covered = set(
            "\uA9B2\uA9A4\uA995\uA9AB\uA98F\uA9A2\uA9A0\uA9B1\uA9AE\uA9AD"
            "\uA9A5\uA99D\uA997\uA9AA\uA99A\uA9A9\uA992\uA9A7\uA99B\uA994"
            "\uA9B6\uA9B8\uA9BC\uA9BA\uA9B4\uA982\uA981\uA983\uA9C0\uA9BF\uA9BD\uA9BE"
            "\uA9CB\uA9C8\uA9C9"
        )
        for ln in open(seti_p, "r", encoding="utf-8"):
            r = json.loads(ln)
            for ch in r.get("unicode", ""):
                if 0xA980 <= ord(ch) <= 0xA9DF and ch not in covered:
                    oov_lines_freq[f"{ch} (U+{ord(ch):04X})"] += 1

    heldout_rows = [json.loads(l) for l in open(HELDOUT_MANIFEST, "r", encoding="utf-8") if l.strip()] if os.path.exists(HELDOUT_MANIFEST) else []
    heldout_sub = Counter(r.get("subset") for r in heldout_rows)
    leak = json.load(open(LEAK_JSON, "r", encoding="utf-8")) if os.path.exists(LEAK_JSON) else {}

    stats = {
        "sources": per_source,
        "tier1_55_classes": {
            c: {
                "real_boxes_train_val": class_boxes_real[c],
                "real_unique_writers": len(class_writers_real[c]),
                "synth_v4_boxes": class_boxes_synth[c],
            }
            for c in CLASSES_55
        },
        "tier2_3_oov_chars_in_setilanaji_lines": dict(oov_lines_freq),
        "heldout_real": {"total": len(heldout_rows), "subsets": dict(heldout_sub)},
        "leakage_audit": {k: leak.get(k) for k in ("status", "total_records_checked", "writer_overlap_count", "cluster_overlap_count", "sha256_overlap_count", "near_duplicate_cross_split_count")},
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    lines = [
        "# Statistik & Dokumentasi Dataset AksaraScan v4.0 (`docs/DATASET_V4.md`)",
        "",
        "Dihasilkan otomatis oleh `tools/datasets/summarize_dataset_v4.py` dari manifes v4 terverifikasi.",
        "",
        "## 1. Status Audit Kebocoran Data (`tools/check_leakage.py`)",
        f"- **Status:** **`{stats['leakage_audit'].get('status', 'N/A')}`**",
        f"- **Total rekaman diperiksa:** `{stats['leakage_audit'].get('total_records_checked', 0)}`",
        f"- **Overlap `writer_id`:** `{stats['leakage_audit'].get('writer_overlap_count', 0)}`",
        f"- **Overlap `cluster_id`:** `{stats['leakage_audit'].get('cluster_overlap_count', 0)}`",
        f"- **Overlap `SHA-256`:** `{stats['leakage_audit'].get('sha256_overlap_count', 0)}`",
        f"- **Duplikat dekat lintas-split (`dHash+pHash+cos>=0.85`):** `{stats['leakage_audit'].get('near_duplicate_cross_split_count', 0)}`",
        "",
        "## 2. Ringkasan Sumber Data & Split v4 (`manifests_v4/`)",
        "",
        "| Manifes / Sumber | Lisensi | Rekaman | Kotak YOLO | Penulis Unik | Cluster Unik | Rincian Split |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for name, s in per_source.items():
        sp_txt = ", ".join(f"{k}={v}" for k, v in s["splits"].items())
        lines.append(f"| `{name}` | {s['license']} | {s['rows']} | {s['boxes']} | {s['unique_writers']} | {s['unique_clusters']} | `{sp_txt}` |")

    lines += [
        "",
        "## 3. Set Uji Nyata Terkunci (`eval/heldout_real/`)",
        f"- **Total citra:** `{len(heldout_rows)}` (dikunci oleh `eval/heldout_real/SHA256SUMS.lock`)",
    ]
    for k, v in heldout_sub.items():
        lines.append(f"- `{k}`: `{v}` citra")

    lines += [
        "",
        "## 4. Histogram Kotak Kelas Tier 1 (55 Kelas Aktif)",
        "",
        "| ID | Kelas | Kotak Nyata (train+val) | Penulis Nyata | Kotak Sintetis v4 | Total Aktif |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for idx, c in enumerate(CLASSES_55):
        rb = class_boxes_real[c]
        rw = len(class_writers_real[c])
        sb = class_boxes_synth[c]
        lines.append(f"| {idx} | `{c}` | {rb} | {rw} | {sb} | {rb + sb} |")

    lines += [
        "",
        "## 5. Kebijakan Kelas Tier 2 & Tier 3 (Ditunda / *Deferred*)",
        "Sesuai aturan skema `dataset/schema/classes_v4.yaml`, kelas Tier 2/3 hanya diaktifkan bila memiliki `>= 100` kotak nyata teranotasi dari `>= 5` penulis/cluster independen.",
        "- Sumber bebas-lisensi saat ini hanya memiliki catatan baris tanpa kotak karakter di `setilanaji`: "
        + (", ".join(f"`{k}`: `{v}`" for k, v in oov_lines_freq.items()) or "tidak ada"),
        "- Dataset Murda `RayFox52` ditandai **`DIBLOKIR`** (tanpa lisensi), dan ekspor Roboflow/Kaggle menunggu kunci API/unduhan manual. Maka **seluruh kelas Tier 2/3 tetap berstatus `deferred`** pada Fase 1 agar tidak melatih kelas hantu.",
        "",
        "## 6. Contact Sheet Audit Visual (`dataset/external/reports/`)",
        "- `v4_contact_hanacaraka.jpg`",
        "- `v4_contact_jawa_zip.jpg` & `v4_contact_syllables.jpg`",
        "- `v4_contact_setilanaji.jpg` & `v4_contact_nusaaksara_eval.jpg`",
        "- `v4_contact_synth_tps.jpg` (*TPS Elastic Handwriting*)",
        "- `v4_contact_synth_multiink_wall.jpg` (*Pulpen Multi-Warna, Garis Tembok/Buku, Dark Mode*)",
        "- `v4_contact_synth_hardneg_latin.jpg` (*Teks Latin Campuran & Hard Negatives*)",
        "- `v4_contact_heldout_real.jpg` (*42 Citra Evaluasi Terkunci*)",
    ]

    with open(OUT_DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[DONE] {OUT_DOC} + {OUT_JSON} berhasil dibuat.")


if __name__ == "__main__":
    main()
