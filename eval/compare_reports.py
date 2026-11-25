"""
Pembanding Evaluasi AksaraScan Baseline v2.5 vs v3.0 (Fase 5a-5d):
  - Menghitung interval kepercayaan Bootstrap 95% (B = 1000 resamples) pada CER, Grapheme F1, & Akurasi Kata
  - Membandingkan detektor Track A + Beam Decoder (v3.0) terhadap baseline v2.5 per kondisi (`diag_01..diag_08`)
  - Mengevaluasi Track B CRNN-CTC ONNX (`runs_crnn/aksara_crnn_ctc.onnx`) pada baris tunggal & menghasilkan tabel perbandingan
  - Menulis laporan terukur ke `eval/reports/v3.0.md` dan `eval/reports/v3.0_comparison.json`.
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
from run_eval import compute_cer, split_javanese_graphemes


def bootstrap_ci(values, n_boot=1000, seed=42):
    arr = np.asarray(list(values), dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    return {
        "mean": round(float(np.mean(arr)), 4),
        "ci_low": round(float(np.percentile(means, 2.5)), 4),
        "ci_high": round(float(np.percentile(means, 97.5)), 4),
        "n": int(n),
    }


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def eval_track_b_on_lines(crnn_onnx, vocab_json, manifest_items):
    if not os.path.exists(crnn_onnx) or not os.path.exists(vocab_json):
        return None
    vocab = json.load(open(vocab_json, "r", encoding="utf-8"))["chars"]
    sess = ort.InferenceSession(crnn_onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    rows = []
    for it in manifest_items:
        ip = os.path.join(ROOT_DIR, it["image"]) if not os.path.isabs(it["image"]) else it["image"]
        if not os.path.exists(ip):
            continue
        gray = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape[:2]
        sc = 64.0 / max(1, h)
        nw = min(768, max(16, int(round(w * sc))))
        resized = cv2.resize(gray, (nw, 64), interpolation=cv2.INTER_AREA)
        canvas = np.full((64, 768), 245, dtype=np.uint8)
        canvas[:, :nw] = resized
        norm = ((255.0 - canvas.astype(np.float32)) / 255.0)[None, None, :, :]
        logits = sess.run(None, {inp_name: norm})[0]  # [T, 1, V]
        ids = np.argmax(logits[:, 0, :], axis=-1)
        chars = []
        prev = 0
        for cid in ids:
            if cid != 0 and cid != prev and (cid - 1) < len(vocab):
                chars.append(vocab[cid - 1])
            prev = int(cid)
        pred_u = "".join(chars)
        cer = compute_cer(it["unicode_gt"], pred_u)
        rows.append({"id": it["id"], "unicode_gt": it["unicode_gt"], "pred_track_b": pred_u, "cer": round(cer, 4)})
    return rows


def git_commit_short():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="eval/reports/baseline_v2.5.json")
    p.add_argument("--candidate", default="eval/reports/v3_phase4_browser/baseline_v2.5.json")
    p.add_argument("--crnn-onnx", default="runs_crnn/aksara_crnn_ctc.onnx")
    p.add_argument("--crnn-vocab", default="runs_crnn/crnn_vocab.json")
    p.add_argument("--out-md", default="eval/reports/v3.0.md")
    p.add_argument("--out-json", default="eval/reports/v3.0_comparison.json")
    args = p.parse_args()

    b_path = os.path.join(ROOT_DIR, args.baseline) if not os.path.isabs(args.baseline) else args.baseline
    c_path = os.path.join(ROOT_DIR, args.candidate) if not os.path.isabs(args.candidate) else args.candidate
    base = load_json(b_path)
    cand = load_json(c_path)
    if not base or not cand:
        raise FileNotFoundError(f"Laporan JSON tidak lengkap: base={b_path}, cand={c_path}")

    base_items = base.get("diagnostic_failures", base.get("failures", {})).get("items", [])
    cand_items = cand.get("diagnostic_failures", cand.get("failures", {})).get("items", [])

    # Bootstrap CI (satuan persen -> 0..1 agar konsisten)
    b_cers = [it.get("cer_pct", it.get("unicode_cer", 0) * 100.0) / 100.0 for it in base_items]
    c_cers = [it.get("cer_pct", it.get("unicode_cer", 0) * 100.0) / 100.0 for it in cand_items]
    b_gf1 = [it.get("grapheme_f1_pct", it.get("grapheme_f1", 0) * 100.0) / 100.0 for it in base_items]
    c_gf1 = [it.get("grapheme_f1_pct", it.get("grapheme_f1", 0) * 100.0) / 100.0 for it in cand_items]
    b_wacc = [it.get("word_acc_pct", it.get("word_acc", 0) * 100.0) / 100.0 for it in base_items]
    c_wacc = [it.get("word_acc_pct", it.get("word_acc", 0) * 100.0) / 100.0 for it in cand_items]

    summary = {
        "commit": git_commit_short(),
        "baseline_cer": bootstrap_ci(b_cers),
        "v3_cer": bootstrap_ci(c_cers),
        "baseline_grapheme_f1": bootstrap_ci(b_gf1),
        "v3_grapheme_f1": bootstrap_ci(c_gf1),
        "baseline_word_acc": bootstrap_ci(b_wacc),
        "v3_word_acc": bootstrap_ci(c_wacc),
        "parity_v3": cand.get("parity_20", cand.get("parity", {})),
    }

    # Cek file Track B jika ada
    crnn_path = os.path.join(ROOT_DIR, args.crnn_onnx)
    vocab_path = os.path.join(ROOT_DIR, args.crnn_vocab)
    if not os.path.exists(crnn_path):
        # Cek lokasi alternatif runs_crnn jika ada
        alt = os.path.join(ROOT_DIR, "Aksara_Scan", "runs_crnn", "aksara_crnn_ctc.onnx")
        alt_v = os.path.join(ROOT_DIR, "Aksara_Scan", "runs_crnn", "crnn_vocab.json")
        if os.path.exists(alt):
            crnn_path, vocab_path = alt, alt_v

    diag_manifest = []
    diag_p = os.path.join(ROOT_DIR, "eval", "failures", "manifest.jsonl")
    if os.path.exists(diag_p):
        diag_manifest = [json.loads(l) for l in open(diag_p, "r", encoding="utf-8") if l.strip()]
    track_b_rows = eval_track_b_on_lines(crnn_path, vocab_path, diag_manifest[:3]) or []
    summary["track_b_diag_preview"] = track_b_rows

    out_json_abs = os.path.join(ROOT_DIR, args.out_json) if not os.path.isabs(args.out_json) else args.out_json
    out_md_abs = os.path.join(ROOT_DIR, args.out_md) if not os.path.isabs(args.out_md) else args.out_md
    os.makedirs(os.path.dirname(out_json_abs), exist_ok=True)
    with open(out_json_abs, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    base_map = {it["id"]: it for it in base_items}
    lines = [
        "# Laporan Evaluasi AksaraScan v3.0 vs Baseline v2.5",
        "",
        f"- **Commit Referensi**: `{summary['commit']}`",
        "- **Cakupan Uji**: Set Diagnostik Terkurasi (`eval/failures/manifest.jsonl`, n=8), Golden Parity Set (`eval/parity_20.jsonl`, n=20), dan Track B CRNN-CTC (prototype CPU).",
        "- **Catatan Jujur**: Evaluasi detektor YOLOX memakai bobot `aksara_yolo.onnx` + jalur penalaran Fase 4 penuh (validator ortografi + geometri relatif + Dekoder Beam + LM 866k kata + kamus 4.500 entri + afiks Jawa), sementara pelatihan GPU penuh `yolox-tiny`/`yolox-s` 960px siap dijalankan via `training/COLAB_GPU_RUNBOOK.md`.",
        "",
        "## 1. Ringkasan Metrik Set Diagnostik (Bootstrap 95% CI, B=1000)",
        "",
        "| Metrik | Baseline v2.5 (95% CI) | AksaraScan v3.0 (95% CI) | Delta |",
        "|---|---|---|---|",
        f"| **Unicode CER** (lebih rendah lebih baik) | `{summary['baseline_cer']['mean']*100:.2f}%` (`{summary['baseline_cer']['ci_low']*100:.1f}%–{summary['baseline_cer']['ci_high']*100:.1f}%`) | `{summary['v3_cer']['mean']*100:.2f}%` (`{summary['v3_cer']['ci_low']*100:.1f}%–{summary['v3_cer']['ci_high']*100:.1f}%`) | `{(summary['v3_cer']['mean']-summary['baseline_cer']['mean'])*100:+.2f} pp` |",
        f"| **Grapheme F1** (lebih tinggi lebih baik) | `{summary['baseline_grapheme_f1']['mean']*100:.2f}%` (`{summary['baseline_grapheme_f1']['ci_low']*100:.1f}%–{summary['baseline_grapheme_f1']['ci_high']*100:.1f}%`) | `{summary['v3_grapheme_f1']['mean']*100:.2f}%` (`{summary['v3_grapheme_f1']['ci_low']*100:.1f}%–{summary['v3_grapheme_f1']['ci_high']*100:.1f}%`) | `{(summary['v3_grapheme_f1']['mean']-summary['baseline_grapheme_f1']['mean'])*100:+.2f} pp` |",
        f"| **Akurasi Kata Latin** (lebih tinggi lebih baik) | `{summary['baseline_word_acc']['mean']*100:.2f}%` (`{summary['baseline_word_acc']['ci_low']*100:.1f}%–{summary['baseline_word_acc']['ci_high']*100:.1f}%`) | `{summary['v3_word_acc']['mean']*100:.2f}%` (`{summary['v3_word_acc']['ci_low']*100:.1f}%–{summary['v3_word_acc']['ci_high']*100:.1f}%`) | `{(summary['v3_word_acc']['mean']-summary['baseline_word_acc']['mean'])*100:+.2f} pp` |",
        "",
        "## 2. Rincian Per Kasus Diagnostik (`diag_01` s.d. `diag_08`)",
        "",
        "| ID | Kondisi Domain | CER v2.5 | CER v3.0 | Kata Latin v3.0 (Browser) |",
        "|---|---|---:|---:|---|",
    ]
    for it in cand_items:
        b = base_map.get(it["id"], {})
        b_cer_v = b.get("cer_pct", b.get("unicode_cer", 0) * 100.0) if b else None
        b_cer = f"{b_cer_v:.2f}%" if b_cer_v is not None else "-"
        c_cer_v = it.get("cer_pct", it.get("unicode_cer", 0) * 100.0)
        c_cer = f"{c_cer_v:.2f}%"
        lat_p = (it.get("latin_pred") or it.get("pred_latin") or "").replace("|", "/").replace("\n", " ")[:90]
        lines.append(f"| `{it['id']}` | {it.get('domain','-')} | {b_cer} | **{c_cer}** | `{lat_p}` |")

    p_sum = summary["parity_v3"]
    if p_sum:
        n_par = p_sum.get("n_images", 20)
        lines.extend([
            "",
            "## 3. Verifikasi Paritas & Nol Kebocoran Jaringan Browser",
            "",
            f"- **Golden Set (n={n_par})**: `{p_sum.get('browser_e2e_unicode_exact_matches', 'N/A')}` Unicode identik terhadap Python, `{p_sum.get('browser_e2e_box_agreement_iou070', 'N/A')}` kotak selaras (IoU >= 0.70).",
            f"- **Permintaan Jaringan Eksternal Saat Inferensi**: `{len(p_sum.get('browser_external_requests_during_inference', []))}` (0 = 100% lokal di peramban).",
            f"- **Console / Page Error Browser**: `{len(p_sum.get('browser_console_errors', []))}`.",
        ])

    if track_b_rows:
        lines.extend([
            "",
            "## 4. Pratinjau Jalur Penantang Track B (CRNN-CTC Prototype CPU, 4 Epoch)",
            "",
            "| ID | CER Track B (Prototype 4 Epoch CPU) | Transkripsi Mentah Track B |",
            "|---|---:|---|",
        ])
        for r in track_b_rows:
            lines.append(f"| `{r['id']}` | `{r['cer']*100:.1f}%` | `{r['pred_track_b']}` |")

    with open(out_md_abs, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Compare] Tersimpan laporan v3.0 -> {out_md_abs}")


if __name__ == "__main__":
    main()
