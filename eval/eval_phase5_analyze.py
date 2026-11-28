"""
Analisis Benchmark, Matriks Ablasi, Matriks Konfusi, Taksonomi Kesalahan,
& Galeri 20 Kegagalan Terburuk — AksaraScan Fase 5 (`v4/fase-5-eval`)
-------------------------------------------------------------------------
Membaca hasil browser mentah (`eval/artifacts/phase5_browser_raw.json`) serta laporan
ablasi Python (`eval/reports/phase2_ablation.json`), memverifikasi kunci SHA-256
`eval/heldout_real/`, lalu menghasilkan:
  1. `eval/reports/phase5_benchmark.json` — angka terukur + bootstrap 95% CI (B=2000, seed=42)
  2. `eval/reports/phase5_worst20.html` + `eval/reports/phase5_worst20_imgs/` — 20 kasus
     kegagalan terburuk (detektor v3.1 vs v4.0 + kotak prediksi & ground-truth teks)

Penggunaan:
  python -u eval/eval_phase5_analyze.py
"""
import hashlib
import html
import json
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
import run_eval as re_base

RAW_PATH = os.path.join(ROOT_DIR, "eval", "artifacts", "phase5_browser_raw.json")
P2_ABLATION_PATH = os.path.join(ROOT_DIR, "eval", "reports", "phase2_ablation.json")
META_PATH = os.path.join(ROOT_DIR, "app", "web_camscanner", "yolo_meta.json")
OUT_JSON_PATH = os.path.join(ROOT_DIR, "eval", "reports", "phase5_benchmark.json")
WORST_HTML_PATH = os.path.join(ROOT_DIR, "eval", "reports", "phase5_worst20.html")
WORST_IMG_DIR = os.path.join(ROOT_DIR, "eval", "reports", "phase5_worst20_imgs")

PRONE_PAIRS = [("da", "sa"), ("ka", "na"), ("tha", "ca"), ("ha", "ra")]
SANDHANGAN_CP = set("\uA980\uA981\uA982\uA983\uA9B3\uA9B4\uA9B5\uA9B6\uA9B7\uA9B8\uA9B9\uA9BA\uA9BB\uA9BC\uA9BD\uA9BE\uA9BF")


def verify_heldout_lock():
    lock_path = os.path.join(ROOT_DIR, "eval", "heldout_real", "SHA256SUMS.lock")
    if not os.path.exists(lock_path):
        raise RuntimeError(f"Lock file tidak ditemukan: {lock_path}")
    checked = 0
    with open(lock_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_hash, rel_path = line.split(None, 1)
            fp = os.path.join(ROOT_DIR, "eval", "heldout_real", rel_path.strip())
            if not os.path.exists(fp):
                raise RuntimeError(f"Missing locked file: {rel_path}")
            h = hashlib.sha256()
            with open(fp, "rb") as bf:
                for chunk in iter(lambda: bf.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected_hash.lower():
                raise RuntimeError(f"Checksum mismatch: {rel_path}")
            checked += 1
    print(f"[HELDOUT LOCK OK] verified {checked} locked files, mismatches=0")
    return checked


def fold_diacritics(s):
    if not s or s == "—":
        return ""
    return (
        s.replace("é", "e").replace("è", "e").replace("ê", "e")
         .replace("É", "e").replace("È", "e").replace("Ê", "e")
    )


def compute_word_acc_folded(gt_lat, pred_lat):
    if not gt_lat or gt_lat == "—":
        return {"word_acc": None, "folded_word_acc": None}
    wa_strict = re_base.compute_word_accuracy(gt_lat, pred_lat)["word_acc"] * 100.0
    wa_folded = re_base.compute_word_accuracy(fold_diacritics(gt_lat), fold_diacritics(pred_lat))["word_acc"] * 100.0
    return {"word_acc": wa_strict, "folded_word_acc": wa_folded}


def bootstrap_ci(vals, n_boot=2000, seed=42):
    arr = np.array([v for v in vals if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return {"mean": round(mean, 2), "ci_lo": round(mean, 2), "ci_hi": round(mean, 2), "n": 1}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(arr), size=(n_boot, len(arr)))
    boots = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": round(mean, 2), "ci_lo": round(float(lo), 2), "ci_hi": round(float(hi), 2), "n": len(arr)}


def latency_stats(ms_list):
    arr = np.array(ms_list, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(float(np.mean(arr)), 1),
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
    }


def load_synth2_extra():
    path = os.path.join(ROOT_DIR, "dataset", "external", "synth_v2", "manifest.jsonl")
    extra = {}
    if not os.path.exists(path):
        return extra
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != "val" or r.get("distractor"):
                continue
            lbl_rel = (r.get("label") or "").replace("\\", "/")
            boxes = []
            lbl_abs = os.path.join(ROOT_DIR, lbl_rel)
            if lbl_rel and os.path.exists(lbl_abs):
                with open(lbl_abs, "r", encoding="utf-8") as lf:
                    for bl in lf:
                        parts = bl.strip().split()
                        if len(parts) == 5:
                            boxes.append({
                                "cid": int(parts[0]),
                                "bbox": [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])],
                            })
            extra[r["id"]] = {"latin_gt": r.get("latin_gt", ""), "boxes": boxes}
    return extra


def levenshtein_ops(seq_a, seq_b):
    """Kembalikan jumlah (substitusi, delesi, insersi) via DP Backtrack deterministik."""
    la, lb = len(seq_a), len(seq_b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    i, j = la, lb
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    sub += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return sub, dele, ins


def summarize_mode_datasets(items_meta, mode_payload, synth2_extra, lex_keys):
    meta_by_id = {m["id"]: m for m in items_meta}
    per_item = {}
    for res in mode_payload["results"]:
        mid = res["id"]
        gt = meta_by_id[mid]
        gt_u = gt.get("unicode_gt", "")
        pred_u = "" if res["unicodeText"] == "—" else res["unicodeText"]
        pred_lat = "" if res["sentenceLatin"] == "—" else res["sentenceLatin"]

        # Pilih Latin GT terbaik: manual (`failures`), manifest (`synth_v2`), atau kanonik (`heldout_real`)
        if gt.get("latin_gt"):
            gt_lat = gt["latin_gt"]
            gt_lat_source = "human_manifest"
        elif mid in synth2_extra and synth2_extra[mid].get("latin_gt"):
            gt_lat = synth2_extra[mid]["latin_gt"]
            gt_lat_source = "synth_manifest"
        else:
            gt_lat = gt.get("canonical_latin_gt", "")
            gt_lat_source = "unicode_canonical_decode"

        cer_pct = re_base.compute_cer(gt_u, pred_u) * 100.0 if gt_u else (0.0 if not pred_u else 100.0)
        gf1_pct = re_base.compute_grapheme_f1(gt_u, pred_u)["f1"] * 100.0 if gt_u else (100.0 if not pred_u else 0.0)
        wa_res = compute_word_acc_folded(gt_lat, pred_lat)

        # Taksonomi kesalahan tingkat karakter & kata
        gt_cl = re_base.split_javanese_graphemes(gt_u)
        pr_cl = re_base.split_javanese_graphemes(pred_u)
        sub_g, del_g, ins_g = levenshtein_ops(gt_cl, pr_cl)

        gt_sand = sum(1 for ch in gt_u if ch in SANDHANGAN_CP)
        pr_sand = sum(1 for ch in pred_u if ch in SANDHANGAN_CP)
        missing_sand = max(0, gt_sand - pr_sand)

        pred_words = re_base.normalize_latin_words(pred_lat)
        oov_pred_words = [w for w in pred_words if fold_diacritics(w) not in lex_keys]
        is_oov_gap = bool(
            gt_lat and cer_pct <= 15.0 and (wa_res["folded_word_acc"] is not None) and wa_res["folded_word_acc"] < 85.0
        )

        per_item[mid] = {
            "id": mid,
            "dataset": res["dataset"],
            "subset": res["subset"],
            "condition": res["condition"],
            "image": gt["image"],
            "cer_pct": round(cer_pct, 2),
            "grapheme_f1_pct": round(gf1_pct, 2),
            "word_acc_pct": round(wa_res["word_acc"], 2) if wa_res["word_acc"] is not None else None,
            "word_acc_folded_pct": round(wa_res["folded_word_acc"], 2) if wa_res["folded_word_acc"] is not None else None,
            "gt_lat_source": gt_lat_source,
            "elapsed_ms": res["elapsedMs"],
            "n_boxes": len(res["detections"]),
            "uncertain_boxes": sum(1 for d in res["detections"] if d.get("uncertain")),
            "gt_u": gt_u,
            "pred_u": res["unicodeText"],
            "gt_lat": gt_lat,
            "pred_lat": res["sentenceLatin"],
            "tax_sub_grapheme": sub_g,
            "tax_del_missed": del_g,
            "tax_ins_spurious": ins_g,
            "tax_missing_sandhangan": missing_sand,
            "tax_oov_words": len(oov_pred_words),
            "tax_oov_gap_item": is_oov_gap,
            "detections": res["detections"],
        }

    def agg(rows):
        return {
            "n": len(rows),
            "cer_pct": bootstrap_ci([r["cer_pct"] for r in rows]),
            "grapheme_f1_pct": bootstrap_ci([r["grapheme_f1_pct"] for r in rows]),
            "word_acc_pct": bootstrap_ci([r["word_acc_pct"] for r in rows if r["word_acc_pct"] is not None]),
            "word_acc_folded_pct": bootstrap_ci([r["word_acc_folded_pct"] for r in rows if r["word_acc_folded_pct"] is not None]),
            "latency_ms": latency_stats([r["elapsed_ms"] for r in rows]),
            "mean_boxes": round(float(np.mean([r["n_boxes"] for r in rows])), 2) if rows else 0.0,
        }

    hr_rows = [r for r in per_item.values() if r["dataset"] == "heldout_real"]
    fl_rows = [r for r in per_item.values() if r["dataset"] == "failures"]
    s2_rows = [r for r in per_item.values() if r["dataset"] == "clean_synth_v2_val"]
    neg_rows = [r for r in per_item.values() if r["dataset"] == "distractors_v4_val"]

    blur_conds = {"motion_blur", "defocus_blur", "latin_distractor_blur"}
    blur_hr = [r for r in hr_rows if r["condition"] in blur_conds]
    challenged_hr = [r for r in hr_rows if r["subset"] == "hanacaraka_12_unseen_human_writers" and r["condition"] != "sharp_clean"]

    by_subset_hr = {}
    for sub in sorted(set(r["subset"] for r in hr_rows)):
        by_subset_hr[sub] = agg([r for r in hr_rows if r["subset"] == sub])

    by_cond_hr = {}
    for cond in sorted(set(r["condition"] for r in hr_rows if r["condition"])):
        by_cond_hr[cond] = agg([r for r in hr_rows if r["condition"] == cond])

    real50 = hr_rows + fl_rows
    taxonomy = {
        "n_items": len(real50),
        "total_gt_graphemes": int(sum(len(re_base.split_javanese_graphemes(r["gt_u"])) for r in real50)),
        "grapheme_substitutions": int(sum(r["tax_sub_grapheme"] for r in real50)),
        "grapheme_missed_deletions": int(sum(r["tax_del_missed"] for r in real50)),
        "grapheme_spurious_insertions": int(sum(r["tax_ins_spurious"] for r in real50)),
        "missing_sandhangan_marks": int(sum(r["tax_missing_sandhangan"] for r in real50)),
        "oov_predicted_words": int(sum(r["tax_oov_words"] for r in real50)),
        "oov_segmentation_gap_items": int(sum(1 for r in real50 if r["tax_oov_gap_item"])),
    }

    summary = {
        "init_ready_ms": mode_payload["initReadyMs"],
        "console_errors": len(mode_payload["consoleErrors"]),
        "external_inf_requests": len(mode_payload["externalRequestsDuringInference"]),
        "heldout_real_n42": {
            **agg(hr_rows),
            "by_subset": by_subset_hr,
            "by_condition": by_cond_hr,
            "blur_real_n5": agg(blur_hr),
            "hanacaraka_challenged_n10": agg(challenged_hr),
        },
        "failures_n8": agg(fl_rows),
        "clean_synth_v2_val_n40": agg(s2_rows),
        "distractors_v4_val_n29": {
            "n": len(neg_rows),
            "fp_boxes_per_img": round(float(np.mean([r["n_boxes"] for r in neg_rows])), 2) if neg_rows else 0.0,
            "clean_zero_box_rate_pct": round(100.0 * sum(1 for r in neg_rows if r["n_boxes"] == 0) / max(1, len(neg_rows)), 1),
            "total_fp_boxes": int(sum(r["n_boxes"] for r in neg_rows)),
            "latency_ms": latency_stats([r["elapsed_ms"] for r in neg_rows]),
        },
        "failure_taxonomy_real_n50": taxonomy,
    }
    return summary, per_item


def analyze_box_confusion(items_meta, mode_items, synth2_extra, yolo_meta, label_name):
    classes = yolo_meta["classes"]
    cls_to_id = {c: i for i, c in enumerate(classes)}
    nc = len(classes)
    conf = np.zeros((nc + 1, nc + 1), dtype=np.int64)  # terakhir = BG/FN/FP
    total_gt = 0
    matched_tp = 0
    matched_cls_ok = 0

    meta_by_id = {m["id"]: m for m in items_meta}
    boxed_ids = [
        m["id"] for m in items_meta
        if m.get("boxes") or (m["id"] in synth2_extra and synth2_extra[m["id"]]["boxes"])
    ]

    for mid in boxed_ids:
        m = meta_by_id[mid]
        gt_boxes = m.get("boxes") or synth2_extra[mid]["boxes"]
        img_abs = os.path.join(ROOT_DIR, m["image"].replace("/", os.sep))
        bgr = cv2.imread(img_abs)
        if bgr is None:
            continue
        ih, iw = bgr.shape[:2]
        sc = min(1.0, 1500.0 / max(iw, ih))

        gt_px = []
        for gb in gt_boxes:
            cx, cy, bw, bh = gb["bbox"]
            cid = int(gb["cid"])
            x0 = (cx - bw / 2.0) * iw * sc
            y0 = (cy - bh / 2.0) * ih * sc
            x1 = (cx + bw / 2.0) * iw * sc
            y1 = (cy + bh / 2.0) * ih * sc
            gt_px.append((cid, [x0, y0, x1, y1]))

        preds = mode_items[mid]["detections"]
        # Greedy bipartite matching IoU >= 0.35
        pairs = []
        for gi, (gc, gbox) in enumerate(gt_px):
            for pi, pd in enumerate(preds):
                pc = cls_to_id.get(pd["label"], -1)
                if pc < 0 or not pd.get("box"):
                    continue
                iou = re_base.box_iou(gbox, pd["box"])
                if iou >= 0.35:
                    pairs.append((iou, gi, pi))
        pairs.sort(reverse=True)
        used_g, used_p = set(), set()
        for _iou, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            gc = gt_px[gi][0]
            pc = cls_to_id[preds[pi]["label"]]
            conf[gc, pc] += 1
            matched_tp += 1
            if gc == pc:
                matched_cls_ok += 1
        for gi, (gc, _gbox) in enumerate(gt_px):
            total_gt += 1
            if gi not in used_g:
                conf[gc, nc] += 1
        for pi, pd in enumerate(preds):
            if pi not in used_p:
                pc = cls_to_id.get(pd["label"], -1)
                if pc >= 0:
                    conf[nc, pc] += 1

    confused = []
    for gc in range(nc):
        for pc in range(nc):
            if gc != pc and conf[gc, pc] > 0:
                confused.append({"gt": classes[gc], "pred": classes[pc], "count": int(conf[gc, pc])})
    confused.sort(key=lambda x: x["count"], reverse=True)

    prone = {}
    for a, b in PRONE_PAIRS:
        ia, ib = cls_to_id[a], cls_to_id[b]
        c_ab = int(conf[ia, ib])
        c_ba = int(conf[ib, ia])
        prone[f"{a}<->{b}"] = {
            f"{a}->{b}": c_ab,
            f"{b}->{a}": c_ba,
            "total": c_ab + c_ba,
            f"{a}_gt_total": int(conf[ia, :].sum()),
            f"{b}_gt_total": int(conf[ib, :].sum()),
        }

    return {
        "mode": label_name,
        "boxed_images": len(boxed_ids),
        "total_gt_boxes": total_gt,
        "matched_iou35_boxes": matched_tp,
        "matched_recall_pct": round(100.0 * matched_tp / max(1, total_gt), 2),
        "matched_cls_acc_pct": round(100.0 * matched_cls_ok / max(1, matched_tp), 2),
        "end2end_box_recall_cls_pct": round(100.0 * matched_cls_ok / max(1, total_gt), 2),
        "prone_pairs": prone,
        "top_confused_pairs": confused[:20],
        "top_missed_fn": sorted(
            [{"class": classes[c], "fn": int(conf[c, nc]), "gt": int(conf[c, :].sum())} for c in range(nc) if conf[c, nc] > 0],
            key=lambda x: x["fn"], reverse=True
        )[:15],
        "top_spurious_fp": sorted(
            [{"class": classes[c], "fp": int(conf[nc, c])} for c in range(nc) if conf[nc, c] > 0],
            key=lambda x: x["fp"], reverse=True
        )[:15],
    }


def build_worst20_gallery(per_item_v31, per_item_v4):
    os.makedirs(WORST_IMG_DIR, exist_ok=True)
    real50_ids = [
        mid for mid, r in per_item_v4.items()
        if r["dataset"] in ("heldout_real", "failures")
    ]
    ranked = sorted(
        real50_ids,
        key=lambda mid: (per_item_v4[mid]["cer_pct"], -per_item_v4[mid]["grapheme_f1_pct"]),
        reverse=True,
    )[:20]

    colors = {
        "nglegena": (67, 164, 217),
        "pasangan": (246, 141, 92),
        "sandhangan": (48, 140, 232),
        "pada": (122, 169, 63),
    }

    worst_entries = []
    cards_html = []
    for rank, mid in enumerate(ranked, 1):
        r4 = per_item_v4[mid]
        r3 = per_item_v31[mid]
        img_abs = os.path.join(ROOT_DIR, r4["image"].replace("/", os.sep))
        bgr = cv2.imread(img_abs)
        vis = bgr.copy()
        ih, iw = vis.shape[:2]
        sc = min(1.0, 1500.0 / max(iw, ih))
        if sc < 1.0:
            vis = cv2.resize(vis, (int(round(iw * sc)), int(round(ih * sc))), interpolation=cv2.INTER_AREA)

        for det in r4["detections"]:
            box = det.get("box") or []
            if len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(v) for v in box]
            col = (59, 86, 209) if det.get("uncertain") else colors.get(det.get("category"), (67, 164, 217))
            cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
            cv2.putText(vis, det["label"], (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        out_fname = f"{rank:02d}_{mid}.jpg"
        cv2.imwrite(os.path.join(WORST_IMG_DIR, out_fname), vis, [cv2.IMWRITE_JPEG_QUALITY, 84])
        rel_img = f"phase5_worst20_imgs/{out_fname}"

        delta_cer = round(r4["cer_pct"] - r3["cer_pct"], 2)
        worst_entries.append({
            "rank": rank,
            "id": mid,
            "dataset": r4["dataset"],
            "subset": r4["subset"],
            "condition": r4["condition"],
            "source_image": r4["image"],
            "annotated_image": f"eval/reports/{rel_img}",
            "cer_v31_pct": r3["cer_pct"],
            "cer_v4_pct": r4["cer_pct"],
            "delta_cer_pp": delta_cer,
            "gf1_v4_pct": r4["grapheme_f1_pct"],
            "gt_u": r4["gt_u"],
            "pred_u_v31": r3["pred_u"],
            "pred_u_v4": r4["pred_u"],
            "pred_lat_v4": r4["pred_lat"],
        })
        cards_html.append(
            f"<article class='card'>"
            f"<header><b>#{rank:02d} · {html.escape(mid)}</b> "
            f"<span class='tag'>{html.escape(r4['subset'])} · {html.escape(r4['condition'])}</span></header>"
            f"<div class='meta'>CER v3.1={r3['cer_pct']}% → v4.0={r4['cer_pct']}% (delta {delta_cer:+.2f} pp) · "
            f"GF1 v4.0={r4['grapheme_f1_pct']}% · kotak={r4['n_boxes']} ({r4['uncertain_boxes']} ragu)</div>"
            f"<div class='txt'><div><b>GT:</b> {html.escape(r4['gt_u'])}</div>"
            f"<div><b>v3.1:</b> {html.escape(r3['pred_u'])}</div>"
            f"<div><b>v4.0:</b> {html.escape(r4['pred_u'])} <i>({html.escape(r4['pred_lat'])})</i></div></div>"
            f"<img src='{html.escape(rel_img)}' alt='{html.escape(mid)}'>"
            f"</article>"
        )

    doc = (
        "<!DOCTYPE html><html lang='id'><head><meta charset='utf-8'>"
        "<title>AksaraScan v4.0 — Galeri 20 Kegagalan Terburuk (Fase 5)</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#121318;color:#edeae3;max-width:1100px;margin:24px auto;padding:0 16px}"
        ".card{background:#1b1d24;border:1px solid #303442;border-radius:10px;padding:14px;margin-bottom:16px}"
        ".tag{color:#9ca3b5;font-size:13px}.meta{font-family:monospace;font-size:12px;color:#e6bc6a;margin:6px 0}"
        ".txt{font-size:14px;line-height:1.5;margin-bottom:10px}img{max-width:100%;height:auto;border:1px solid #303442;border-radius:6px}</style>"
        "</head><body><h1>Galeri 20 Kegagalan Terburuk — AksaraScan v4.0 Browser</h1>"
        "<p>Diurutkan berdasarkan Unicode CER tertinggi dari gabungan set uji nyata <code>eval/heldout_real/</code> (n=42) dan <code>eval/failures/</code> (n=8).</p>"
        + "\n".join(cards_html)
        + "</body></html>"
    )
    with open(WORST_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[WORST-20 GALLERY] {WORST_HTML_PATH} ({len(worst_entries)} items)")
    return worst_entries


def main():
    verify_heldout_lock()
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        yolo_meta = json.load(f)
    p2_ablation = {}
    if os.path.exists(P2_ABLATION_PATH):
        with open(P2_ABLATION_PATH, "r", encoding="utf-8") as f:
            p2_ablation = json.load(f)

    # Load lexicon keys
    import subprocess
    lex_keys_json = subprocess.check_output([
        "node", "-e",
        "const fs=require('fs'),vm=require('vm'),c={window:{}};vm.createContext(c);"
        "vm.runInContext(fs.readFileSync('app/web_camscanner/javanese_lexicon.js','utf8'),c);"
        "console.log(JSON.stringify(Object.keys(c.window.AksaraLexicon.JAVANESE_DICT)));"
    ], cwd=ROOT_DIR).decode("utf-8")
    lex_keys = {fold_diacritics(k.lower()) for k in json.loads(lex_keys_json)}

    synth2_extra = load_synth2_extra()
    items_meta = raw["itemsMeta"]

    summaries = {}
    per_items = {}
    for mode_name, mode_payload in raw["modes"].items():
        s, pi = summarize_mode_datasets(items_meta, mode_payload, synth2_extra, lex_keys)
        summaries[mode_name] = s
        per_items[mode_name] = pi

    conf_v31 = analyze_box_confusion(items_meta, per_items["v3_1"], synth2_extra, yolo_meta, "v3_1")
    conf_v4 = analyze_box_confusion(items_meta, per_items["v4_full"], synth2_extra, yolo_meta, "v4_full")

    worst20 = build_worst20_gallery(per_items["v3_1"], per_items["v4_full"])

    # Rincian per item failures (n=8) untuk perbandingan jelas before vs after
    failures_detail = []
    for fid in sorted(mid for mid, r in per_items["v4_full"].items() if r["dataset"] == "failures"):
        r3 = per_items["v3_1"][fid]
        r4 = per_items["v4_full"][fid]
        failures_detail.append({
            "id": fid,
            "condition": r4["condition"],
            "cer_v31_pct": r3["cer_pct"],
            "cer_v4_pct": r4["cer_pct"],
            "gf1_v31_pct": r3["grapheme_f1_pct"],
            "gf1_v4_pct": r4["grapheme_f1_pct"],
            "wacc_v31_pct": r3["word_acc_folded_pct"],
            "wacc_v4_pct": r4["word_acc_folded_pct"],
            "pred_lat_v31": r3["pred_lat"],
            "pred_lat_v4": r4["pred_lat"],
            "gt_lat": r4["gt_lat"],
        })

    out_report = {
        "generated_at": raw["generatedAt"],
        "environment": raw["environment"],
        "bootstrap_config": {"B": 2000, "seed": 42},
        "commits": {"baseline_v31": "eb19337", "v4_branch": "v4/fase-5-eval"},
        "browser_modes": summaries,
        "python_phase2_ablation_ref": {
            "heldout_real_n42": p2_ablation.get("heldout_real_n42", {}),
            "failures_n8": p2_ablation.get("failures_n8", {}),
            "clean_synth_v2_val_n40": p2_ablation.get("clean_synth_v2_val_n40", {}),
            "c1_note": p2_ablation.get("c1_note", ""),
        },
        "confusion_analysis": {"v3_1": conf_v31, "v4_full": conf_v4},
        "failures_n8_per_case": failures_detail,
        "worst20_real_cases": worst20,
    }

    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out_report, f, ensure_ascii=False, indent=2)
    print(f"[SAVED PHASE 5 REPORT] -> {OUT_JSON_PATH}")

    print("\n=== HEADLINE SUMMARY: v3.1 (eb19337) vs v4.0 Full (Browser Chromium WASM) ===")
    for ds_key in ("heldout_real_n42", "failures_n8", "clean_synth_v2_val_n40"):
        s3 = summaries["v3_1"][ds_key]
        s4 = summaries["v4_full"][ds_key]
        print(
            f"  {ds_key}: CER {s3['cer_pct']['mean']}% -> {s4['cer_pct']['mean']}% | "
            f"GF1 {s3['grapheme_f1_pct']['mean']}% -> {s4['grapheme_f1_pct']['mean']}% | "
            f"WAcc(fold) {s3['word_acc_folded_pct']['mean']}% -> {s4['word_acc_folded_pct']['mean']}% | "
            f"p50={s4['latency_ms']['p50']}ms p95={s4['latency_ms']['p95']}ms"
        )
    print(
        f"  distractors_v4_val_n29 FP/img: v3.1={summaries['v3_1']['distractors_v4_val_n29']['fp_boxes_per_img']} -> "
        f"v4.0={summaries['v4_full']['distractors_v4_val_n29']['fp_boxes_per_img']}"
    )


if __name__ == "__main__":
    main()
