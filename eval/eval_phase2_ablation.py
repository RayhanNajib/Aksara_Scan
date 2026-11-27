"""
Kalibrasi Stage-2 & Track B (Hanya pada split VAL, 0 Sentuhan eval/heldout_real)
serta Uji Ablation Jujur C0..C3 dengan Bootstrap 95% CI untuk Fase 2 AksaraScan v4.0.

Konfigurasi yang diuji:
  C0: Baseline v3.1 (aksara_yolo.onnx, 416px)
  C1: Detektor YOLOX P2 (opsional --c1-onnx jika bobot GPU tersedia)
  C2: C0 + Stage-2 Metric Embedder + Prototype Bank + Fusion & Penolak non_aksara terkalibrasi
  C3: C2 + Track B CRNN-CTC (runs_crnn/aksara_crnn_ctc.onnx) Pemeriksa Silang

Penggunaan:
  python -u eval/eval_phase2_ablation.py
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))

import run_eval as re_base
from train_stage2_embedder import NON_AKSARA_CID, letterbox_crop_patch, patch_to_tensor


def verify_heldout_lock():
    lock_path = os.path.join(ROOT_DIR, "eval", "heldout_real", "SHA256SUMS.lock")
    if not os.path.exists(lock_path):
        raise RuntimeError(f"Lock file tidak ditemukan: {lock_path}")
    mismatches = []
    checked = 0
    with open(lock_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            expected_hash, rel_path = line.split(None, 1)
            rel_path = rel_path.strip()
            fp = os.path.join(ROOT_DIR, "eval", "heldout_real", rel_path)
            if not os.path.exists(fp):
                mismatches.append((rel_path, "MISSING"))
                continue
            h = hashlib.sha256()
            with open(fp, "rb") as bf:
                for chunk in iter(lambda: bf.read(1024 * 1024), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            checked += 1
            if actual != expected_hash:
                mismatches.append((rel_path, f"MISMATCH {actual[:12]}!={expected_hash[:12]}"))
    if mismatches:
        raise RuntimeError(f"Integritas heldout_real terganggu! {mismatches[:5]}")
    print(f"[HELDOUT LOCK OK] verified {checked} locked files, mismatches=0")


def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class Stage2Verifier:
    """Mesin verifikasi metrik Stage-2 + Bank Prototipe (identik yang akan diport ke JS di Fase 3)."""

    def __init__(self, embed_onnx_path, prototypes_json_path):
        self.sess = ort.InferenceSession(embed_onnx_path, providers=["CPUExecutionProvider"])
        with open(prototypes_json_path, "r", encoding="utf-8") as f:
            self.bank = json.load(f)
        self.proto_mats = []  # per class 0..55 numpy [K_c, 64]
        for c in range(56):
            vecs = [p["vec"] for p in self.bank["classes"][c].get("prototypes", [])]
            if vecs:
                mat = np.array(vecs, dtype=np.float32)
                mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
                self.proto_mats.append(mat)
            else:
                self.proto_mats.append(np.zeros((0, 64), dtype=np.float32))

    def score_crops(self, crops_bgr):
        """Kembalikan (emb [N,64], logits [N,56], proto_max_sim [N,56]) untuk list crop BGR."""
        if not crops_bgr:
            return np.zeros((0, 64), np.float32), np.zeros((0, 56), np.float32), np.zeros((0, 56), np.float32)
        batch = np.stack([patch_to_tensor(letterbox_crop_patch(c, 64)) for c in crops_bgr], axis=0).astype(np.float32)
        emb, logits = self.sess.run(["emb", "logits"], {"crops": batch})
        proto_sims = np.full((len(crops_bgr), 56), -1.0, dtype=np.float32)
        for c, mat in enumerate(self.proto_mats):
            if mat.shape[0] > 0:
                proto_sims[:, c] = (emb @ mat.T).max(axis=1)
        return emb, logits, proto_sims


class CRNNTrackBVerifier:
    """Wrapper inferensi Track B CRNN-CTC (runs_crnn/aksara_crnn_ctc.onnx)."""

    def __init__(self, crnn_onnx_path, vocab_json_path):
        self.sess = ort.InferenceSession(crnn_onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        with open(vocab_json_path, "r", encoding="utf-8") as f:
            v = json.load(f)
        self.chars = v["chars"]

    def predict_line(self, line_bgr):
        gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY) if line_bgr.ndim == 3 else line_bgr
        h, w = gray.shape[:2]
        sc = 64.0 / max(1, h)
        nw = min(768, max(16, int(round(w * sc))))
        resized = cv2.resize(gray, (nw, 64), interpolation=cv2.INTER_AREA if nw < w else cv2.INTER_LINEAR)
        canvas = np.full((64, 768), 245, dtype=np.uint8)
        canvas[:, :nw] = resized
        norm = ((255.0 - canvas.astype(np.float32)) / 255.0)[None, None, ...]
        logits = self.sess.run(None, {self.in_name: norm})[0]  # [T, 1, 97]
        probs = softmax_np(logits[:, 0, :], axis=-1)
        pred_ids = np.argmax(probs, axis=-1)
        max_p = np.max(probs, axis=-1)
        chars = []
        non_blank_ps = []
        prev = 0
        # Hanya baca hingga timestep akhir isi gambar (nw // 4 + 2)
        t_lim = min(len(pred_ids), max(4, (nw // 4) + 2))
        for t in range(t_lim):
            cid = int(pred_ids[t])
            if cid != 0 and cid != prev and (cid - 1) < len(self.chars):
                chars.append(self.chars[cid - 1])
                non_blank_ps.append(float(max_p[t]))
            prev = cid
        mean_conf = float(np.mean(non_blank_ps)) if non_blank_ps else 0.0
        return "".join(chars), mean_conf


def detect_candidates_rich(session, meta, img_bgr, conf_entry=0.25):
    """
    Mirip run_browser_replica_py tetapi menyimpan vektor skor 55 kelas (cls_vec), top1-top2 margin,
    dan gambar RGB skalanya (scaled_rgb) agar Stage-2 bisa meng-crop persis di koordinat canvas.
    """
    ih, iw = img_bgr.shape[:2]
    scale0 = min(1.0, 1500.0 / max(iw, ih))
    CW = int(round(iw * scale0))
    CH = int(round(ih * scale0))
    scaled_bgr = cv2.resize(img_bgr, (CW, CH), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(scaled_bgr, cv2.COLOR_BGR2RGB)
    gray = np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)

    t_otsu = re_base.calculate_otsu_py(gray)
    binary = re_base.adaptive_binary_py(gray, t_otsu)
    binary = re_base.clear_border_noise_py(binary)
    filtered = re_base.apply_median3x3_py(binary)
    morph = re_base.apply_morphology_py(filtered)

    ys, xs = np.where(morph > 0)
    if len(ys) > 30:
        pad = 36
        bx0, by0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        bx1, by1 = min(CW, int(xs.max()) + 1 + pad), min(CH, int(ys.max()) + 1 + pad)
    else:
        bx0, by0, bx1, by1 = 0, 0, CW, CH

    bw_area = max(1, bx1 - bx0)
    bh_area = max(1, by1 - by0)
    tiles = []
    line_bands_out = []
    if max(bw_area, bh_area) <= 580 and (bw_area / bh_area) <= 2.2:
        tiles.append((bx0, by0, bx1, by1))
        line_bands_out.append((by0, by1))
    else:
        line_bands = []
        l_start, empty_run = -1, 0
        min_row_ink = max(2, round(bw_area * 0.004))
        row_sums = morph[by0:by1, bx0:bx1].sum(axis=1)
        for off_y, s in enumerate(row_sums):
            yy = by0 + off_y
            if s >= min_row_ink:
                if l_start < 0:
                    l_start = yy
                empty_run = 0
            elif l_start >= 0:
                empty_run += 1
                if empty_run > 10:
                    if yy - empty_run - l_start >= 14:
                        line_bands.append((l_start, yy - empty_run))
                    l_start, empty_run = -1, 0
        if l_start >= 0 and by1 - l_start >= 14:
            line_bands.append((l_start, by1))

        if line_bands and len(line_bands) <= 6:
            max_lh = max(b[1] - b[0] for b in line_bands)
            active = [b for b in line_bands if (b[1] - b[0]) >= max(20, max_lh * 0.45)] if len(line_bands) >= 2 else line_bands
            chosen = active if active else line_bands
            line_bands_out = chosen
            for ly0, ly1 in chosen:
                lh = max(36, ly1 - ly0)
                pad_v = min(20, max(10, round(lh * 0.22)))
                min_top = 0
                for ay0, ay1 in line_bands:
                    if ay1 <= ly0:
                        min_top = max(min_top, min(ly0 - 2, ay1 + 2))
                max_bot = CH
                for ay0, ay1 in line_bands:
                    if ay0 >= ly1:
                        max_bot = min(max_bot, max(ly1 + 2, ay0 - 2))
                        break
                ty0, ty1 = max(min_top, ly0 - pad_v), min(max_bot, ly1 + pad_v)
                band_h = max(1, ty1 - ty0)
                tile_w = min(440, max(260, round(band_h * 2.8)))
                if bw_area <= tile_w * 1.12:
                    tiles.append((bx0, ty0, bx1, ty1))
                else:
                    step = round(tile_w * 0.65)
                    tx = bx0
                    while tx < bx1:
                        tx_end = min(bx1, tx + tile_w)
                        tx_start = max(bx0, tx_end - tile_w) if tx_end == bx1 else tx
                        tiles.append((tx_start, ty0, tx_end, ty1))
                        if tx_end >= bx1 or len(tiles) >= 12:
                            break
                        tx += step
                if len(tiles) >= 12:
                    break
        else:
            ts = 480
            step = round(ts * 0.72)
            ty = by0
            while ty < by1:
                ty_end = min(by1, ty + ts)
                ty_start = max(by0, ty_end - ts) if (ty_end == by1 and ty_end - ty < ts * 0.5) else ty
                tx = bx0
                while tx < bx1:
                    tx_end = min(bx1, tx + ts)
                    tx_start = max(bx0, tx_end - ts) if (tx_end == bx1 and tx_end - tx < ts * 0.5) else tx
                    tiles.append((tx_start, ty_start, tx_end, ty_end))
                    if tx_end >= bx1 or len(tiles) >= 12:
                        break
                    tx += step
                if ty_end >= by1 or len(tiles) >= 12:
                    break
                ty += step

    imgsz = meta.get("imgsz", 416)
    iou_thr = meta.get("iou_thresh", 0.42)
    nc = len(meta["classes"])
    inp_name = session.get_inputs()[0].name

    cands = []
    for tx0, ty0, tx1, ty1 in tiles:
        crop_rgb = rgb[ty0:ty1, tx0:tx1]
        oh, ow = crop_rgb.shape[:2]
        sc = min(imgsz / ow, imgsz / oh)
        nw, nh = int(round(ow * sc)), int(round(oh * sc))
        pad_x, pad_y = int(math.floor((imgsz - nw) / 2)), int(math.floor((imgsz - nh) / 2))
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(crop_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        chw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        arr = session.run(None, {inp_name: chw})[0][0]  # (4+nc, A)

        cls_scores = arr[4:4 + nc, :]
        top2_idx = np.argpartition(cls_scores, -2, axis=0)[-2:, :]
        best_c = np.argmax(cls_scores, axis=0)
        best_s = cls_scores[best_c, np.arange(arr.shape[1])]
        keep_a = np.where(best_s > conf_entry)[0]
        local_cands = []
        for a in keep_a:
            c = int(best_c[a])
            s1 = float(best_s[a])
            vec = cls_scores[:, a].copy()
            s_sorted = np.sort(vec)[-2:]
            margin = float(s_sorted[-1] - s_sorted[-2]) if len(s_sorted) >= 2 else s1
            cx, cy, bw, bh = float(arr[0, a]), float(arr[1, a]), float(arr[2, a]), float(arr[3, a])
            x0 = max(0.0, (cx - bw / 2.0 - pad_x) / sc) + tx0
            y0 = max(0.0, (cy - bh / 2.0 - pad_y) / sc) + ty0
            x1 = min(float(ow), (cx + bw / 2.0 - pad_x) / sc) + tx0
            y1 = min(float(oh), (cy + bh / 2.0 - pad_y) / sc) + ty0
            if x1 - x0 < 3.0 or y1 - y0 < 3.0:
                continue
            if len(tiles) > 1:
                if (tx0 > bx0 and x0 <= tx0 + 4) or (tx1 < bx1 and x1 >= tx1 - 4):
                    continue
            lbl = meta["classes"][c]
            local_cands.append({
                "classIdx": c, "label": lbl, "category": meta["categories"][c],
                "unicode": meta["unicode"].get(lbl, ""), "confidence": s1,
                "margin": margin, "det_scores": vec,
                "box": [max(0.0, x0), max(0.0, y0), min(float(CW), x1), min(float(CH), y1)],
            })
        cands.extend(re_base.yolo_nms(local_cands, iou_thr, 0.72))

    kept = re_base.yolo_nms(cands, iou_thr, 0.72)
    kept.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return kept, scaled_bgr, line_bands_out, (bx0, by0, bx1, by1)


def precompute_stage2_for_dets(dets, scaled_bgr, verifier):
    if not dets:
        return np.zeros((0, 56), np.float32), np.zeros((0, 56), np.float32)
    H, W = scaled_bgr.shape[:2]
    crops = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        bw, bh = max(2.0, x1 - x0), max(2.0, y1 - y0)
        px0, py0 = max(0, int(round(x0 - 0.10 * bw))), max(0, int(round(y0 - 0.10 * bh)))
        px1, py1 = min(W, int(round(x1 + 0.10 * bw))), min(H, int(round(y1 + 0.10 * bh)))
        crops.append(scaled_bgr[py0:py1, px0:px1] if (px1 - px0 >= 2 and py1 - py0 >= 2) else scaled_bgr)
    _emb, s2_logits, proto_sims = verifier.score_crops(crops)
    return s2_logits, proto_sims


def apply_stage2_fusion(dets, scaled_bgr, verifier, meta, cfg, precomputed=None):
    """Terapkan Stage-2 metric verifier + fusi terkalibrasi + penolakan non_aksara & tanda ragu."""
    if not dets:
        return [], {"verified": 0, "relabeled": 0, "rejected_non_aksara": 0, "abstained": 0}

    if precomputed is not None:
        s2_logits, proto_sims = precomputed
        verify_idxs = [
            (i, (cfg["trigger_lo"] <= d["confidence"] <= cfg["trigger_hi"]) or (d["margin"] < cfg["margin_trigger"]))
            for i, d in enumerate(dets)
        ]
        idx_map = {
            i: (i, is_amb) for i, is_amb in verify_idxs
            if is_amb or cfg.get("vet_all_non_aksara", True)
        }
    else:
        H, W = scaled_bgr.shape[:2]
        verify_idxs = []
        crops = []
        for i, d in enumerate(dets):
            is_ambig = (cfg["trigger_lo"] <= d["confidence"] <= cfg["trigger_hi"]) or (d["margin"] < cfg["margin_trigger"])
            if is_ambig or cfg.get("vet_all_non_aksara", True):
                x0, y0, x1, y1 = d["box"]
                bw, bh = max(2.0, x1 - x0), max(2.0, y1 - y0)
                px0 = max(0, int(round(x0 - 0.10 * bw)))
                py0 = max(0, int(round(y0 - 0.10 * bh)))
                px1 = min(W, int(round(x1 + 0.10 * bw)))
                py1 = min(H, int(round(y1 + 0.10 * bh)))
                crop = scaled_bgr[py0:py1, px0:px1] if (px1 - px0 >= 2 and py1 - py0 >= 2) else scaled_bgr
                verify_idxs.append((i, is_ambig))
                crops.append(crop)
        _emb, s2_logits, proto_sims = verifier.score_crops(crops)
        idx_map = {orig_i: (row_i, is_amb) for row_i, (orig_i, is_amb) in enumerate(verify_idxs)}

    s2_probs = softmax_np(s2_logits / cfg["temp_s2"], axis=1) if len(s2_logits) else s2_logits
    proto_probs = softmax_np(proto_sims * cfg["proto_alpha"], axis=1) if len(proto_sims) else proto_sims

    out = []
    stats = {"verified": len(idx_map), "relabeled": 0, "rejected_non_aksara": 0, "abstained": 0}

    for i, d in enumerate(dets):
        d2 = dict(d)
        d2.pop("det_scores", None)
        if i not in idx_map:
            if d["confidence"] >= cfg["base_conf_min"]:
                out.append(d2)
            continue

        ri, is_ambig = idx_map[i]
        p_s2 = s2_probs[ri]
        p_pr = proto_probs[ri]
        sim_row = proto_sims[ri]

        s2_aks = p_s2[:55]
        pr_aks = p_pr[:55]
        det_aks = d["det_scores"]
        # Gabungan Stage-2 (classifier + nearest prototype similarity)
        s2_comb_aks = cfg["w_cls"] * s2_aks + (1.0 - cfg["w_cls"]) * pr_aks
        neg_bin_cls = float(p_s2[NON_AKSARA_CID] / max(1e-8, p_s2[NON_AKSARA_CID] + np.max(s2_aks)))
        neg_bin_pr = float(p_pr[NON_AKSARA_CID] / max(1e-8, p_pr[NON_AKSARA_CID] + np.max(pr_aks)))
        s2_neg_prob = float(cfg["w_cls"] * neg_bin_cls + (1.0 - cfg["w_cls"]) * neg_bin_pr)

        # Penolakan distraktor non_aksara (Latin / garis dinding / noise)
        neg_thr = cfg["neg_reject_thr_hi"] if d["confidence"] >= 0.55 else cfg["neg_reject_thr_lo"]
        if s2_neg_prob >= neg_thr or neg_bin_cls >= cfg["neg_hard_thr"]:
            stats["rejected_non_aksara"] += 1
            continue

        # Jika detektor kurang dari threshold dasar (mis. 0.18..0.24), butuh konfirmasi positif Stage-2!
        fused = cfg["w_det"] * det_aks + (1.0 - cfg["w_det"]) * s2_comb_aks
        if d["confidence"] < cfg["base_conf_min"]:
            best_c = int(np.argmax(fused))
            if max(float(s2_comb_aks[best_c]), float(p_s2[best_c])) < cfg["recover_s2_min"]:
                continue

        # Hanya ubah label saat Stage-2 memang dipicu (ambigu) DAN keyakinan Stage-2 cukup kuat
        if is_ambig:
            best_c = int(np.argmax(fused))
            top2_f = np.sort(fused)[-2:]
            f_margin = float(top2_f[-1] - top2_f[-2]) if len(top2_f) >= 2 else float(fused[best_c])
            new_score = float(fused[best_c])
            old_score = float(fused[d["classIdx"]])
            if best_c != d["classIdx"] and (new_score - old_score) >= cfg["relabel_delta"] and float(s2_aks[best_c]) >= cfg["relabel_s2_min"]:
                lbl = meta["classes"][best_c]
                d2["classIdx"] = best_c
                d2["label"] = lbl
                d2["category"] = meta["categories"][best_c]
                d2["unicode"] = meta["unicode"].get(lbl, "")
                stats["relabeled"] += 1
            d2["confidence"] = round(new_score, 4)
            d2["proto_sim"] = round(float(sim_row[d2["classIdx"]]), 4)
            d2["non_aksara_prob"] = round(s2_neg_prob, 4)
            if new_score < cfg["abstain_thr"] or (f_margin < cfg["abstain_margin"] and d2["proto_sim"] < cfg["abstain_proto_min"]):
                d2["uncertain"] = True
                stats["abstained"] += 1
        out.append(d2)

    kept = re_base.yolo_nms(out, meta.get("iou_thresh", 0.42), 0.72)
    kept.sort(key=lambda x: (x["box"][1], x["box"][0]))
    return kept, stats


def _decode_pipeline_output(dets):
    uni, lat = re_base.reconstruct_unicode_py(dets)
    return "" if uni == "—" else uni, lat


def bootstrap_ci(values, n_boot=1000, seed=42):
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    mean_v = float(np.mean(arr))
    if len(arr) == 1:
        return {"mean": round(mean_v, 2), "ci_lo": round(mean_v, 2), "ci_hi": round(mean_v, 2)}
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[b] = float(np.mean(arr[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": round(mean_v, 2), "ci_lo": round(float(lo), 2), "ci_hi": round(float(hi), 2)}


def eval_isolated_val_stage2(verifier, seed=42):
    """Ukur akurasi murni Stage-2 Embedder + Prototipe pada split VAL (hanacaraka/jawazip/syllables/synth/distractors)."""
    from train_stage2_embedder import load_stage2_records
    recs = load_stage2_records("val", max_per_real_single=1500, max_synth_per_class=60, seed=seed)
    crops = []
    tgts = []
    doms = []
    cache = {}
    for r in recs:
        ip = r["image"]
        img = cache.get(ip)
        if img is None:
            img = cv2.imread(ip)
            cache[ip] = img
        if img is None:
            continue
        H, W = img.shape[:2]
        if r["bbox"] is None:
            c = img
        else:
            cx, cy, bw, bh = r["bbox"]
            px1, py1 = max(0, int(round((cx - bw * 0.58) * W))), max(0, int(round((cy - bh * 0.58) * H)))
            px2, py2 = min(W, int(round((cx + bw * 0.58) * W))), min(H, int(round((cy + bh * 0.58) * H)))
            c = img[py1:py2, px1:px2] if (px2 - px1 >= 2 and py2 - py1 >= 2) else img
        crops.append(c)
        tgts.append(r["cid"])
        doms.append(r["domain"])

    t0 = time.time()
    emb, logits, proto_sims = verifier.score_crops(crops)
    elapsed_ms = (time.time() - t0) * 1000.0
    tgts = np.array(tgts)

    cls_pred = np.argmax(logits, axis=1)
    proto_pred = np.argmax(proto_sims, axis=1)
    comb_pred = np.argmax(0.6 * softmax_np(logits, 1) + 0.4 * softmax_np(proto_sims * 15.0, 1), axis=1)

    aks = tgts < NON_AKSARA_CID
    neg = tgts == NON_AKSARA_CID
    neg_tp = int(((comb_pred == NON_AKSARA_CID) & neg).sum())
    neg_fp = int(((comb_pred == NON_AKSARA_CID) & aks).sum())
    neg_fn = int(((comb_pred != NON_AKSARA_CID) & neg).sum())
    neg_prec = neg_tp / max(1, neg_tp + neg_fp)
    neg_rec = neg_tp / max(1, neg_tp + neg_fn)
    neg_f1 = 2 * neg_prec * neg_rec / max(1e-8, neg_prec + neg_rec)

    real_m = np.array([d.startswith("hw_") or d == "real_manuscript" for d in doms])
    return {
        "n_val_crops": len(tgts),
        "cls_aks_acc_pct": round(100.0 * float((cls_pred[aks] == tgts[aks]).mean()), 2),
        "proto_nn_aks_acc_pct": round(100.0 * float((proto_pred[aks] == tgts[aks]).mean()), 2),
        "blended_aks_acc_pct": round(100.0 * float((comb_pred[aks] == tgts[aks]).mean()), 2),
        "real_hw_acc_pct": round(100.0 * float((comb_pred[real_m] == tgts[real_m]).mean()), 2) if real_m.any() else None,
        "non_aksara_f1_pct": round(100.0 * neg_f1, 2),
        "non_aksara_prec_pct": round(100.0 * neg_prec, 2),
        "non_aksara_rec_pct": round(100.0 * neg_rec, 2),
        "batch_ms_per_crop": round(elapsed_ms / max(1, len(tgts)), 3),
    }


def calibrate_on_val(det_sess, meta, verifier, crnn, seed=42):
    """
    Kalibrasi grid hiperparameter Stage-2 & Track B STRICT pada `val` split:
      - synth_v4/val (positif multi-baris + halaman negatif Latin/wall murni)
      - setilanaji_lines_v4 (split=="val", bebas dari writer_id yang ada di heldout_real!)
    DILARANG keras membaca eval/heldout_real/ di fungsi ini!
    """
    # Muat heldout writer_ids hanya sebagai blocklist agar val calibration tidak menyentuh satu pun gambar setilanaji val yg kebetulan sama
    heldout_manifest = os.path.join(ROOT_DIR, "eval", "heldout_real", "manifest.jsonl")
    blocked_writers = set()
    with open(heldout_manifest, "r", encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                blocked_writers.add(json.loads(ln).get("writer_id", ""))

    # Muat synth_v4 val (positif & negatif)
    synth_val_pos, synth_val_neg = [], []
    with open(os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4", "synth_v4.jsonl"), "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != "val":
                continue
            ip = os.path.join(ROOT_DIR, r["image"])
            if not os.path.exists(ip):
                continue
            if not r.get("boxes"):
                synth_val_neg.append(r)
            else:
                # Rekonstruksi unicode_gt langsung dari boxes tidak punya spasi/kluster sempurna jika tidak ada; cek label jika ada!
                synth_val_pos.append(r)

    # Batasi 55 setilanaji val + 55 jawa_syllables val + 40 synth_pos + 29 synth_neg (cepat & seimbang, 0 heldout)
    rng = np.random.RandomState(seed)
    if len(synth_val_pos) > 40:
        idx = rng.choice(len(synth_val_pos), 40, replace=False)
        synth_val_pos = [synth_val_pos[i] for i in sorted(idx)]

    jsyl_val = []
    jsyl_manifest = os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4", "jawa_syllables_v4.jsonl")
    if os.path.exists(jsyl_manifest):
        with open(jsyl_manifest, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                if r.get("split") == "val" and r.get("boxes") and os.path.exists(os.path.join(ROOT_DIR, r["image"])):
                    jsyl_val.append(r)
    if len(jsyl_val) > 55:
        idx = rng.choice(len(jsyl_val), 55, replace=False)
        jsyl_val = [jsyl_val[i] for i in sorted(idx)]

    seti_val_lines = []
    with open(os.path.join(ROOT_DIR, "dataset", "external", "manifests_v4", "setilanaji_lines_v4.jsonl"), "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("split") != "val":
                continue
            if r.get("writer_id", "") in blocked_writers:
                continue
            if r.get("unicode") and os.path.exists(os.path.join(ROOT_DIR, r["image"])):
                seti_val_lines.append(r)
    if len(seti_val_lines) > 55:
        idx = rng.choice(len(seti_val_lines), 55, replace=False)
        seti_val_lines = [seti_val_lines[i] for i in sorted(idx)]

    print(f"[CALIBRATE ON VAL] seti_val={len(seti_val_lines)} | jsyl_val={len(jsyl_val)} | synth_pos={len(synth_val_pos)} | synth_neg={len(synth_val_neg)}")

    cached_seti = []
    for r in seti_val_lines:
        bgr = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
        dets18, sc_bgr, _lb, _bx = detect_candidates_rich(det_sess, meta, bgr, conf_entry=0.18)
        pre = precompute_stage2_for_dets(dets18, sc_bgr, verifier)
        crnn_txt, crnn_c = crnn.predict_line(bgr) if crnn else ("", 0.0)
        cached_seti.append((r["unicode"], dets18, sc_bgr, pre, crnn_txt, crnn_c))

    cached_boxed_val = []
    for r in (jsyl_val + synth_val_pos):
        bgr = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
        dets18, sc_bgr, _lb, _bx = detect_candidates_rich(det_sess, meta, bgr, conf_entry=0.18)
        pre = precompute_stage2_for_dets(dets18, sc_bgr, verifier)
        ih, iw = bgr.shape[:2]
        sc0 = min(1.0, 1500.0 / max(iw, ih))
        gt_px = []
        for gb in r["boxes"]:
            cx, cy, bw, bh = gb["bbox"]
            gt_px.append((int(gb["cid"]), [
                (cx - bw / 2.0) * iw * sc0, (cy - bh / 2.0) * ih * sc0,
                (cx + bw / 2.0) * iw * sc0, (cy + bh / 2.0) * ih * sc0,
            ]))
        cached_boxed_val.append((gt_px, dets18, sc_bgr, pre))

    cached_neg = []
    for r in synth_val_neg:
        bgr = cv2.imread(os.path.join(ROOT_DIR, r["image"]))
        dets18, sc_bgr, _lb, _bx = detect_candidates_rich(det_sess, meta, bgr, conf_entry=0.18)
        pre = precompute_stage2_for_dets(dets18, sc_bgr, verifier)
        cached_neg.append((dets18, sc_bgr, pre))

    def eval_boxed_recall_acc(pred_dets_list):
        correct, recalled, tot_gt = 0, 0, 0
        for (gt_px, _d, _s, _p), kept in zip(cached_boxed_val, pred_dets_list):
            for cid, gbox in gt_px:
                tot_gt += 1
                best_iou, best_d = 0.0, None
                for pd in kept:
                    iou = re_base.box_iou(gbox, pd["box"])
                    if iou > best_iou:
                        best_iou, best_d = iou, pd
                if best_d is not None and best_iou >= 0.35:
                    recalled += 1
                    if best_d["classIdx"] == cid:
                        correct += 1
        return (100.0 * correct / max(1, tot_gt)), (100.0 * recalled / max(1, tot_gt))

    # Hitung baseline C0 di val seti_lines + boxed_val + neg
    c0_cers, c0_gf1s, c0_box_lists = [], [], []
    for gt_u, dets18, _sb, _pre, _ct, _cc in cached_seti:
        dets25 = [dict(d) for d in dets18 if d["confidence"] >= 0.25]
        for d in dets25:
            d.pop("det_scores", None)
        u_p, _ = _decode_pipeline_output(dets25)
        c0_cers.append(re_base.compute_cer(gt_u, u_p) * 100.0)
        c0_gf1s.append(re_base.compute_grapheme_f1(gt_u, u_p)["f1"] * 100.0)
    for _gt, dets18, _sb, _pre in cached_boxed_val:
        dets25 = [dict(d) for d in dets18 if d["confidence"] >= 0.25]
        for d in dets25:
            d.pop("det_scores", None)
        c0_box_lists.append(dets25)
    c0_box_acc, c0_box_rec = eval_boxed_recall_acc(c0_box_lists)
    c0_neg_fps = [sum(1 for d in dets18 if d["confidence"] >= 0.25) for dets18, _sb, _pre in cached_neg]

    # Grid search konservatif terarah pada Stage-2 params (di val ONLY!)
    candidates_cfg = []
    for w_det in (0.50, 0.62, 0.74):
        for neg_lo in (0.48, 0.58, 0.68, 0.78):
            for rec_min in (0.72, 1.01):
                for trig_hi in (0.52, 0.62):
                    for rel_delta in (0.04, 0.08):
                        cfg = {
                            "trigger_lo": 0.18,
                            "trigger_hi": trig_hi,
                            "margin_trigger": 0.15,
                            "vet_all_non_aksara": True,
                            "temp_s2": 1.25,
                            "proto_alpha": 14.0,
                            "w_det": w_det,
                            "w_cls": 0.65,
                            "base_conf_min": 0.25,
                            "recover_s2_min": rec_min,
                            "relabel_delta": rel_delta,
                            "relabel_s2_min": 0.45,
                            "neg_reject_thr_lo": neg_lo,
                            "neg_reject_thr_hi": min(0.94, neg_lo + 0.14),
                            "neg_hard_thr": 0.88,
                            "abstain_thr": 0.34,
                            "abstain_margin": 0.12,
                            "abstain_proto_min": 0.42,
                        }
                        cers, gf1s = [], []
                        for gt_u, dets18, sb, pre, _ct, _cc in cached_seti:
                            kept, _ = apply_stage2_fusion(dets18, sb, verifier, meta, cfg, precomputed=pre)
                            u_p, _ = _decode_pipeline_output(kept)
                            cers.append(re_base.compute_cer(gt_u, u_p) * 100.0)
                            gf1s.append(re_base.compute_grapheme_f1(gt_u, u_p)["f1"] * 100.0)
                        c2_box_lists = [apply_stage2_fusion(d18, sb, verifier, meta, cfg, precomputed=pre)[0] for _gt, d18, sb, pre in cached_boxed_val]
                        b_acc, b_rec = eval_boxed_recall_acc(c2_box_lists)
                        neg_fps = []
                        for dets18, sb, pre in cached_neg:
                            kept, _ = apply_stage2_fusion(dets18, sb, verifier, meta, cfg, precomputed=pre)
                            neg_fps.append(len(kept))

                        m_cer = float(np.mean(cers))
                        m_gf1 = float(np.mean(gf1s))
                        m_nfp = float(np.mean(neg_fps))
                        obj = b_acc + 0.8 * m_gf1 - 0.8 * m_cer - 1.5 * m_nfp
                        candidates_cfg.append((obj, m_cer, m_gf1, b_acc, b_rec, m_nfp, cfg))

    candidates_cfg.sort(key=lambda x: x[0], reverse=True)
    best_obj, best_cer, best_gf1, best_bacc, best_brec, best_nfp, best_cfg = candidates_cfg[0]

    # Uji Track B (CRNN-CTC) pada seti_val_lines untuk kalibrasi gerbang Track B
    crnn_cers = []
    crnn_confs = []
    for gt_u, _d18, _sb, _pre, crnn_txt, crnn_c in cached_seti:
        crnn_cers.append(re_base.compute_cer(gt_u, crnn_txt) * 100.0)
        crnn_confs.append(crnn_c)

    # Uji apakah C3 hybrid (ambil CRNN bila line tunggal & crnn_conf >= gate & C2 rata-rata conf < low) membantu di val!
    c3_results = {}
    if crnn is not None:
        c2_seti_cached = [apply_stage2_fusion(dets18, sb, verifier, meta, best_cfg, precomputed=pre)[0] for _gt_u, dets18, sb, pre, _ct, _cc in cached_seti]
        best_c3_cer, best_gate = 999.0, 0.99
        for gate in (0.55, 0.65, 0.75, 0.85, 0.95):
            cers3 = []
            for (gt_u, _d18, _sb, _pre, crnn_txt, crnn_c), kept in zip(cached_seti, c2_seti_cached):
                u_p, _ = _decode_pipeline_output(kept)
                det_mean_c = float(np.mean([d["confidence"] for d in kept])) if kept else 0.0
                chosen_u = crnn_txt if (crnn_txt and crnn_c >= gate and det_mean_c < 0.50) else u_p
                cers3.append(re_base.compute_cer(gt_u, chosen_u) * 100.0)
            mc3 = float(np.mean(cers3))
            if mc3 < best_c3_cer:
                best_c3_cer, best_gate = mc3, gate
        c3_results = {
            "crnn_standalone_val_cer_pct": round(float(np.mean(crnn_cers)), 2),
            "crnn_mean_conf": round(float(np.mean(crnn_confs)), 3),
            "best_gate": best_gate,
            "c3_fused_val_cer_pct": round(best_c3_cer, 2),
        }
    best_cfg["crnn_gate_conf"] = c3_results.get("best_gate", 0.85)
    best_cfg["crnn_low_det_thresh"] = 0.50

    cal_summary = {
        "val_c0_baseline": {
            "seti_lines_n": len(cached_seti),
            "boxed_val_n": len(cached_boxed_val),
            "seti_cer_pct": round(float(np.mean(c0_cers)), 2),
            "seti_grapheme_f1_pct": round(float(np.mean(c0_gf1s)), 2),
            "boxed_val_cls_acc_pct": round(c0_box_acc, 2),
            "boxed_val_recall_pct": round(c0_box_rec, 2),
            "neg_fp_per_img": round(float(np.mean(c0_neg_fps)), 2),
        },
        "val_c2_best": {
            "seti_cer_pct": round(best_cer, 2),
            "seti_grapheme_f1_pct": round(best_gf1, 2),
            "boxed_val_cls_acc_pct": round(best_bacc, 2),
            "boxed_val_recall_pct": round(best_brec, 2),
            "neg_fp_per_img": round(best_nfp, 2),
        },
        "val_c3_track_b": c3_results,
        "frozen_config": best_cfg,
    }
    print(f"  [VAL CALIB] C0: CER={cal_summary['val_c0_baseline']['seti_cer_pct']}% GF1={cal_summary['val_c0_baseline']['seti_grapheme_f1_pct']}% BoxAcc={cal_summary['val_c0_baseline']['boxed_val_cls_acc_pct']}% NegFP={cal_summary['val_c0_baseline']['neg_fp_per_img']}")
    print(f"  [VAL CALIB] C2: CER={cal_summary['val_c2_best']['seti_cer_pct']}% GF1={cal_summary['val_c2_best']['seti_grapheme_f1_pct']}% BoxAcc={cal_summary['val_c2_best']['boxed_val_cls_acc_pct']}% NegFP={cal_summary['val_c2_best']['neg_fp_per_img']}")
    if c3_results:
        print(f"  [VAL CALIB] TrackB standalone CER={c3_results['crnn_standalone_val_cer_pct']}% | C3 fused={c3_results['c3_fused_val_cer_pct']}% (gate={best_cfg['crnn_gate_conf']})")
    return best_cfg, cal_summary


def run_dataset_eval(manifest_path, label, det_sess, meta, verifier, crnn, cal_cfg, c1_sess=None, c1_meta=None):
    with open(manifest_path, "r", encoding="utf-8") as f:
        items = [json.loads(ln) for ln in f if ln.strip()]

    modes = ["C0"]
    if c1_sess is not None:
        modes.append("C1")
    modes.extend(["C2", "C3"] if crnn is not None else ["C2"])

    per_mode = {m: [] for m in modes}
    box_cls_stats = {m: {"correct": 0, "total": 0} for m in modes}

    for it in items:
        ip = os.path.join(ROOT_DIR, it["image"]) if not os.path.isabs(it["image"]) else it["image"]
        bgr = cv2.imread(ip)
        gt_u = it.get("unicode_gt", "")
        meta_subset = it.get("subset", label)
        cond = it.get("condition", it.get("domain", ""))

        # C0: Baseline v3.1
        t0 = time.time()
        dets_c0, sc_bgr, line_bands, _bx = detect_candidates_rich(det_sess, meta, bgr, conf_entry=0.18)
        dets_c0_base = []
        for d in dets_c0:
            if d["confidence"] >= 0.25:
                dc = dict(d)
                dc.pop("det_scores", None)
                dets_c0_base.append(dc)
        u0, l0 = _decode_pipeline_output(dets_c0_base)
        ms0 = (time.time() - t0) * 1000.0

        # C2: Baseline + Stage-2 Verifier + Reference Prototypes
        t2 = time.time()
        dets_c2, s2_stats = apply_stage2_fusion(dets_c0, sc_bgr, verifier, meta, cal_cfg)
        u2, l2 = _decode_pipeline_output(dets_c2)
        ms2 = ms0 + (time.time() - t2) * 1000.0

        # C3: C2 + Track B CRNN-CTC Cross-Check
        u3, l3, dets_c3, ms3 = u2, l2, dets_c2, ms2
        crnn_txt, crnn_conf = "", 0.0
        if crnn is not None:
            t3 = time.time()
            # Untuk single-line strips gunakan seluruh baris; multi-baris rakit per band
            if len(line_bands) <= 1 or bgr.shape[0] <= 120:
                crnn_txt, crnn_conf = crnn.predict_line(bgr)
            else:
                band_txts, band_cs = [], []
                for y0, y1 in line_bands:
                    sub = sc_bgr[max(0, y0 - 4):min(sc_bgr.shape[0], y1 + 4), :]
                    bt, bc = crnn.predict_line(sub)
                    if bt:
                        band_txts.append(bt)
                        band_cs.append(bc)
                crnn_txt = "\n".join(band_txts)
                crnn_conf = float(np.mean(band_cs)) if band_cs else 0.0

            det_mean_c = float(np.mean([d["confidence"] for d in dets_c2])) if dets_c2 else 0.0
            if crnn_txt and crnn_conf >= cal_cfg["crnn_gate_conf"] and det_mean_c < cal_cfg["crnn_low_det_thresh"]:
                u3 = crnn_txt
            ms3 = ms2 + (time.time() - t3) * 1000.0

        # C1 optional (YOLOX P2 ONNX)
        dets_c1, u1, l1, ms1 = [], "", "", 0.0
        if c1_sess is not None:
            t1 = time.time()
            d1_raw, _, _, _ = detect_candidates_rich(c1_sess, c1_meta, bgr, conf_entry=0.25)
            for d in d1_raw:
                dc = dict(d)
                dc.pop("det_scores", None)
                dets_c1.append(dc)
            u1, l1 = _decode_pipeline_output(dets_c1)
            ms1 = (time.time() - t1) * 1000.0

        mode_outputs = {
            "C0": (u0, l0, dets_c0_base, ms0, {}),
            "C2": (u2, l2, dets_c2, ms2, s2_stats),
            "C3": (u3, l3, dets_c3, ms3, {"crnn_conf": round(crnn_conf, 3)}),
        }
        if c1_sess is not None:
            mode_outputs["C1"] = (u1, l1, dets_c1, ms1, {})

        # Box-level IoU >= 0.35 accuracy pada subset beranotasi kotak (hanacaraka_12_unseen_human_writers)
        gt_boxes = it.get("boxes", [])
        ih, iw = bgr.shape[:2]
        sc_box = min(1.0, 1500.0 / max(iw, ih))
        for m, (up, lp, pdets, mms, extra) in mode_outputs.items():
            cer = re_base.compute_cer(gt_u, up) * 100.0
            gf1 = re_base.compute_grapheme_f1(gt_u, up)["f1"] * 100.0
            per_mode[m].append({
                "id": it["id"], "subset": meta_subset, "condition": cond,
                "cer_pct": round(cer, 2), "gf1_pct": round(gf1, 2),
                "n_boxes": len(pdets), "ms": round(mms, 1),
                "gt_u": gt_u, "pred_u": up, "extra": extra,
            })
            if gt_boxes:
                for gb in gt_boxes:
                    cx, cy, bw, bh = gb["bbox"]
                    gx0 = (cx - bw / 2.0) * iw * sc_box
                    gy0 = (cy - bh / 2.0) * ih * sc_box
                    gx1 = (cx + bw / 2.0) * iw * sc_box
                    gy1 = (cy + bh / 2.0) * ih * sc_box
                    best_iou, best_d = 0.0, None
                    for pd in pdets:
                        iou = re_base.box_iou([gx0, gy0, gx1, gy1], pd["box"])
                        if iou > best_iou:
                            best_iou, best_d = iou, pd
                    box_cls_stats[m]["total"] += 1
                    if best_d is not None and best_iou >= 0.35 and best_d["classIdx"] == int(gb["cid"]):
                        box_cls_stats[m]["correct"] += 1

    # Rangkuman dengan Bootstrap 95% CI & rincian subset
    summary = {}
    for m, rows in per_mode.items():
        subsets_map = {}
        for sub_name in sorted(set(r["subset"] for r in rows)):
            sub_rows = [r for r in rows if r["subset"] == sub_name]
            subsets_map[sub_name] = {
                "n": len(sub_rows),
                "cer_pct": bootstrap_ci([r["cer_pct"] for r in sub_rows]),
                "grapheme_f1_pct": bootstrap_ci([r["gf1_pct"] for r in sub_rows]),
            }
        cond_map = {}
        for cname in sorted(set(r["condition"] for r in rows if r["condition"])):
            c_rows = [r for r in rows if r["condition"] == cname]
            cond_map[cname] = {
                "n": len(c_rows),
                "cer_pct": round(float(np.mean([r["cer_pct"] for r in c_rows])), 2),
                "grapheme_f1_pct": round(float(np.mean([r["gf1_pct"] for r in c_rows])), 2),
            }
        b_tot = box_cls_stats[m]["total"]
        b_acc = round(100.0 * box_cls_stats[m]["correct"] / b_tot, 2) if b_tot else None
        summary[m] = {
            "n": len(rows),
            "cer_pct": bootstrap_ci([r["cer_pct"] for r in rows]),
            "grapheme_f1_pct": bootstrap_ci([r["gf1_pct"] for r in rows]),
            "mean_ms": round(float(np.mean([r["ms"] for r in rows])), 1),
            "box_cls_acc_pct": b_acc,
            "by_subset": subsets_map,
            "by_condition": cond_map,
        }
    return summary, per_mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c0-onnx", default="app/web_camscanner/aksara_yolo.onnx")
    p.add_argument("--meta", default="app/web_camscanner/yolo_meta.json")
    p.add_argument("--stage2-onnx", default="app/web_camscanner/aksara_embed.onnx")
    p.add_argument("--prototypes", default="app/web_camscanner/reference_prototypes.json")
    p.add_argument("--crnn-onnx", default="runs_crnn/aksara_crnn_ctc.onnx")
    p.add_argument("--crnn-vocab", default="runs_crnn/crnn_vocab.json")
    p.add_argument("--c1-onnx", default="runs_yolox/v4_smoke/yolox-tiny_img640_s42_fp16.onnx")
    p.add_argument("--c1-imgsz", type=int, default=640)
    p.add_argument("--out-json", default="eval/reports/phase2_ablation.json")
    p.add_argument("--cal-out", default="app/web_camscanner/stage2_calibration.json")
    args = p.parse_args()

    verify_heldout_lock()

    with open(os.path.join(ROOT_DIR, args.meta), "r", encoding="utf-8") as f:
        meta = json.load(f)
    det_sess = ort.InferenceSession(os.path.join(ROOT_DIR, args.c0_onnx), providers=["CPUExecutionProvider"])
    verifier = Stage2Verifier(os.path.join(ROOT_DIR, args.stage2_onnx), os.path.join(ROOT_DIR, args.prototypes))
    crnn = None
    crnn_abs = os.path.join(ROOT_DIR, args.crnn_onnx)
    if os.path.exists(crnn_abs):
        crnn = CRNNTrackBVerifier(crnn_abs, os.path.join(ROOT_DIR, args.crnn_vocab))

    c1_sess, c1_meta = None, None
    c1_abs = os.path.join(ROOT_DIR, args.c1_onnx) if not os.path.isabs(args.c1_onnx) else args.c1_onnx
    if args.c1_onnx and os.path.exists(c1_abs):
        c1_sess = ort.InferenceSession(c1_abs, providers=["CPUExecutionProvider"])
        c1_meta = dict(meta)
        c1_meta["imgsz"] = args.c1_imgsz

    # 1. Ukur Stage-2 pada val terisolasi + Kalibrasi HANYA pada val
    stage2_val_metrics = eval_isolated_val_stage2(verifier)
    print("[STAGE-2 VAL ISOLATED]", json.dumps(stage2_val_metrics))

    cal_cfg, cal_summary = calibrate_on_val(det_sess, meta, verifier, crnn)
    cal_out_abs = os.path.join(ROOT_DIR, args.cal_out)
    with open(cal_out_abs, "w", encoding="utf-8") as f:
        json.dump({"calibration_summary": cal_summary, **cal_cfg}, f, indent=2)
    print(f"[CALIBRATION FROZEN] {cal_out_abs}")

    # 2. Evaluasi sekali pada heldout_real (n=42), failures (n=8), dan parity_20 (regresi sintetis bersih)
    print("\n[EVAL HELD-OUT REAL (n=42)] Running C0..C3...")
    hr_summary, hr_items = run_dataset_eval(
        os.path.join(ROOT_DIR, "eval", "heldout_real", "manifest.jsonl"),
        "heldout_real", det_sess, meta, verifier, crnn, cal_cfg, c1_sess=c1_sess, c1_meta=c1_meta
    )
    print("[EVAL FAILURES (n=8)] Running C0..C3...")
    fl_summary, fl_items = run_dataset_eval(
        os.path.join(ROOT_DIR, "eval", "failures", "manifest.jsonl"),
        "failures", det_sess, meta, verifier, crnn, cal_cfg, c1_sess=c1_sess, c1_meta=c1_meta
    )
    synth2_manifest = os.path.join(ROOT_DIR, "dataset", "external", "synth_v2", "manifest.jsonl")
    clean_synth_summary = {}
    if os.path.exists(synth2_manifest):
        print("[EVAL CLEAN SYNTH_V2 VAL (n=40)] Guarding clean synthetic regression...")
        sv2_clean = []
        with open(synth2_manifest, "r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                if rec.get("split") == "val" and not rec.get("distractor") and rec.get("unicode_gt"):
                    rec["image"] = rec["image"].replace("\\", "/")
                    if os.path.exists(os.path.join(ROOT_DIR, rec["image"])):
                        sv2_clean.append(rec)
        sv2_clean = sv2_clean[:40]
        tmp_sv2 = os.path.join(ROOT_DIR, "eval", "reports", "_tmp_synth2_val40.jsonl")
        os.makedirs(os.path.dirname(tmp_sv2), exist_ok=True)
        with open(tmp_sv2, "w", encoding="utf-8") as f:
            for rec in sv2_clean:
                f.write(json.dumps({"id": rec["id"], "image": rec["image"], "unicode_gt": rec["unicode_gt"], "subset": "synth_v2_val_clean", "condition": rec.get("bg_style", "clean")}) + "\n")
        clean_synth_summary, _ = run_dataset_eval(
            tmp_sv2, "synth_v2_val_clean", det_sess, meta, verifier, crnn, cal_cfg, c1_sess=None
        )
        os.remove(tmp_sv2)

    report = {
        "stage2_val_isolated": stage2_val_metrics,
        "calibration": cal_summary,
        "heldout_real_n42": hr_summary,
        "failures_n8": fl_summary,
        "clean_synth_v2_val_n40": clean_synth_summary,
        "c1_note": f"C1 memakai {args.c1_onnx} (smoke test 2-epoch CPU — full 100-epoch menunggu GPU bundle)" if c1_sess else "C1 skipped",
    }
    out_abs = os.path.join(ROOT_DIR, args.out_json)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[REPORT SAVED] {out_abs}")
    print("=== HELDOUT REAL SUMMARY ===")
    for m, st in hr_summary.items():
        print(f"  {m}: CER={st['cer_pct']} | GF1={st['grapheme_f1_pct']} | BoxClsAcc={st['box_cls_acc_pct']}% | {st['mean_ms']}ms/img")
    print("=== FAILURES SUMMARY ===")
    for m, st in fl_summary.items():
        print(f"  {m}: CER={st['cer_pct']} | GF1={st['grapheme_f1_pct']} | {st['mean_ms']}ms/img")


if __name__ == "__main__":
    main()
