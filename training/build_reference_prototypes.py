"""
Pembangun Bank Prototipe Referensi Multi-Sumber (Strict Train-Only, Zero Heldout Leakage)
untuk Stage-2 Reference Prototype Metric Verifier (`reference_prototypes.json`).

Keluaran JSON memuat per kelas (0..54 Aksara + 55 non_aksara):
  - Maksimum 4-6 centroid L2-ternormalisasi per kelas hasil Spherical K-Means pada split TRAIN SAJA
    (dibagi per gaya domain: `handwritten`, `manuscript`, `printed_ofl`, `non_aksara`)
  - Statistik dispersi cosine internal (`mean_cos`, `p10_cos`)
  - Miniatur PNG base64 28x28 kanonik (1 per gaya utama) untuk UI Reference Inspector di Fase 4

Penggunaan:
  python training/build_reference_prototypes.py --onnx app/web_camscanner/aksara_embed.onnx --out app/web_camscanner/reference_prototypes.json
"""
import argparse
import base64
from collections import defaultdict
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
from schema_55 import CLASSES_55
from train_stage2_embedder import NON_AKSARA_CID, NUM_CLASSES, letterbox_crop_patch, load_stage2_records, patch_to_tensor


def spherical_kmeans(embs, k, iters=20, seed=42):
    """Klastering L2-ternormalisasi sederhana via cosine similarity."""
    n = embs.shape[0]
    if n == 0:
        return np.zeros((0, embs.shape[1]), dtype=np.float32), []
    k = min(k, n)
    rng = np.random.RandomState(seed)
    # K-means++ cosine initialization
    centers = [embs[rng.randint(n)]]
    for _ in range(1, k):
        sims = embs @ np.stack(centers, axis=1)
        max_sim = sims.max(axis=1)
        dist = np.clip(1.0 - max_sim, 1e-6, None)
        probs = dist / dist.sum()
        idx = int(rng.choice(n, p=probs))
        centers.append(embs[idx])
    centers = np.stack(centers, axis=0)

    assign = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        sims = embs @ centers.T
        new_assign = np.argmax(sims, axis=1)
        if np.array_equal(new_assign, assign) and _ > 0:
            break
        assign = new_assign
        for j in range(k):
            m = (assign == j)
            if m.any():
                c = embs[m].mean(axis=0)
                norm = np.linalg.norm(c)
                if norm > 1e-8:
                    centers[j] = c / norm

    # Pilih indeks medoid (sampel asli terdekat ke tiap centroid) untuk miniatur representatif
    medoids = []
    for j in range(k):
        m = np.where(assign == j)[0]
        if len(m) == 0:
            medoids.append(0)
            continue
        sub_sims = embs[m] @ centers[j]
        medoids.append(int(m[int(np.argmax(sub_sims))]))
    return centers, medoids


def patch_to_b64_thumb(bgr_64, thumb_sz=28):
    sm = cv2.resize(bgr_64, (thumb_sz, thumb_sz), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", sm)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="app/web_camscanner/aksara_embed.onnx")
    p.add_argument("--out", default="app/web_camscanner/reference_prototypes.json")
    p.add_argument("--max-per-cls-dom", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    onnx_abs = args.onnx if os.path.isabs(args.onnx) else os.path.join(ROOT_DIR, args.onnx)
    sess = ort.InferenceSession(onnx_abs, providers=["CPUExecutionProvider"])

    # Strictly TRAIN split ONLY! Never touch val, test_heldout, or eval/heldout_real!
    recs = load_stage2_records("train", max_per_real_single=5000, max_synth_per_class=220, seed=args.seed)

    # Kelompokkan (cid, style_group)
    def style_group(dom):
        if dom in ("hw_hanacaraka", "hw_jawazip"):
            return "hw"
        if dom in ("hw_syllables", "real_manuscript"):
            return "manuscript"
        if dom.startswith("neg_"):
            return dom
        return "printed_tps"

    buckets = defaultdict(list)
    for r in recs:
        sg = style_group(r["domain"])
        buckets[(r["cid"], sg)].append(r)

    rng = np.random.RandomState(args.seed)
    chosen = []
    for (cid, sg), items in sorted(buckets.items()):
        if len(items) > args.max_per_cls_dom:
            idx = rng.choice(len(items), size=args.max_per_cls_dom, replace=False)
            items = [items[i] for i in idx]
        for it in items:
            chosen.append((cid, sg, it))

    print(f"[PROTOTYPE BANK] Ekstraksi embedding dari {len(chosen)} contoh split=TRAIN (0 data val/heldout)...")
    img_cache = {}
    tensors = []
    patches_64 = []
    meta_rows = []
    for cid, sg, r in chosen:
        ip = r["image"]
        img = img_cache.get(ip)
        if img is None:
            img = cv2.imread(ip)
            if len(img_cache) < 6000:
                img_cache[ip] = img
        if img is None:
            continue
        H, W = img.shape[:2]
        if r["bbox"] is None:
            crop = img
        else:
            cx, cy, bw, bh = r["bbox"]
            px1 = max(0, int(round((cx - bw * 0.56) * W)))
            py1 = max(0, int(round((cy - bh * 0.56) * H)))
            px2 = min(W, int(round((cx + bw * 0.56) * W)))
            py2 = min(H, int(round((cy + bh * 0.56) * H)))
            crop = img[py1:py2, px1:px2] if (px2 - px1 >= 3 and py2 - py1 >= 3) else img
        p64 = letterbox_crop_patch(crop, 64)
        tensors.append(patch_to_tensor(p64))
        patches_64.append(p64)
        meta_rows.append((cid, sg))

    # Batch inference ONNX
    embs_all = []
    bs = 256
    for i in range(0, len(tensors), bs):
        batch_np = np.stack(tensors[i:i + bs], axis=0).astype(np.float32)
        emb, _logits = sess.run(["emb", "logits"], {"crops": batch_np})
        embs_all.append(emb)
    embs_all = np.concatenate(embs_all, axis=0)

    # Susun prototipe per kelas (0..55)
    by_cls = defaultdict(list)
    for idx, (cid, sg) in enumerate(meta_rows):
        by_cls[cid].append((idx, sg))

    classes_out = []
    total_protos = 0
    for cid in range(NUM_CLASSES):
        name = CLASSES_55[cid] if cid < len(CLASSES_55) else "non_aksara"
        rows_c = by_cls.get(cid, [])
        if not rows_c:
            classes_out.append({"cid": cid, "label": name, "prototypes": [], "thumbs": [], "stats": {"n": 0}})
            continue

        idxs_all = np.array([x[0] for x in rows_c])
        embs_c = embs_all[idxs_all]

        # Klastering terpisah untuk gaya tulisan tangan vs cetak/TPS agar kedua ragam terwakili
        protos = []
        thumbs = []
        if cid == NON_AKSARA_CID:
            # Untuk non_aksara, ambil 6 prototipe lintas latin & garis dinding
            centers, medoids = spherical_kmeans(embs_c, k=6, seed=args.seed + cid)
            for j, c_vec in enumerate(centers):
                med_idx = int(idxs_all[medoids[j]])
                protos.append({"style": meta_rows[med_idx][1], "vec": [round(float(v), 5) for v in c_vec]})
            if medoids:
                thumbs.append({"style": "non_aksara", "png": patch_to_b64_thumb(patches_64[int(idxs_all[medoids[0]])])})
        else:
            hw_sub = [x[0] for x in rows_c if x[1] in ("hw", "manuscript")]
            pt_sub = [x[0] for x in rows_c if x[1] == "printed_tps"]
            for sub_name, sub_idxs, k_target in [
                ("handwritten", hw_sub, 2),
                ("printed_tps", pt_sub, 2),
            ]:
                if not sub_idxs:
                    continue
                sub_arr = np.array(sub_idxs)
                centers, medoids = spherical_kmeans(embs_all[sub_arr], k=k_target, seed=args.seed + cid)
                for j, c_vec in enumerate(centers):
                    protos.append({"style": sub_name, "vec": [round(float(v), 5) for v in c_vec]})
                if medoids:
                    thumbs.append({"style": sub_name, "png": patch_to_b64_thumb(patches_64[int(sub_arr[medoids[0]])])})

        proto_mat = np.array([p["vec"] for p in protos], dtype=np.float32)
        sims = (embs_c @ proto_mat.T).max(axis=1)
        classes_out.append({
            "cid": cid,
            "label": name,
            "prototypes": protos,
            "thumbs": thumbs,
            "stats": {
                "n_train": int(len(embs_c)),
                "mean_cos": round(float(np.mean(sims)), 4),
                "p10_cos": round(float(np.percentile(sims, 10)), 4),
            },
        })
        total_protos += len(protos)

    payload = {
        "schema_version": "4.0",
        "emb_dim": 64,
        "num_classes": NUM_CLASSES,
        "non_aksara_cid": NON_AKSARA_CID,
        "source_split": "train_only",
        "total_prototypes": total_protos,
        "classes": classes_out,
    }
    out_abs = args.out if os.path.isabs(args.out) else os.path.join(ROOT_DIR, args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    sz_kb = os.path.getsize(out_abs) / 1024.0
    print(f"[DONE] {out_abs} | classes={len(classes_out)} | total_prototypes={total_protos} | size={sz_kb:.1f} KB")


if __name__ == "__main__":
    main()
