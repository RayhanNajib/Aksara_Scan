"""
Bangun arsip pelatihan GPU portabel (ZIP / TAR.GZ) untuk Kaggle Notebooks, Lightning AI Studio,
HuggingFace, atau Google Colab tanpa mengikutsertakan split evaluasi terkunci (`test_heldout` / `eval/heldout_real`).

Penggunaan:
  python tools/build_gpu_bundle.py --out dist_gpu/aksarascan_v4_gpu_bundle.zip
  python tools/build_gpu_bundle.py --dry-run
"""
import argparse
import json
import os
import tarfile
import zipfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_SPLITS = {"train", "val"}
TRAIN_VAL_MANIFESTS = [
    "dataset/external/manifests_v4/synth_v4.jsonl",
    "dataset/external/manifests_v4/hanacaraka_v4.jsonl",
    "dataset/external/manifests_v4/jawa_zip_v4.jsonl",
    "dataset/external/manifests_v4/jawa_syllables_v4.jsonl",
    "dataset/external/manifests_v4/setilanaji_teacher_v4.jsonl",
    "dataset/external/manifests_v4/setilanaji_lines_v4.jsonl",
    "dataset/external/manifests_v4/non_aksara_distractors_v4.jsonl",
    "dataset/external/manifests_v4/stage2_hard_neg_train_only.jsonl",
]
STATIC_CODE_PATHS = [
    "training/requirements.txt",
    "training/yolox_model.py",
    "training/train_yolox.py",
    "training/train_crnn_track_b.py",
    "training/train_stage2_embedder.py",
    "training/build_reference_prototypes.py",
    "training/AksaraScan_v4_GPU_MultiPlatform.ipynb",
    "training/GPU_ALTERNATIVES_RUNBOOK.md",
    "tools/datasets/schema_55.py",
    "tools/datasets/schema_v4.py",
    "tools/datasets/mine_stage2_hard_negatives.py",
    "dataset/schema/classes_v4.yaml",
    "runs_crnn/crnn_vocab.json",
]


def collect_bundle_entries():
    entries = []  # (abs_src_or_None, arcname, optional_bytes)
    seen_arcs = set()
    split_counts = {"train": 0, "val": 0, "excluded": 0}
    total_bytes = 0

    for rel in STATIC_CODE_PATHS:
        ap = os.path.join(ROOT_DIR, rel)
        if os.path.isfile(ap) and rel not in seen_arcs:
            entries.append((ap, rel, None))
            seen_arcs.add(rel)
            total_bytes += os.path.getsize(ap)

    for m_rel in TRAIN_VAL_MANIFESTS:
        m_abs = os.path.join(ROOT_DIR, m_rel)
        if not os.path.isfile(m_abs):
            continue
        kept_lines = []
        with open(m_abs, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                sp = rec.get("split", "train")
                if sp not in ALLOWED_SPLITS:
                    split_counts["excluded"] += 1
                    continue
                if rec.get("qa_status") == "reject":
                    split_counts["excluded"] += 1
                    continue
                split_counts[sp] = split_counts.get(sp, 0) + 1
                kept_lines.append(json.dumps(rec, ensure_ascii=False))

                for key in ("image", "label"):
                    p = rec.get(key)
                    if not p or os.path.isabs(p):
                        continue
                    arc = p.replace("\\", "/")
                    if arc.startswith("eval/"):
                        raise RuntimeError(f"Kebocoran terdeteksi ke eval/: {arc}")
                    ap = os.path.join(ROOT_DIR, arc)
                    if os.path.isfile(ap) and arc not in seen_arcs:
                        entries.append((ap, arc, None))
                        seen_arcs.add(arc)
                        total_bytes += os.path.getsize(ap)

        m_arc = m_rel.replace("\\", "/")
        payload = ("\n".join(kept_lines) + "\n").encode("utf-8")
        entries.append((None, m_arc, payload))
        total_bytes += len(payload)

    return entries, split_counts, total_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dist_gpu/aksarascan_v4_gpu_bundle.zip")
    p.add_argument("--format", choices=["zip", "tar.gz"], default="zip")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    entries, split_counts, total_bytes = collect_bundle_entries()
    print(f"[GPU BUNDLE] files={len(entries)} | raw_size={total_bytes / (1024 * 1024):.1f} MB | splits={split_counts}")
    if args.dry_run:
        return

    out_abs = args.out if os.path.isabs(args.out) else os.path.join(ROOT_DIR, args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    if args.format == "zip" or out_abs.endswith(".zip"):
        with zipfile.ZipFile(out_abs, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for src, arc, payload in entries:
                if payload is not None:
                    zf.writestr(arc, payload)
                else:
                    zf.write(src, arc)
    else:
        with tarfile.open(out_abs, "w:gz") as tf:
            import io
            for src, arc, payload in entries:
                if payload is not None:
                    ti = tarfile.TarInfo(name=arc)
                    ti.size = len(payload)
                    tf.addfile(ti, io.BytesIO(payload))
                else:
                    tf.add(src, arcname=arc)
    sz = os.path.getsize(out_abs) / (1024 * 1024)
    print(f"[BUNDLE SAVED] {out_abs} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
