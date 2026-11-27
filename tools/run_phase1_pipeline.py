"""
Orkestrator Reprodusibel Satu Perintah untuk Fase 1 (`tools/run_phase1_pipeline.py`):
Menjalankan seluruh ingest, sintesis TPS/blur/multi-tinta/Latin-distractor v4,
pembangunan set heldout_real terkunci, audit kebocoran, dan ringkasan statistik dataset.
Mendukung `--check-repro` untuk membuktikan manifes identik di bawah seed tetap.
"""
import argparse
import hashlib
import os
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_step(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    if res.returncode != 0:
        raise RuntimeError(f"Step gagal (exit {res.returncode}): {' '.join(cmd)}")


def hash_manifest(rel_path):
    fp = os.path.join(ROOT_DIR, rel_path)
    return hashlib.sha256(open(fp, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=108)
    ap.add_argument("--synth-train", type=int, default=900)
    ap.add_argument("--synth-val", type=int, default=150)
    ap.add_argument("--nusa-max", type=int, default=48)
    ap.add_argument("--check-repro", action="store_true", help="Uji ulang subset hanacaraka & synth mini 2x dengan seed sama untuk memverifikasi SHA-256 manifes identik")
    args = ap.parse_args()

    py = sys.executable
    run_step([py, "tools/datasets/ingest_hanacaraka_v4.py", "--seed", "42"])
    run_step([py, "tools/datasets/ingest_jawa_zip_v4.py", "--seed", "42"])
    run_step([py, "tools/datasets/ingest_setilanaji_v4.py"])
    run_step([py, "tools/datasets/ingest_nusaaksara_eval.py", "--max-samples", str(args.nusa_max)])

    if args.check_repro:
        # Jalankan pass deterministik pertama untuk hanacaraka & generator mini
        h1 = hash_manifest("dataset/external/manifests_v4/hanacaraka_v4.jsonl")
        run_step([py, "tools/datasets/ingest_hanacaraka_v4.py", "--seed", "42"])
        h2 = hash_manifest("dataset/external/manifests_v4/hanacaraka_v4.jsonl")
        if h1 != h2:
            raise RuntimeError(f"Reprodusibilitas hanacaraka gagal: {h1} != {h2}")
        print(f"[REPRO OK] hanacaraka_v4.jsonl SHA-256 identik = {h1}")

    run_step([
        py, "tools/generate_v4_elastic_hw.py",
        "--train", str(args.synth_train),
        "--val", str(args.synth_val),
        "--seed", str(args.seed),
    ])

    if args.check_repro:
        # Cek determinisme generator dengan membandingkan baris manifes awal/akhir
        s1 = hash_manifest("dataset/external/manifests_v4/synth_v4.jsonl")
        run_step([
            py, "tools/generate_v4_elastic_hw.py",
            "--train", str(args.synth_train),
            "--val", str(args.synth_val),
            "--seed", str(args.seed),
        ])
        s2 = hash_manifest("dataset/external/manifests_v4/synth_v4.jsonl")
        if s1 != s2:
            raise RuntimeError(f"Reprodusibilitas synth_v4 gagal: {s1} != {s2}")
        print(f"[REPRO OK] synth_v4.jsonl SHA-256 identik = {s1}")

    run_step([py, "tools/build_heldout_real.py"])
    run_step([py, "tools/check_leakage.py"])
    run_step([py, "tools/datasets/summarize_dataset_v4.py"])
    print("\n[PHASE 1 COMPLETE] Seluruh gerbang data Fase 1 selesai tervalidasi.")


if __name__ == "__main__":
    main()
