"""
Pembagi Set Data Nyata AksaraScan berdasarkan ID Penulis (writer_id) & Sesi (session_id).
Menjamin TIDAK ada kebocoran penulis/sesi antara real_train (60%), real_val (20%),
dan real_test (20%), serta menghasilkan checksum SHA-256 kunci manifes real_test.

Penggunaan:
  python eval/split_real.py --input-manifest eval/real_raw/manifest.jsonl --out-dir eval/splits --seed 42
"""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def split_by_writer(records, ratios=(0.60, 0.20, 0.20), seed=42):
    """Kelompokkan rekaman menurut writer_id dan alokasikan ke train/val/test."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Rasio harus berjumlah 1.0, ditemukan {ratios}")

    by_writer = defaultdict(list)
    for rec in records:
        wid = rec.get("writer_id")
        if not wid:
            raise ValueError(f"Rekaman wajib memiliki writer_id anonim (mis. W01): {rec}")
        by_writer[wid].append(rec)

    writers = sorted(by_writer.keys())
    if len(writers) < 5:
        raise ValueError(
            f"Minimal 5 penulis diperlukan agar pembagian 60/20/20 valid per penulis (saat ini: {len(writers)})."
        )

    rng = random.Random(seed)
    # Acak penulis dengan stratifikasi ringan terhadap jumlah sampel agar seimbang
    writers.sort(key=lambda w: len(by_writer[w]), reverse=True)
    buckets = [writers[i:i + 5] for i in range(0, len(writers), 5)]
    shuffled = []
    for b in buckets:
        rng.shuffle(b)
        shuffled.extend(b)

    total = len(records)
    target_val = total * ratios[1]
    target_test = total * ratios[2]

    split_records = {"real_train": [], "real_val": [], "real_test": []}
    split_writers = {"real_train": [], "real_val": [], "real_test": []}

    # Greedy round-robin assignment on stratified chunks to closely match 20% test, 20% val, 60% train
    for w in shuffled:
        n_test = len(split_records["real_test"])
        n_val = len(split_records["real_val"])
        if len(split_writers["real_test"]) == 0 or (n_test < target_test and (target_test - n_test) >= (target_val - n_val)):
            dst = "real_test"
        elif len(split_writers["real_val"]) == 0 or n_val < target_val:
            dst = "real_val"
        else:
            dst = "real_train"
        split_records[dst].extend(by_writer[w])
        split_writers[dst].append(w)

    # Assert strict writer & session disjointness
    assert set(split_writers["real_train"]).isdisjoint(split_writers["real_val"])
    assert set(split_writers["real_train"]).isdisjoint(split_writers["real_test"])
    assert set(split_writers["real_val"]).isdisjoint(split_writers["real_test"])

    sessions = {
        k: {r.get("session_id") for r in v if r.get("session_id")}
        for k, v in split_records.items()
    }
    assert sessions["real_train"].isdisjoint(sessions["real_val"])
    assert sessions["real_train"].isdisjoint(sessions["real_test"])
    assert sessions["real_val"].isdisjoint(sessions["real_test"])

    return split_records, split_writers


def main():
    parser = argparse.ArgumentParser(description="Pembagi Set Nyata Per Penulis AksaraScan")
    parser.add_argument("--input-manifest", required=True, help="Jalur JSONL semua sampel nyata beranotasi")
    parser.add_argument("--out-dir", default="eval/splits", help="Direktori keluaran manifes split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input_manifest, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    split_records, split_writers = split_by_writer(records, seed=args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for sname, recs in split_records.items():
        out_path = os.path.join(args.out_dir, f"{sname}_manifest.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in recs:
                r_out = dict(r)
                r_out["split"] = sname
                f.write(json.dumps(r_out, ensure_ascii=False) + "\n")
        summary[sname] = {
            "images": len(recs),
            "writers": len(split_writers[sname]),
            "writer_ids": split_writers[sname],
            "manifest": out_path,
        }

    test_manifest = summary["real_test"]["manifest"]
    lock_hash = sha256_file(test_manifest)
    lock_path = os.path.join(args.out_dir, "real_test.lock.sha256")
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(f"{lock_hash}  real_test_manifest.jsonl\n")

    print("Pembagian per penulis selesai:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Kunci SHA-256 real_test tersimpan di: {lock_path}")


if __name__ == "__main__":
    main()
