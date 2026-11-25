"""
Pengunduh Dataset Idempoten & Terverifikasi Lisensi — AksaraScan v3.0 (Fase 1)
Fitur:
  - Membaca tools/datasets/datasets.yaml
  - MENOLAK sumber tanpa `license_verified: true` atau status != OK kecuali `--force-review`
  - Mendukung resume HTTP / verifikasi SHA-256 / cermin lokal (Jawa.zip, font OFL)
  - Menyimpan tanda terima & checksum ke dataset/external/_download_receipts.json

Penggunaan:
  python tools/datasets/download.py --all-ok
  python tools/datasets/download.py --id nusax_mt_lexicon --skip-wiki
"""
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.parse

import requests
import yaml
from huggingface_hub import snapshot_download

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YAML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets.yaml")
RECEIPT_PATH = os.path.join(ROOT_DIR, "dataset", "external", "_download_receipts.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_receipts():
    if os.path.exists(RECEIPT_PATH):
        with open(RECEIPT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_receipts(rec):
    os.makedirs(os.path.dirname(RECEIPT_PATH), exist_ok=True)
    with open(RECEIPT_PATH, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)


def download_http_resumable(url, dest_path, expected_sha256=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        if expected_sha256:
            cur = sha256_file(dest_path)
            if cur == expected_sha256:
                return {"file": dest_path, "sha256": cur, "bytes": os.path.getsize(dest_path), "cached": True}
        else:
            # Verify remote Content-Length if available
            try:
                head = requests.head(url, allow_redirects=True, timeout=15, headers={"User-Agent": "AksaraScan-Downloader/1.0"})
                clen = int(head.headers.get("Content-Length", 0))
                if clen > 0 and os.path.getsize(dest_path) == clen:
                    return {"file": dest_path, "sha256": sha256_file(dest_path), "bytes": clen, "cached": True}
            except Exception:
                return {"file": dest_path, "sha256": sha256_file(dest_path), "bytes": os.path.getsize(dest_path), "cached": True}

    part_path = dest_path + ".part"
    resume_pos = os.path.getsize(part_path) if os.path.exists(part_path) else 0
    headers = {"User-Agent": "AksaraScan-Downloader/1.0"}
    if resume_pos > 0:
        headers["Range"] = f"bytes={resume_pos}-"

    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as r:
        if r.status_code == 416 and os.path.exists(part_path):
            shutil.move(part_path, dest_path)
        else:
            r.raise_for_status()
            mode = "ab" if (resume_pos > 0 and r.status_code == 206) else "wb"
            with open(part_path, mode) as f:
                for chunk in r.iter_content(chunk_size=262144):
                    if chunk:
                        f.write(chunk)
            shutil.move(part_path, dest_path)

    digest = sha256_file(dest_path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Checksum tidak cocok untuk {dest_path}: {digest} != {expected_sha256}")
    return {"file": os.path.relpath(dest_path, ROOT_DIR), "sha256": digest, "bytes": os.path.getsize(dest_path), "cached": False}


def process_entry(entry, receipts, force_review=False, skip_large_wiki=False):
    sid = entry["id"]
    status = entry.get("status", "PERLU_KONFIRMASI")
    lic_ok = bool(entry.get("license_verified", False))

    if status == "PERLU_LOGIN":
        print(f"[SKIP LOGIN] {sid}: membutuhkan akun/API key. Ikuti tools/datasets/MANUAL.md.")
        return
    if (status != "OK" or not lic_ok) and not force_review:
        print(f"[DITOLAK LISENSI] {sid} (status={status}, license={entry.get('license')}). Pakai --force-review hanya untuk audit manual.")
        return
    if sid == "jv_wikipedia_dump" and skip_large_wiki:
        print(f"[SKIP] {sid} dilewati (--skip-wiki).")
        return

    local_dir = os.path.join(ROOT_DIR, entry["local_dir"])
    os.makedirs(local_dir, exist_ok=True)
    method = entry.get("method", "")
    files_rec = []

    if method == "local_repo":
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"Folder lokal tidak ditemukan: {local_dir}")
        count = sum(len(fs) for _, _, fs in os.walk(local_dir))
        files_rec.append({"dir": entry["local_dir"], "file_count": count})
        print(f"[OK LOKAL] {sid}: {count} berkas terdeteksi di {entry['local_dir']}")

    elif method == "local_zip_or_manual":
        found = None
        for cand in entry.get("local_mirror_candidates", []):
            p = os.path.abspath(os.path.join(ROOT_DIR, cand))
            if os.path.exists(p):
                found = p
                break
        if found:
            dest = os.path.join(local_dir, os.path.basename(found))
            if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(found):
                shutil.copy2(found, dest)
            files_rec.append({"file": os.path.relpath(dest, ROOT_DIR), "sha256": sha256_file(dest), "bytes": os.path.getsize(dest)})
            print(f"[OK MIRROR] {sid}: terverifikasi ({os.path.getsize(dest)//(1024*1024)} MB)")
        else:
            print(f"[PERLU MANUAL] {sid}: salin Jawa.zip ke {entry['local_dir']} atau lihat MANUAL.md.")

    elif method == "http_or_mirror":
        copied = False
        for cand in entry.get("local_mirror_candidates", []):
            p = os.path.abspath(os.path.join(ROOT_DIR, cand))
            if os.path.exists(p):
                dest = os.path.join(local_dir, os.path.basename(p))
                if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(p):
                    shutil.copy2(p, dest)
                files_rec.append({"file": os.path.relpath(dest, ROOT_DIR), "sha256": sha256_file(dest), "bytes": os.path.getsize(dest)})
                copied = True
        for u in entry.get("urls", []):
            fname = os.path.basename(urllib.parse.urlparse(u).path)
            if not copied or fname.endswith(".txt"):
                dest = os.path.join(local_dir, fname)
                try:
                    files_rec.append(download_http_resumable(u, dest))
                except Exception as e:
                    print(f"[WARN] {sid} unduh {u} gagal ({e}); memakai mirror lokal bila tersedia.")
        print(f"[OK] {sid}: {len(files_rec)} berkas siap.")

    elif method == "http_files":
        for u in entry.get("urls", []):
            fname = os.path.basename(urllib.parse.urlparse(u).path)
            dest = os.path.join(local_dir, fname)
            print(f"  -> Mengunduh {sid}/{fname}...")
            files_rec.append(download_http_resumable(u, dest))
        print(f"[OK] {sid}: {len(files_rec)} berkas siap di {entry['local_dir']}.")

    elif method == "hf_snapshot":
        repo_id = entry["hf_repo"]
        print(f"  -> Mengunduh snapshot HuggingFace {repo_id} (idempoten)...")
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        count = sum(len(fs) for _, _, fs in os.walk(local_dir))
        files_rec.append({"hf_repo": repo_id, "local_dir": entry["local_dir"], "file_count": count})
        print(f"[OK HF] {sid}: {count} berkas siap di {entry['local_dir']}.")

    receipts[sid] = {
        "name": entry["name"],
        "license": entry["license"],
        "status": status,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "artifacts": files_rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Pengunduh Dataset AksaraScan")
    parser.add_argument("--all-ok", action="store_true", help="Unduh seluruh sumber berstatus OK")
    parser.add_argument("--id", help="Unduh satu ID sumber tertentu")
    parser.add_argument("--force-review", action="store_true", help="Izinkan entri belum terverifikasi (khusus audit)")
    parser.add_argument("--skip-wiki", action="store_true", help="Lewati dump jvwiki (58 MB) untuk eksekusi cepat")
    args = parser.parse_args()

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    receipts = load_receipts()
    targets = cfg["sources"]
    if args.id:
        targets = [s for s in targets if s["id"] == args.id]
    elif not args.all_ok:
        parser.error("Gunakan --all-ok atau --id <source_id>")

    for entry in targets:
        process_entry(entry, receipts, force_review=args.force_review, skip_large_wiki=args.skip_wiki)
        save_receipts(receipts)

    print(f"\n[SELESAI] Tanda terima checksum tersimpan di {RECEIPT_PATH}")


if __name__ == "__main__":
    main()
