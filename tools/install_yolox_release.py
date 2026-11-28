"""
Pemasang Rilis AksaraYOLOX-P2 (Apache-2.0) untuk AksaraScan v4 Web Runtime
===========================================================================
Skrip drop-in satu perintah untuk mengganti Stage-1 detector
(`app/web_camscanner/aksara_yolo.onnx`) setelah training GPU selesai
(Kaggle T4 x2 / Lightning.ai L4 / Colab / GPU lokal >= 4 GB VRAM).

Fitur:
  1. Memvalidasi model ONNX memakai `onnxruntime` (input 'images' [1,3,imgsz,imgsz],
     output 'output0' [1, 4+nc, num_anchors], cek tidak ada NaN/Inf).
  2. Mendeteksi `imgsz` otomatis dari bentuk input ONNX (mis. 640 atau 960).
  3. Menyimpan cadangan `aksara_yolo.onnx` & `yolo_meta.json` ke `models/backups/`.
  4. Menyalin ONNX baru ke `app/web_camscanner/aksara_yolo.onnx` serta
     memperbarui `app/web_camscanner/yolo_meta.json` (`model_type`, `imgsz`,
     `detector_arch`, `license`, `sha256`, `version`).
  5. Mencetak perintah kalibrasi ulang Stage-2 & evaluasi end-to-end.

Penggunaan:
  python tools/install_yolox_release.py --onnx runs_yolox/v4_stage1/yolox-tiny_img640_s42_fp16.onnx
  python tools/install_yolox_release.py --verify-only app/web_camscanner/aksara_yolo.onnx
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(ROOT_DIR, "app", "web_camscanner")
TARGET_ONNX = os.path.join(WEB_DIR, "aksara_yolo.onnx")
TARGET_META = os.path.join(WEB_DIR, "yolo_meta.json")
BACKUP_ROOT = os.path.join(ROOT_DIR, "models", "backups")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_stage1_onnx(onnx_path: str, expected_nc: int = 55, forced_imgsz: int = 0) -> dict:
    """Memvalidasi tensor I/O dan kestabilan numerik ONNX Stage-1 via ORT."""
    import onnxruntime as ort

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"Berkas ONNX tidak ditemukan: {onnx_path}")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    if len(inps) != 1 or len(outs) != 1:
        raise ValueError(f"Ekspektasi 1 input & 1 output, ditemukan inp={len(inps)} out={len(outs)}")

    inp_name = inps[0].name
    out_name = outs[0].name
    inp_shape = list(inps[0].shape)
    if inp_name != "images" or out_name != "output0":
        raise ValueError(f"Nama tensor wajib images->output0, ditemukan {inp_name}->{out_name}")

    inferred_imgsz = forced_imgsz
    if not inferred_imgsz:
        h = inp_shape[2] if len(inp_shape) >= 4 else None
        w = inp_shape[3] if len(inp_shape) >= 4 else None
        if isinstance(h, int) and isinstance(w, int) and h == w and h >= 256:
            inferred_imgsz = h
        else:
            inferred_imgsz = 640

    dummy = np.random.default_rng(42).uniform(0.0, 1.0, size=(1, 3, inferred_imgsz, inferred_imgsz)).astype(np.float32)
    res = sess.run([out_name], {inp_name: dummy})[0]
    if res.ndim != 3 or res.shape[0] != 1 or res.shape[1] != 4 + expected_nc:
        raise ValueError(f"Bentuk output tidak valid: {res.shape}, ekspektasi (1, {4 + expected_nc}, num_anchors)")
    if not np.isfinite(res).all():
        raise ValueError("Output ONNX mengandung NaN atau Inf pada inferensi uji!")

    sz_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0)
    return {
        "onnx_path": os.path.abspath(onnx_path),
        "size_mb": round(sz_mb, 2),
        "sha256": file_sha256(onnx_path),
        "input_name": inp_name,
        "output_name": out_name,
        "imgsz": int(inferred_imgsz),
        "channels": int(res.shape[1]),
        "num_anchors": int(res.shape[2]),
        "input_dtype": inps[0].type,
    }


def main():
    p = argparse.ArgumentParser(description="Validasi & instalasi release AksaraYOLOX-P2 ke Web Runtime")
    p.add_argument("--onnx", help="Path berkas .onnx hasil training GPU (mis. _fp16.onnx)")
    p.add_argument("--verify-only", default="", help="Hanya validasi ONNX tanpa menyalin ke app/web_camscanner")
    p.add_argument("--imgsz", type=int, default=0, help="Paksa ukuran input imgsz jika tensor dinamis")
    p.add_argument("--arch-name", default="aksara-yolox-p2", help="Nama arsitektur di yolo_meta.json")
    p.add_argument("--license", default="Apache-2.0", help="Metadata lisensi model Stage-1")
    p.add_argument("--version", default="4.1.0-yolox-p2", help="Versi metadata rilis di yolo_meta.json")
    p.add_argument("--dry-run", action="store_true", help="Validasi tanpa mengubah app/web_camscanner")
    args = p.parse_args()

    src = args.verify_only or args.onnx
    if not src:
        p.error("Harus menyertakan --onnx <path> atau --verify-only <path>")
    if os.path.isabs(src) or os.path.exists(src):
        src_abs = os.path.abspath(src)
    else:
        src_abs = os.path.join(ROOT_DIR, src)

    meta = {}
    if os.path.isfile(TARGET_META):
        with open(TARGET_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
    expected_nc = len(meta.get("classes", [])) or 55

    info = verify_stage1_onnx(src_abs, expected_nc=expected_nc, forced_imgsz=args.imgsz)
    print(f"[VERIFIED ONNX] {info['onnx_path']}")
    print(f"  size={info['size_mb']} MB | imgsz={info['imgsz']} | anchors={info['num_anchors']} | sha256={info['sha256'][:16]}...")

    if args.verify_only or args.dry_run:
        print("[DRY-RUN / VERIFY-ONLY] Tidak ada berkas produksi yang diubah.")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bkp_dir = os.path.join(BACKUP_ROOT, f"stage1_{stamp}")
    os.makedirs(bkp_dir, exist_ok=True)
    if os.path.isfile(TARGET_ONNX):
        shutil.copy2(TARGET_ONNX, os.path.join(bkp_dir, "aksara_yolo.onnx"))
    if os.path.isfile(TARGET_META):
        shutil.copy2(TARGET_META, os.path.join(bkp_dir, "yolo_meta.json"))
    print(f"[BACKUP] Disimpan ke {os.path.relpath(bkp_dir, ROOT_DIR)}")

    shutil.copy2(src_abs, TARGET_ONNX)
    meta["model_type"] = args.arch_name
    meta["detector_arch"] = args.arch_name
    meta["imgsz"] = info["imgsz"]
    meta["license"] = args.license
    meta["version"] = args.version
    meta["sha256"] = info["sha256"]
    meta["size_mb"] = info["size_mb"]
    meta["num_anchors"] = info["num_anchors"]

    with open(TARGET_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[INSTALLED] {os.path.relpath(TARGET_ONNX, ROOT_DIR)} & {os.path.relpath(TARGET_META, ROOT_DIR)}")
    print("\nLangkah berikutnya:")
    print("  1. Naikkan CACHE_NAME di app/web_camscanner/sw.js (mis. aksarascan-v18).")
    print("  2. Kalibrasi ulang ambang Stage-2 pada split val saja: python tools/calibrate_stage2.py (jika perlu)")
    print("  3. Jalankan benchmark & pre-release gate:")
    print("     node eval/eval_phase5_browser.mjs && python -u eval/eval_phase5_analyze.py && node eval/eval_phase5_prerelease.mjs")


if __name__ == "__main__":
    main()
