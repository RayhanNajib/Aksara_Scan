"""
Benchmark Arsitektur Detektor Apache-2.0 / MIT untuk AksaraScan v3.0 (Fase 3a).
Mengukur jumlah parameter, estimasi ukuran ONNX FP32 vs FP16 vs INT8 terhadap batas < 20 MB,
jumlah titik jangkar fitur (termasuk P2 stride-4 untuk objek kecil < 16 px), serta latensi CPU lokal.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "training"))
from yolox_model import AksaraYOLOX


def bench_one(variant, use_p2, imgsz):
    model = AksaraYOLOX(num_classes=55, variant=variant, use_p2=use_p2)
    model.eval()
    model.export_onnx = True
    params = sum(p.numel() for p in model.parameters())

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        onnx_path = tf.name
    try:
        dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
        with torch.no_grad():
            pt_out = model(dummy).numpy()
        export_kw = {"input_names": ["images"], "output_names": ["output0"], "opset_version": 14}
        if "dynamo" in torch.onnx.export.__code__.co_varnames:
            export_kw["dynamo"] = False
        torch.onnx.export(model, dummy, onnx_path, **export_kw)
        fp32_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        est_fp16_mb = fp32_mb / 2.0

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inp = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
        # Warmup
        ort_out = sess.run(None, {"images": inp})[0]
        max_diff = float(np.max(np.abs(pt_out - ort_out)))

        t0 = time.perf_counter()
        runs = 3
        for _ in range(runs):
            sess.run(None, {"images": inp})
        cpu_ms = ((time.perf_counter() - t0) / runs) * 1000.0
        anchors = int(ort_out.shape[-1])
    finally:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    return {
        "variant": variant,
        "use_p2": use_p2,
        "imgsz": imgsz,
        "license": "Apache-2.0 (Custom PyTorch YOLOX)",
        "params_M": round(params / 1e6, 2),
        "anchors": anchors,
        "onnx_fp32_mb": round(fp32_mb, 2),
        "est_fp16_mb": round(est_fp16_mb, 2),
        "fits_20mb_fp32": fp32_mb <= 20.0,
        "fits_20mb_fp16": est_fp16_mb <= 20.0,
        "pt_ort_max_diff": round(max_diff, 7),
        "cpu_ort_ms": round(cpu_ms, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval/reports/arch_benchmark.json")
    args = p.parse_args()

    configs = [
        ("yolox-nano", True, 640),
        ("yolox-nano", True, 960),
        ("yolox-tiny", False, 960),
        ("yolox-tiny", True, 960),
        ("yolox-tiny", True, 1280),
        ("yolox-s", True, 960),
    ]
    results = []
    for var, p2, sz in configs:
        r = bench_one(var, p2, sz)
        results.append(r)
        print(
            f"  {r['variant']:10s} | P2={str(r['use_p2']):5s} | imgsz={r['imgsz']:4d} | "
            f"params={r['params_M']:.2f}M | FP32={r['onnx_fp32_mb']:.2f}MB | FP16~={r['est_fp16_mb']:.2f}MB | "
            f"anchors={r['anchors']:6d} | cpu={r['cpu_ort_ms']:.0f}ms | diff={r['pt_ort_max_diff']:.2e}"
        )

    out_abs = args.out if os.path.isabs(args.out) else os.path.join(ROOT_DIR, args.out)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(
            {
                "configs": results,
                "reference_alt": {
                    "name": "rtdetr_v2_r18vd",
                    "license": "Apache-2.0",
                    "note": "Transformer detector bebas-NMS (referensi Fase 3a untuk eksperimen GPU Colab; bobot ~79MB FP32 perlu INT8/distilasi agar <20MB)",
                },
            },
            f,
            indent=2,
        )
    print(f"[Benchmark] Tersimpan ke {out_abs}")


if __name__ == "__main__":
    main()
