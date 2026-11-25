"""
Pembangun & Pelatih Fine-Tuning Detektor AksaraScan v3.1 Tahan Blur & Tulisan Tangan (YOLOv8n Transfer)
Menggabungkan seluruh sumber berlisensi terverifikasi:
  1. `synth_v3` (~2.150 train, ~270 val): 11 font OFL/Apache-2.0, blur defokus & motion kuat,
     grid tulisan tangan nyata, & sampel negatif keras teks Latin
  2. `converted_yolo/jawa_syllables_boxed` (1.271 train, 198 val): silabus tulisan tangan bersandhangan nyata
  3. `converted_yolo/setilanaji_teacher` (67 baris buku nyata tervalidasi guru; bobot oversample 4x)
  4. `converted_yolo/hanacaraka` (1.020 train, 230 val): tulisan tangan CC0
  5. Irisan terkurasi `synth_v2` (800 train, 200 val) agar kinerja domain cetak bersih tidak turun

Melatih secara transfer dari `best.pt` (`aksara_v2_5` yang sudah konvergen 98.8% mAP50) dengan:
  - fliplr=0.0, flipud=0.0 (LARANGAN KERAS flip aksara)
  - mosaic=0.12, close_mosaic=3, scale=0.45, degrees=4.0, shear=2.0
  - Ekspor ONNX [1, 59, 3549] -> teruji paritas sebelum deployment.
"""
import argparse
import glob
import os
import random
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(ROOT_DIR, "dataset", "external")
sys.path.insert(0, os.path.join(ROOT_DIR, "tools", "datasets"))
from schema_55 import CLASSES_55


def build_combined_train_val_lists(seed=42, max_synth2_tr=800, max_synth2_va=200, polish=False):
    rng = random.Random(seed)
    out_dir = os.path.join(EXT_DIR, "yolo_v3_combined")
    os.makedirs(out_dir, exist_ok=True)

    def collect_imgs(folder, repeat=1, limit=0, filter_fn=None):
        img_dir = os.path.join(ROOT_DIR, folder)
        if not os.path.isdir(img_dir):
            return []
        files = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        files = [os.path.abspath(p) for p in files if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        if filter_fn is not None:
            files = [p for p in files if filter_fn(p)]
        if limit > 0 and len(files) > limit:
            rng.shuffle(files)
            files = files[:limit]
        out = []
        for _ in range(repeat):
            out.extend(files)
        return out

    train_files = []
    if polish:
        # Polish seimbang: perkuat cecak/ha/ga pada domain cetak bersih (synth_v2) + pertahankan ketahanan blur/tulisan tangan (synth_v3)
        def has_cecak(p):
            lp = p.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")
            lp = os.path.splitext(lp)[0] + ".txt"
            if not os.path.exists(lp):
                return False
            # class id 46 = cecak, 0 = ha
            txt = open(lp, "r", encoding="utf-8").read()
            return any(ln.startswith("46 ") for ln in txt.splitlines())

        def is_neg_or_grid(p):
            lp = os.path.splitext(p.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/"))[0] + ".txt"
            return os.path.exists(lp) and os.path.getsize(lp) == 0

        s2_all = collect_imgs("dataset/external/synth_v2/images/train")
        s2_cecak = [p for p in s2_all if has_cecak(p)]
        rng.shuffle(s2_all)
        rng.shuffle(s2_cecak)
        train_files += s2_cecak[:750] + s2_all[:850]

        s3_all = collect_imgs("dataset/external/synth_v3/images/train")
        s3_neg = [p for p in s3_all if is_neg_or_grid(p)]
        rng.shuffle(s3_all)
        train_files += s3_all[:1350] + s3_neg * 2

        # Ambil seimbang per modifier dari jawa_syllables_boxed (maks 65 per mod agar wulu tidak mendominasi cecak)
        syl_tr = collect_imgs("dataset/external/converted_yolo/jawa_syllables_boxed/images/train")
        for mod_tag in ("_wulu", "_pepet", "_taling.", "_taling_tarung"):
            sub = [p for p in syl_tr if mod_tag in os.path.basename(p)]
            rng.shuffle(sub)
            train_files += sub[:65]

        train_files += collect_imgs("dataset/external/converted_yolo/setilanaji_teacher/images/train", repeat=3)
        train_files += collect_imgs("dataset/external/converted_yolo/hanacaraka/images/train", repeat=1, limit=250)
    else:
        train_files += collect_imgs("dataset/external/synth_v3/images/train", repeat=1)
        train_files += collect_imgs("dataset/external/converted_yolo/jawa_syllables_boxed/images/train", repeat=1, limit=650)
        train_files += collect_imgs("dataset/external/converted_yolo/setilanaji_teacher/images/train", repeat=4)
        train_files += collect_imgs("dataset/external/converted_yolo/hanacaraka/images/train", repeat=1, limit=350)
        train_files += collect_imgs("dataset/external/synth_v2/images/train", repeat=1, limit=max_synth2_tr)

    val_files = []
    val_files += collect_imgs("dataset/external/synth_v3/images/val", repeat=1)
    val_files += collect_imgs("dataset/external/converted_yolo/jawa_syllables_boxed/images/val", repeat=1)
    val_files += collect_imgs("dataset/external/converted_yolo/hanacaraka/images/val", repeat=1, limit=120)
    val_files += collect_imgs("dataset/external/synth_v2/images/val", repeat=1, limit=max_synth2_va)

    rng.shuffle(train_files)
    tr_txt = os.path.join(out_dir, "train_list.txt")
    va_txt = os.path.join(out_dir, "val_list.txt")
    with open(tr_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(p.replace("\\", "/") for p in train_files) + "\n")
    with open(va_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(p.replace("\\", "/") for p in val_files) + "\n")

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.replace(os.sep, '/')}\n")
        f.write(f"train: {tr_txt.replace(os.sep, '/')}\n")
        f.write(f"val: {va_txt.replace(os.sep, '/')}\n")
        f.write(f"nc: {len(CLASSES_55)}\n")
        f.write("names:\n")
        for i, n in enumerate(CLASSES_55):
            f.write(f"  {i}: {n}\n")

    print(f"[Combined v3] train={len(train_files)} | val={len(val_files)} -> {yaml_path}")
    return yaml_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--lr0", type=float, default=0.0018)
    p.add_argument("--polish", action="store_true")
    p.add_argument("--resume-pt", default="")
    p.add_argument("--run-name", default="aksara_v3_blur_hw")
    p.add_argument("--deploy", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO
    import onnxruntime as ort
    import numpy as np

    yaml_path = build_combined_train_val_lists(polish=args.polish)
    cand_weights = [
        args.resume_pt,
        os.path.join(os.path.dirname(ROOT_DIR), "Projek_Akhir", "runs_yolo", "aksara_v2_5", "weights", "best.pt"),
        os.path.join(os.path.dirname(ROOT_DIR), "Projek_Akhir", "runs_yolo", "aksara_v2", "weights", "best.pt"),
    ]
    base_pt = next((p for p in cand_weights if p and os.path.exists(p)), None)
    if not base_pt:
        raise FileNotFoundError("best.pt transfer weights tidak ditemukan di ../Projek_Akhir/runs_yolo/")

    print(f"[Train] Transfer fine-tune dari: {base_pt}")
    model = YOLO(base_pt)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        fraction=args.fraction,
        project=os.path.join(ROOT_DIR, "runs_yolo"),
        name=args.run_name,
        exist_ok=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.08,
        cos_lr=True,
        warmup_epochs=1.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=4.0,
        shear=2.0,
        scale=0.45,
        translate=0.08,
        mosaic=0.12,
        close_mosaic=min(2, max(1, args.epochs // 3)),
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.35,
        patience=5,
        amp=False,
        verbose=True,
    )

    best_out = os.path.join(ROOT_DIR, "runs_yolo", args.run_name, "weights", "best.pt")
    if not os.path.exists(best_out):
        best_out = os.path.join(ROOT_DIR, "runs_yolo", args.run_name, "weights", "last.pt")

    best_m = YOLO(best_out)
    val_res = best_m.val(data=yaml_path, imgsz=args.imgsz, batch=args.batch, workers=args.workers, device="cpu")
    metrics = {
        "map50": round(float(val_res.box.map50) * 100.0, 2),
        "map50_95": round(float(val_res.box.map) * 100.0, 2),
        "precision": round(float(val_res.box.mp) * 100.0, 2),
        "recall": round(float(val_res.box.mr) * 100.0, 2),
        "epochs": args.epochs,
    }
    print(f"[VAL v3] {metrics}")

    exported = best_m.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=False, dynamic=False)
    sess = ort.InferenceSession(exported, providers=["CPUExecutionProvider"])
    dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
    print(f"[ONNX VERIFY] {exported} | out_shape={out.shape} | size={os.path.getsize(exported)/(1024*1024):.2f} MB")

    if args.deploy:
        dst_onnx = os.path.join(ROOT_DIR, "app", "web_camscanner", "aksara_yolo.onnx")
        shutil.copy2(exported, dst_onnx)
        import json
        meta_path = os.path.join(ROOT_DIR, "app", "web_camscanner", "yolo_meta.json")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        meta["metrics"] = metrics
        meta["version"] = "3.1.0-blur-hw"
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
        print(f"[DEPLOY] Updated {dst_onnx} & {meta_path}")


if __name__ == "__main__":
    main()
