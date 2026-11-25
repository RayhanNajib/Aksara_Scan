"""
AksaraScan v2.5 YOLOv8-Nano Fine-Tuner & ONNX Web Exporter (55 Kelas)
Melatih / fine-tune detektor YOLOv8n untuk 55 kelas atomik Aksara Jawa:
  - 20 Nglegena (0..19)
  - 20 Pasangan (20..39)
  - 12 Sandhangan & Pangkon (40..51)
  -  3 Tanda Baca Jawa / Pada (52..54: pada_adeg, pada_lingsa, pada_lungsi)
dengan 3 gaya font OpenType (Javanese Text, Noto Sans Javanese, Tuladha Jejeg OT)
+ tulisan tangan nyata Jawa.zip, lalu mengekspor ke ONNX untuk browser.
"""
import os
import json
import shutil
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT_DIR, "dataset_yolo", "data.yaml")
WEB_DIR = os.path.join(ROOT_DIR, "app", "web_camscanner")
ONNX_OUT = os.path.join(WEB_DIR, "aksara_yolo.onnx")
META_OUT = os.path.join(WEB_DIR, "yolo_meta.json")
PREV_BEST = os.path.join(ROOT_DIR, "runs_yolo", "aksara_v2", "weights", "best.pt")

NGLEGENA = [
    ("ha", "\uA9B2"), ("na", "\uA9A4"), ("ca", "\uA995"), ("ra", "\uA9AB"), ("ka", "\uA98F"),
    ("da", "\uA9A2"), ("ta", "\uA9A0"), ("sa", "\uA9B1"), ("wa", "\uA9AE"), ("la", "\uA9AD"),
    ("pa", "\uA9A5"), ("dha", "\uA99D"), ("ja", "\uA997"), ("ya", "\uA9AA"), ("nya", "\uA99A"),
    ("ma", "\uA9A9"), ("ga", "\uA992"), ("ba", "\uA9A7"), ("tha", "\uA99B"), ("nga", "\uA994"),
]

SANDHANGAN_INFO = [
    ("wulu", "\uA9B6", "i", "above"),
    ("suku", "\uA9B8", "u", "below"),
    ("pepet", "\uA9BC", "e", "above"),
    ("taling", "\uA9BA", "è", "left"),
    ("tarung", "\uA9B4", "o_part", "right"),
    ("layar", "\uA982", "r", "above"),
    ("cecak", "\uA981", "ng", "above"),
    ("wignyan", "\uA983", "h", "right"),
    ("pangkon", "\uA9C0", "", "right"),
    ("cakra", "\uA9BF", "ra_med", "below"),
    ("keret", "\uA9BD", "re_med", "below"),
    ("pengkal", "\uA9BE", "ya_med", "right"),
]

PADA_INFO = [
    ("pada_adeg", "\uA9CB", "", "pada"),
    ("pada_lingsa", "\uA9C8", ",", "pada"),
    ("pada_lungsi", "\uA9C9", ".", "pada"),
]


def build_metadata(metrics_dict=None):
    classes, categories = [], []
    unicode_map, latin_map, pos_map = {}, {}, {}

    for name, ch in NGLEGENA:
        classes.append(name)
        categories.append("nglegena")
        unicode_map[name] = ch
        latin_map[name] = name
        pos_map[name] = "base"

    for name, ch in NGLEGENA:
        pname = f"p_{name}"
        classes.append(pname)
        categories.append("pasangan")
        unicode_map[pname] = "\uA9C0" + ch
        latin_map[pname] = name
        pos_map[pname] = "pasangan"

    for name, ch, lat, pos in SANDHANGAN_INFO:
        classes.append(name)
        categories.append("sandhangan")
        unicode_map[name] = ch
        latin_map[name] = lat
        pos_map[name] = pos

    for name, ch, lat, pos in PADA_INFO:
        classes.append(name)
        categories.append("pada")
        unicode_map[name] = ch
        latin_map[name] = lat
        pos_map[name] = pos

    return {
        "model_type": "yolov8n",
        "imgsz": 416,
        "conf_thresh": 0.25,
        "iou_thresh": 0.42,
        "classes": classes,
        "categories": categories,
        "unicode": unicode_map,
        "latin": latin_map,
        "positions": pos_map,
        "metrics": metrics_dict or {},
    }


def main(epochs=20, imgsz=416):
    print("=" * 64)
    print("AKSARASCAN v2.5 — YOLOv8-NANO MULTI-FONT FINE-TUNER (55 KELAS)")
    print("=" * 64)

    base_weights = PREV_BEST if os.path.exists(PREV_BEST) else "yolov8n.pt"
    print(f"[INFO] Memulai transfer/fine-tune dari bobot: {base_weights}")
    model = YOLO(base_weights)

    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        workers=0,
        cache="ram",
        device="cpu",
        project=os.path.join(ROOT_DIR, "runs_yolo"),
        name="aksara_v2_5",
        exist_ok=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=3.0,
        scale=0.18,
        translate=0.05,
        mosaic=0.25,
        mixup=0.0,
        patience=8,
        verbose=True,
    )

    best_pt = os.path.join(ROOT_DIR, "runs_yolo", "aksara_v2_5", "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(ROOT_DIR, "runs_yolo", "aksara_v2_5", "weights", "last.pt")

    print(f"[INFO] Memuat bobot terbaik: {best_pt}")
    best_model = YOLO(best_pt)

    val_res = best_model.val(data=DATA_YAML, imgsz=imgsz, batch=16, workers=0, device="cpu")
    metrics = {
        "map50": round(float(val_res.box.map50) * 100.0, 2),
        "map50_95": round(float(val_res.box.map) * 100.0, 2),
        "precision": round(float(val_res.box.mp) * 100.0, 2),
        "recall": round(float(val_res.box.mr) * 100.0, 2),
        "epochs": epochs,
    }
    print(f"[VAL METRICS] mAP50={metrics['map50']}% | P={metrics['precision']}% | R={metrics['recall']}%")

    exported_path = best_model.export(format="onnx", imgsz=imgsz, opset=12, simplify=False, dynamic=False)
    shutil.copy2(exported_path, ONNX_OUT)
    size_mb = os.path.getsize(ONNX_OUT) / (1024 * 1024)
    print(f"[EXPORT] Tersimpan di {ONNX_OUT} ({size_mb:.2f} MB)")

    sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    out = sess.run(None, {inp_name: dummy})[0]
    print(f"[VERIFY ORT] Input={sess.get_inputs()[0].shape}, Output={out.shape}")

    meta = build_metadata(metrics)
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[META] Metadata tersimpan di {META_OUT}")


if __name__ == "__main__":
    main()
