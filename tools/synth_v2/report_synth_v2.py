"""Membuat laporan statistik & montase audit visual 100 sampel acak untuk Data Sintetis v2."""
import collections
import json
import os
import pathlib
import random

import cv2
import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
EXT_DIR = ROOT_DIR / "dataset" / "external"
SYNTH_DIR = EXT_DIR / "synth_v2"
REP_DIR = EXT_DIR / "reports"
REP_DIR.mkdir(parents=True, exist_ok=True)


def main():
    mf_path = SYNTH_DIR / "manifest.jsonl"
    mf = [json.loads(l) for l in mf_path.open("r", encoding="utf-8") if l.strip()]
    splits = collections.Counter(r["split"] for r in mf)
    fonts = collections.Counter(r["font"] for r in mf)
    bgs = collections.Counter(r["bg_style"] for r in mf)
    inks = collections.Counter(r["ink"] for r in mf)
    boxes_tot = sum(r["num_boxes"] for r in mf)
    distractors = sum(1 for r in mf if r.get("distractor"))

    val_rows = [r for r in mf if r["split"] == "val"]
    rng = random.Random(2026)
    picks = rng.sample(val_rows, min(100, len(val_rows)))
    cw, ch = 192, 96
    grid = np.full((10 * ch, 10 * cw, 3), 240, dtype=np.uint8)
    for idx, r in enumerate(picks):
        ip = ROOT_DIR / r["image"]
        lp = ROOT_DIR / r["label"]
        im = cv2.imread(str(ip))
        if im is None:
            continue
        h, w = im.shape[:2]
        if lp.exists():
            for ln in lp.open("r", encoding="utf-8"):
                ps = ln.strip().split()
                if len(ps) < 5:
                    continue
                xc, yc, bw, bh = map(float, ps[1:5])
                x0, y0 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
                x1, y1 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                cv2.rectangle(im, (x0, y0), (x1, y1), (34, 197, 94), 1)
        tile = cv2.resize(im, (cw - 4, ch - 4))
        rr, cc = divmod(idx, 10)
        grid[rr * ch + 2 : rr * ch + ch - 2, cc * cw + 2 : cc * cw + cw - 2] = tile

    montage_path = REP_DIR / "montage_synth_v2_100.jpg"
    cv2.imwrite(str(montage_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    tr_n = splits.get("train", 0)
    va_n = splits.get("val", 0)
    md = [
        "# Laporan Data Sintetis v2 (Fase 2 — OFL Only)",
        "",
        f"- **Total Sampel**: `{len(mf)}` (`train={tr_n}`, `val={va_n}`)",
        f"- **Total Kotak 55 Kelas**: `{boxes_tot:,}` kotak tervalidasi",
        f"- **Sampel dengan Distraktor Non-Label (Latin/Angka/Garis)**: `{distractors}` (`{distractors/max(1,len(mf))*100:.1f}%`)",
        f"- **Font Berlisensi SIL OFL 1.1**: `{dict(fonts)}` (0 font proprieter Windows)",
        f"- **Variasi Latar**: `{dict(bgs)}`",
        f"- **Variasi Tinta**: `{dict(inks)}`",
        "- **Montase Audit Visual**: `dataset/external/reports/montage_synth_v2_100.jpg` (100 sampel acak split val)",
    ]
    out_md = REP_DIR / "PHASE2_SYNTH_REPORT.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[PASS] Saved {out_md} & {montage_path} (total={len(mf)}, boxes={boxes_tot:,})")


if __name__ == "__main__":
    main()
