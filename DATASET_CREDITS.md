# Dataset Credits & Licenses

AksaraScan uses **third-party datasets**. This project does not own the
handwriting data. Below is who made what, under which license, and how the
data is divided into segments for training.

## 1. Primary training dataset — Hanacaraka handwritten glyphs

| | |
|---|---|
| Name | Aksara Jawa / Hanacaraka (digital handwriting) |
| Author / Owner | **R. P. Nugroho, GitHub **vzrenggamani**** and community contributors |
| Repository | https://github.com/vzrenggamani/aksarajawa-hanacaraka |
| Kaggle mirror | https://www.kaggle.com/datasets/vzrenggamani/hanacaraka |
| License | **CC0 1.0 Universal** (public domain dedication) — plus PDDL for the database |
| Contents used | 1,579 images, 20 nglegena classes (~79 per class) |
| Location here | `dataset/hanacaraka/` (unmodified copy; original LICENSE, LICENSE.md and README kept inside) |
| Used for | Training/validation/test of the SVM model in `training/` |

Original contributors credited in the upstream README:

> 1. **Aditya Bayu Perdana** (@by.abay) — first contributor, manual handwriting
> 2. **Warga SMA Negeri 1 Wonoayu** — helped collect the data
> 3. **Sakura Fukoka Studio Data Branch** — data cleanup
> 4. All internet participants who submitted digital handwriting via the online questionnaire

If you use `dataset/hanacaraka/`, please cite/credit vzrenggamani and the
contributors above. CC0 imposes no legal requirement, but crediting the
people who drew 1,500+ glyphs by hand is the decent thing to do.

## 2. Auxiliary dataset — Indonesian Local Script Characters

| | |
|---|---|
| Name | Indonesian Local Script Characters |
| Author / Owner | **Aditya Firman Ihsan** |
| DOI | https://doi.org/10.17632/vfj32bpjsf (Mendeley Data, v1, 2024) |
| License | **CC BY 4.0** — attribution is legally required |
| Contents | ~13,400 images incl. Javanese with sandhangan (subsets) |
| Location here | `dataset/indonesian-scripts/` (Jawa.zip, DOI.txt, LICENSE) |
| Used for | Robustness experiments (sandhangan variants); NOT part of the shipped MVP model |

Citation:
> Ihsan, Aditya Firman (2024), 'Indonesian Local Script Characters', Mendeley Data, v1, doi:10.17632/vfj32bpjsf.1

## 3. Segmentation of the data (lecturer requirement)

The 1,579 hanacaraka images are deterministically split per class
(`training/split_dataset.py`, seed 42):

| Segment | Images | Share | Purpose |
|---|---|---|---|
| `dataset/split/train/` | 1,100 | 70% | Model fitting (+9x augmentation each during training) |
| `dataset/split/validation/` | 240 | 15% | Hyperparameter selection (C, augmentation blend) |
| `dataset/split/test/` | 239 | 15% | Final honest evaluation (never seen in training) |

Every file is traceable: `dataset/manifest.csv` lists
`path, class, split, source_license`.

Held-out test metrics of the shipped model (`app/web_camscanner/model.json`):
**68.4%** on clean binarized test segments, **53.2%** on photo-simulated
variants (blur/noise/rotation/illumination).

## 4. Fonts

- **Noto Sans Javanese** (Google Fonts) renders the aksara output.
  License: SIL Open Font License 1.1. Loaded at runtime from Google's CDN;
  the font files are *not* redistributed in this repository.

## 5. What this repository owns

All code (`app/`, `training/`, `docs/`) and the trained model weights
(`model.json` / `model.js`) are original work by **Muhammad Rayhan Najib**
(RayhanNajib), Universitas Negeri Malang, released under the MIT license in
`LICENSE` — except the third-party items listed above, which keep their own
licenses.
