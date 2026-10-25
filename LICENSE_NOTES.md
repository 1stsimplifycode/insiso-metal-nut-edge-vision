# Licence notes

## Dataset — MVTec Anomaly Detection (MVTec AD), category `metal_nut`

- Licence: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**,
  as stated in the `license.txt` shipped inside the category archive.
- Non-commercial use only. This repository uses it for a job-application technical assessment.
- **The dataset is not redistributed here.** `data/` is git-ignored; run `scripts/download_dataset.py`.
- Citation:
  - P. Bergmann, M. Fauser, D. Sattlegger, C. Steger, "MVTec AD — A Comprehensive Real-World Dataset for
    Unsupervised Anomaly Detection", CVPR 2019.
  - P. Bergmann, K. Batzner, M. Fauser, D. Sattlegger, C. Steger, "The MVTec Anomaly Detection Dataset:
    A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", IJCV 129, 2021.
- **Provenance caveat (this run):** on 2026-10-25 the official per-category link
  (`mydrive.ch/shares/150459/...`) returned HTTP 404 for the whole share, and MVTec's page now places
  downloads behind a request form. The copy used here came from the Hugging Face mirror
  `naren-14/mvtec-ad-metal-nut` (pinned revision in `data/raw/MIRROR_REVISION.txt`). The mirror labels
  itself "MIT", which is **incorrect**: the data remain CC BY-NC-SA 4.0. Every folder count matched the
  published MVTec counts and the mirror includes MVTec's original `license.txt`/`readme.txt`, but
  byte-identity with the official archive is **not verified**. Once the official archive is obtained,
  run `python scripts/download_dataset.py --verify-archive <path>/metal_nut.tar.xz` to check every file's SHA-256.

## Derived artifacts

- Bounding-box labels, manifests and trained weights are derived from MVTec AD and therefore inherit the
  **NC-SA** terms; the ONNX models in `models/` are shared under the same terms.

## Software

- Ultralytics YOLO (training/export) is **AGPL-3.0**. Commercial closed-source deployment of Ultralytics
  code or models trained with it requires an Ultralytics Enterprise licence or an alternative detector.
  The deployment path in this repo (`src/inference/onnx_detector.py`) uses only ONNX Runtime (MIT),
  OpenCV (Apache-2.0) and NumPy (BSD), but the *weights* were produced with Ultralytics.
- YOLO11n COCO-pretrained weights: distributed by Ultralytics under AGPL-3.0.
- Project source code in this repository: MIT (see header of README) unless stated otherwise.
