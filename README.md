# insiso-metal-nut-edge-vision

Camera-based **metal-nut defect detection** that flags defective parts, shows *where* the defect is, and
runs on a **CPU with ONNX Runtime**, with no cloud. Built for the InSiSo Technologies Computer Vision Intern
take-home (candidate: Adishree Gupta).

> **What this is:** a reproducible prototype with measured results and an explicit failure analysis.
> **What it is not:** a production-validated inspection system (one public dataset, 29 defective test
> images, no real line, simulated controller interface). See [Limitations](#limitations).

Full write-up: **[REPORT.md](REPORT.md)** · Written answers A–C: [docs/written_answers.md](docs/written_answers.md) ·
Interview Q&A: [docs/interview_defence.md](docs/interview_defence.md) · AI use: [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md)

## Results at a glance

See [REPORT.md §8–12](REPORT.md) for the full tables. All numbers are on the frozen held-out test split (29 defective + 40 good images),
with the operating threshold chosen on the validation split only.

| Configuration | Threshold (from val) | Defective parts rejected | Good parts rejected | Box P / R @thr | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| Baseline YOLO11n, max-F1 rule | 0.73 | 23/29 | 0/40 | 1.000 / 0.625 | 0.856 | 0.554 |
| **Baseline + E1 recall-first rule (deployed)** | **0.11** | **29/29** (95% CI 88–100%) | **0/40** (95% CI 0–8.8%) | 0.694 / 0.850 | 0.856 | 0.554 |
| E2 merged-fragment labels (not adopted) | 0.05 | 28/29 | 0/40 | 0.744 / 0.800 | 0.798 | 0.526 |
| PatchCore normal-only (no defect labels), recall-first | — | 29/29 | 4/40 | image AUROC 0.996 | — | — |
| LOCO: detector trained **without `scratch`**, tested on all 23 scratch images | 0.51 | **3/23** | — | 1.000 / 0.068 | 0.216 | 0.123 |

| Deployment (Ryzen 5 5600H laptop CPU, ONNX Runtime, batch 1) | Model ms | End-to-end ms | Size |
|---|---|---|---|
| FP32 640×640 (isolated run) | 40.6 mean / 48.9 p95 | 56.1 mean / 66.5 p95 | 10.2 MB |
| INT8 640×640 (interleaved run; FP32 = 58.6 in same run) | 33.3 | 51.2 | 3.1 MB |

ONNX parity vs PyTorch: identical detections on 69 test images (max score Δ 1e-6). 6 escaped defects at the
textbook threshold → 0 with a recall-first threshold, which is the central finding of the failure analysis.

## Pipeline

```
download → audit → frozen split → mask→box labels → train (YOLO11n) → val: pick threshold
        → test: metrics + failure gallery → ONNX export → parity test → CPU benchmark → inspection demo
```

| Step | Script | Output |
|---|---|---|
| Download + verify + extract | `scripts/download_dataset.py` | `data/raw/metal_nut`, `data/raw/PROVENANCE.json` |
| Audit | `scripts/audit_dataset.py` | `reports/dataset_audit.md` |
| Split + labels + QA panels | `scripts/prepare_data.py --qa` | `manifests/*.csv` (frozen), `data/yolo/<name>`, `reports/annotation_qa/` |
| Train | `scripts/train.py --config configs/baseline.yaml` | `models/baseline.pt`, `runs/train/baseline/` |
| Evaluate | `scripts/evaluate.py --split val`, then `--split test` | `artifacts/eval/*_metrics.json` |
| Failure gallery | `scripts/analyze_failures.py` | `reports/failures/<name>_test/failures.md` |
| ONNX export | `scripts/export_onnx.py` | `models/*.onnx` + `.onnx.json` I/O contract |
| ONNX parity | `scripts/verify_onnx.py` | `artifacts/onnx_parity/*.json` |
| CPU benchmark | `scripts/benchmark_cpu.py`, `scripts/benchmark_compare.py` | `artifacts/benchmarks/*.json` |
| Resolution / INT8 study | `scripts/deployment_variants.py` | `reports/deployment_tradeoffs.md` |
| Inspection demo | `scripts/run_inspection.py` | annotated frames + JSON-lines controller messages |
| Normal-only baseline | `scripts/anomaly_baseline.py` | `artifacts/anomaly/patchcore_metrics.json` |
| Seed repeats | `scripts/run_seeds.py` | `artifacts/seeds.json` |

## Reproduce

Tested on Windows 11, Python 3.12.10, PyTorch 2.5.1 (CUDA 12.4, GTX 1650 4 GB). Commands are the same on Linux.
For Google Colab, use **[notebooks/insiso_colab_training.ipynb](notebooks/insiso_colab_training.ipynb)** (GPU
detection, CPU fallback, Drive-backed checkpoints with resume).

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux: source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124   # or /whl/cpu
pip install -r requirements.txt

python scripts/download_dataset.py                       # official MVTec URL; add --allow-mirror if it is down
python scripts/audit_dataset.py
python scripts/prepare_data.py --qa                      # baseline labels (one box per mask component)
python scripts/prepare_data.py --policy merged --merge-gap 20 --name merged20   # E2 labels
python -m pytest -q tests

python scripts/train.py --config configs/baseline.yaml   # --resume after an interruption; --workers 0 on low RAM
python scripts/evaluate.py --model models/baseline.pt --split val    # selects + saves threshold (max-F1 rule)
python scripts/evaluate.py --model models/baseline.pt --split test   # refuses to run without a val threshold
python scripts/analyze_failures.py --name baseline --split test

# E1: recall-first operating point (no retraining)
python scripts/evaluate.py --model models/baseline.pt --name baseline_zfr --threshold-rule zero_fpr --split val
python scripts/evaluate.py --model models/baseline.pt --name baseline_zfr --split test

# E2: merged-fragment annotation policy (retrain, everything else identical)
python scripts/train.py --config configs/improvement.yaml
python scripts/evaluate.py --model models/improvement.pt --dataset merged20 --gt-dataset baseline --name improvement_zfr --threshold-rule zero_fpr --split val
python scripts/evaluate.py --model models/improvement.pt --dataset merged20 --gt-dataset baseline --name improvement_zfr --split test

python scripts/export_onnx.py --weights models/baseline.pt
python scripts/verify_onnx.py --weights models/baseline.pt --onnx models/baseline.onnx --conf 0.11
```

### Dataset
MVTec AD `metal_nut` (CC BY-NC-SA 4.0, **not redistributed here**; see [LICENSE_NOTES.md](LICENSE_NOTES.md)).
The downloader skips work already done, resumes partial downloads, validates the full `.tar.xz` stream, refuses
unsafe archive paths (absolute paths, `..`, links), checks counts against MVTec's published numbers and writes
`PROVENANCE.json`. **In this run the official link returned HTTP 404**, so the data came from a pinned
Hugging Face mirror. Counts match the published numbers; byte-identity is unverified until you run
`python scripts/download_dataset.py --verify-archive path/to/metal_nut.tar.xz`. Manual fallback: put
`metal_nut.tar.xz` in `data/archives/` and re-run the downloader.

### Split
MVTec AD is an *unsupervised* benchmark (defects only in `test/`). For supervised detection, defect images are
re-split **per defect type** (≈50/20/30) and all 242 good images are pooled (val 30, test 40, train 172).
The split uses seed 42, is at source-image level, and is frozen by SHA-256 (`manifests/split_seed42.csv`).
Tests assert disjointness. **These numbers are not comparable with the MVTec AD leaderboard.**

| split | bent | color | flip | scratch | good |
|---|---|---|---|---|---|
| train | 12 | 11 | 11 | 11 | 172 |
| val | 5 | 4 | 5 | 5 | 30 |
| test | 8 | 7 | 7 | 7 | 40 |

### Labels
`src/annotations/mask_to_boxes.py`: binarise >127 → 8-connected components → one tight box each, with no area
filter. 24 of 93 masks are fragmented, which is the origin of experiment E2. QA panels showing image, mask,
components and boxes are in `reports/annotation_qa/`.

### CPU inference and benchmarking on *your* machine
Only `requirements-deploy.txt` is needed (ONNX Runtime, OpenCV, NumPy):
```bash
python scripts/benchmark_cpu.py --onnx models/baseline.onnx --threads 0 1 2 4 6 --iters 200 --label my-machine
```
This reports image load, preprocess, model and postprocess times separately, plus end-to-end mean, median,
p95, p99, std and throughput, with CPU model, core counts, OS, Python and ORT versions, and thread settings.
Batch size is 1. Results go to `artifacts/benchmarks/`.

### Inspection demo (CPU)
```bash
python scripts/run_inspection.py --source path/to/image.png              # window (needs opencv-python, not -headless)
python scripts/run_inspection.py --source data/yolo/baseline/images/test --headless --save-dir artifacts/inspection
python scripts/run_inspection.py --source 0                               # webcam
```
The demo outputs PASS, REJECT, or optional REVIEW (`--review-conf`). Each frame also produces one JSON line through a
**simulated** controller adapter (`src/inspection/controller_adapter.py`, untested on any PLC or ctrlX hardware).

### Web demo (Render)
`app/server.py` wraps the same detector, threshold and decision policy in a small FastAPI app: upload an image,
get PASS/REJECT/REVIEW, the annotated frame and the controller JSON. `POST /api/inspect` (multipart `file`,
optional `?review_conf=`), `GET /health`.
```bash
pip install -r requirements-web.txt
uvicorn app.server:app --port 8000
```
`render.yaml` is a Render Blueprint (free plan, CPU, 1 ORT thread). Latency on shared hosting is not
comparable with the benchmarks above.

### Failure cases
`reports/failures/<model>_test/failures.md`: auto-generated table and panels showing ground truth, predictions above
and below threshold, a zoomed crop, error type, confidence, IoU, box size and local contrast. Interpretation is in
REPORT.md §9.

## Model artifacts
* `models/baseline.onnx`: YOLO11n, 1 class `defect`, input `images` float32 [1,3,640,640] RGB/255
  letterboxed (pad 114); output `output0` [1,5,8400] (cx, cy, w, h, score). NMS is done outside the graph
  (`src/inference/onnx_detector.py`). The full contract is in `models/baseline.onnx.json`.
* `models/improvement.onnx`: E2 model, same contract.
* `models/variants/`: 480 and 320 inputs, INT8 (QDQ, decode head in FP32).

## Limitations
* One public dataset, one acquisition setup: no evidence about other cameras, lighting or part batches.
* Small test set: 29/29 has a 95% Clopper-Pearson interval of 88–100% sensitivity; 0/40 false rejects, 0–8.8%.
* The recall-first threshold (0.11) is barely above the highest validation good-part score (0.10).
* The test set informed the decision to revisit the threshold rule (disclosed in REPORT.md §10).
* Baseline training was interrupted by low memory and resumed (workers 2→0).
* Dataset from a mirror, so byte-identity with the official archive is unverified.
* CPU numbers are from one laptop under normal background load; the target controller must be measured.
* Ultralytics (training and export) is AGPL-3.0; see LICENSE_NOTES.md for deployment implications.

## AI use
Built with Claude Code (AI agent) under my direction. Details are in [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md).

## Repository layout
```
configs/      baseline.yaml, improvement.yaml
docs/         written answers, interview Q&A, AI disclosure, demo script
manifests/    frozen split CSVs + SHA-256
models/       ONNX models (+ I/O contracts); .pt checkpoints git-ignored
notebooks/    Colab notebook (generated by tools/make_notebook.py)
reports/      audit, annotation QA, failure galleries, trade-offs, heatmaps
artifacts/    metrics/threshold JSON, parity results, benchmarks
scripts/      CLI entry points          src/  library code          tests/  pytest
```
