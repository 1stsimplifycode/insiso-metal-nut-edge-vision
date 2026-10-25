"""Generate notebooks/insiso_colab_training.ipynb (keeps the notebook in sync with the scripts)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# InSiSo metal-nut defect detection — Colab training notebook

Runs the full pipeline from the repository's scripts: dataset download → audit → frozen split →
mask-to-box labels → training (baseline + improvement) → held-out evaluation → failure gallery →
ONNX export and parity check → artefact packaging.

* Every step calls the same `scripts/*.py` used locally, so results are reproducible outside Colab.
* Checkpoints are written to Google Drive (if mounted) so an interrupted session can **resume**.
* CPU latency measured here is **Colab CPU**, not the deployment CPU. Run `scripts/benchmark_cpu.py`
  on the target machine for deployment numbers.
""")
md("## 0. Runtime check")
code("""
import subprocess, sys, platform
print(platform.platform(), sys.version)
try:
    import torch
    print("torch", torch.__version__, "CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
except ImportError:
    print("torch not installed yet")
DEVICE = "0" if "torch" in sys.modules and torch.cuda.is_available() else "cpu"
print("training device ->", DEVICE)
""")
md("## 1. Get the code and (optionally) persistent storage")
code("""
import os
REPO_URL = "https://github.com/<your-user>/insiso-metal-nut-edge-vision.git"  # <-- set this
if not os.path.exists("insiso-metal-nut-edge-vision"):
    !git clone $REPO_URL
%cd insiso-metal-nut-edge-vision

# Persist runs/ and models/ on Google Drive so training can resume after a disconnect.
PERSIST = None
try:
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST = "/content/drive/MyDrive/insiso-metal-nut"
    os.makedirs(PERSIST + "/runs", exist_ok=True); os.makedirs(PERSIST + "/models", exist_ok=True)
    for d in ("runs", "models"):
        if os.path.isdir(d) and not os.path.islink(d):
            !cp -rn {d}/. {PERSIST}/{d}/ && rm -rf {d}
        if not os.path.exists(d):
            os.symlink(f"{PERSIST}/{d}", d)
except Exception as e:
    print("Drive not mounted; checkpoints are lost if the session ends:", e)
""")
md("## 2. Dependencies (Colab already provides a CUDA PyTorch)")
code("""
!pip -q install ultralytics==8.4.188 onnx==1.23.2 onnxslim==0.1.98 onnxruntime==1.31.0 psutil py-cpuinfo pytest
""")
md("## 3. Dataset download, verification and audit\n`--allow-mirror` falls back to a pinned unofficial mirror if MVTec's link is down; provenance is recorded either way.")
code("""
!python scripts/download_dataset.py --allow-mirror
!cat data/raw/PROVENANCE.json
!python scripts/audit_dataset.py
""")
md("## 4. Frozen split + mask-to-box labels + QA panels + tests")
code("""
!python scripts/prepare_data.py --qa
!python scripts/prepare_data.py --exclude-category scratch --name loco_scratch
!python -m pytest -q tests
""")
md("## 5. Train baseline (resumes automatically if a checkpoint exists)")
code("""
import os
resume = "--resume" if os.path.exists("runs/train/baseline/weights/last.pt") else ""
!python scripts/train.py --config configs/baseline.yaml --device $DEVICE $resume
""")
md("## 6. Evaluate: threshold chosen on val, then frozen and applied to test")
code("""
!python scripts/evaluate.py --model models/baseline.pt --split val
!python scripts/evaluate.py --model models/baseline.pt --split test
!python scripts/analyze_failures.py --name baseline --split test
""")
md("## 7a. E1: recall-first operating point (no retraining; threshold from val only)")
code("""
!python scripts/evaluate.py --model models/baseline.pt --name baseline_zfr --threshold-rule zero_fpr --split val
!python scripts/evaluate.py --model models/baseline.pt --name baseline_zfr --split test
!python scripts/analyze_failures.py --name baseline_zfr --split test
""")
md("## 7b. E2: merged-fragment annotation policy (one factor changed — see configs/improvement.yaml)\nScored against the SAME per-component reference labels as the baseline, and additionally against merged labels.")
code("""
!python scripts/prepare_data.py --policy merged --merge-gap 20 --name merged20
resume = "--resume" if os.path.exists("runs/train/improvement/weights/last.pt") else ""
!python scripts/train.py --config configs/improvement.yaml --device $DEVICE $resume
for name, rule in (("improvement", "max_f1"), ("improvement_zfr", "zero_fpr")):
    !python scripts/evaluate.py --model models/improvement.pt --name {name} --threshold-rule {rule} --gt-dataset baseline --split val
    !python scripts/evaluate.py --model models/improvement.pt --name {name} --gt-dataset baseline --split test
!python scripts/evaluate.py --model models/baseline.pt --name baseline_vs_mergedGT --threshold-from baseline_zfr --gt-dataset merged20 --split test
!python scripts/evaluate.py --model models/improvement.pt --name improvement_vs_mergedGT --threshold-from improvement_zfr --gt-dataset merged20 --split test
!python scripts/analyze_failures.py --name improvement_zfr --split test
""")
md("## 7c. Leave-one-defect-category-out (scratch) and normal-only PatchCore")
code("""
resume = "--resume" if os.path.exists("runs/train/loco_scratch/weights/last.pt") else ""
!python scripts/train.py --config configs/loco_scratch.yaml --device $DEVICE $resume
!python scripts/evaluate.py --model models/loco_scratch.pt --dataset loco_scratch --gt-dataset loco_scratch --name loco_scratch --threshold-rule zero_fpr --split val
!python scripts/evaluate.py --model models/loco_scratch.pt --dataset loco_scratch --gt-dataset loco_scratch --name loco_scratch --split test
!python scripts/evaluate.py --model models/loco_scratch.pt --dataset loco_scratch --gt-dataset loco_scratch --name loco_scratch --split unseen
!python scripts/anomaly_baseline.py
!python tools/patchcore_recall_first.py
""")
md("## 8. Seed repeats (variance check; not run in the submitted results)")
code("""
!python scripts/run_seeds.py --configs configs/baseline.yaml configs/improvement.yaml --seeds 1 2 --device $DEVICE
""")
md("## 9. ONNX export + parity test + Colab-CPU latency (NOT the deployment CPU)")
code("""
!python scripts/export_onnx.py --weights models/baseline.pt
!python scripts/verify_onnx.py --weights models/baseline.pt --onnx models/baseline.onnx --conf 0.11
!python scripts/deployment_variants.py --weights models/baseline.pt --sizes 640 480 320 --int8
!python scripts/benchmark_cpu.py --onnx models/baseline.onnx --label colab-cpu --iters 100
""")
md("## 10. Visualise a few results")
code("""
from IPython.display import Image, display
import glob
for p in sorted(glob.glob("reports/failures/baseline_test/*.jpg"))[:5]:
    display(Image(p, width=900))
""")
md("## 11. Package artefacts")
code("""
!zip -qr insiso_artifacts.zip models/*.onnx models/*.onnx.json artifacts reports manifests configs
if PERSIST:
    !cp insiso_artifacts.zip {PERSIST}/
print("done")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = ROOT / "notebooks" / "insiso_colab_training.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(out)
