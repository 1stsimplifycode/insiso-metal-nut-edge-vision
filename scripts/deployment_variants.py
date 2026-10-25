"""Deployment trade-off study on ONE trained checkpoint: input resolution and INT8 quantisation.

For each variant: build ONNX -> choose threshold on val (recall-first rule) -> score test ->
CPU latency (batch 1, ORT default threads). Writes artifacts/deployment_variants.json and
reports/deployment_tradeoffs.md.

    python scripts/deployment_variants.py --weights models/baseline.pt --sizes 640 480 320 --int8
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.onnx_detector import preprocess  # noqa: E402

PY = sys.executable


def run(*args):
    subprocess.run([PY, *map(str, args)], check=True, cwd=ROOT)


def quantize_int8(fp32: Path, out: Path, size: int, n_calib: int = 64) -> None:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    calib = sorted((ROOT / "data" / "yolo" / "baseline" / "images" / "train").glob("*.png"))
    rng = np.random.default_rng(0)
    calib = [calib[i] for i in rng.choice(len(calib), size=min(n_calib, len(calib)), replace=False)]

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            p = next(self.it, None)
            return None if p is None else {"images": preprocess(cv2.imread(str(p)), size)[0]}

    import onnx
    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    # Keep the detection-head decode (DFL softmax, box arithmetic, final concat of 0-640 px boxes with
    # 0-1 scores) in FP32. Quantising it with one shared scale zeroes every score (observed: mAP 0).
    exclude = [n.name for n in onnx.load(str(pre)).graph.node
               if n.name.startswith("/model.23/") and not n.name.startswith(("/model.23/cv2", "/model.23/cv3"))]
    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, nodes_to_exclude=exclude)
    pre.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models" / "baseline.pt"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 480, 320])
    ap.add_argument("--int8", action="store_true", help="also build static INT8 (QDQ) at the first size")
    ap.add_argument("--iters", type=int, default=150)
    a = ap.parse_args()
    from ultralytics import YOLO

    w = Path(a.weights)
    vdir = ROOT / "models" / "variants"
    vdir.mkdir(parents=True, exist_ok=True)
    # Ultralytics writes <weights>.onnx next to the checkpoint; export from a private copy so the
    # main models/<name>.onnx is never overwritten or moved.
    wcopy = vdir / f"_export_src_{w.name}"
    shutil.copy2(w, wcopy)
    variants = []
    for s in a.sizes:
        dst = vdir / f"{w.stem}_{s}.onnx"
        if not dst.exists():
            out = Path(YOLO(str(wcopy)).export(format="onnx", imgsz=s, opset=17, simplify=True, dynamic=False, device="cpu"))
            shutil.move(str(out), dst)
    wcopy.unlink()
        variants.append((f"{w.stem}_{s}", dst, s, "fp32"))
    if a.int8:
        s = a.sizes[0]
        q = vdir / f"{w.stem}_{s}_int8.onnx"
        if not q.exists():
            quantize_int8(vdir / f"{w.stem}_{s}.onnx", q, s)
        variants.append((f"{w.stem}_{s}_int8", q, s, "int8 static QDQ (64 train images)"))

    rows = []
    ev = ROOT / "artifacts" / "eval"
    for name, path, s, prec in variants:
        run("scripts/evaluate.py", "--model", path, "--name", name, "--threshold-rule", "zero_fpr", "--split", "val")
        run("scripts/evaluate.py", "--model", path, "--name", name, "--split", "test")
        m = json.loads((ev / f"{name}_test_metrics.json").read_text())
        out = subprocess.run([PY, "scripts/benchmark_cpu.py", "--onnx", str(path), "--iters", str(a.iters),
                              "--label", f"variant {name}"], check=True, cwd=ROOT, capture_output=True, text=True).stdout
        bench = sorted((ROOT / "artifacts" / "benchmarks").glob(f"{path.stem}_*.json"))[-1]
        b = json.loads(bench.read_text())["results_by_threads"]["default"]
        rows.append({"variant": name, "input": f"{s}x{s}", "precision": prec, "onnx_mb": round(path.stat().st_size / 2**20, 2),
                     "conf": m["conf_threshold"], "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
                     "box_P": m["box"]["precision"], "box_R": m["box"]["recall"],
                     "img_flagged": f"{m['image']['defective_flagged']}/{m['image']['defective_total']}",
                     "good_rejected": f"{m['image']['normal_flagged']}/{m['image']['normal_total']}",
                     "model_ms_mean": b["inference"]["mean_ms"], "model_ms_p95": b["inference"]["p95_ms"],
                     "e2e_ms_mean": b["end_to_end"]["mean_ms"], "e2e_ms_p95": b["end_to_end"]["p95_ms"],
                     "benchmark_file": bench.name})
        print(rows[-1])
    (ROOT / "artifacts" / "deployment_variants.json").write_text(json.dumps(rows, indent=2))
    md = ["# Deployment trade-offs (one checkpoint, local CPU)", "",
          "Threshold per variant chosen on **val** (lowest conf with zero val false rejects), then applied to test "
          "(29 defective / 40 good images). Latency: ONNX Runtime CPU EP, batch 1, default threads, "
          f"{a.iters} iterations after warm-up. End-to-end = PNG load + preprocess + model + postprocess.", "",
          "| variant | input | ONNX MB | conf | test mAP50 | mAP50-95 | box P | box R | defective flagged | good rejected | model ms mean / p95 | end-to-end ms mean / p95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['variant']} | {r['input']} | {r['onnx_mb']} | {r['conf']} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | "
                  f"{r['box_P']:.3f} | {r['box_R']:.3f} | {r['img_flagged']} | {r['good_rejected']} | "
                  f"{r['model_ms_mean']:.1f} / {r['model_ms_p95']:.1f} | {r['e2e_ms_mean']:.1f} / {r['e2e_ms_p95']:.1f} |")
    (ROOT / "reports" / "deployment_tradeoffs.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
