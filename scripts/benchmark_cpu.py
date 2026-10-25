"""Standalone CPU latency benchmark (ONNX Runtime, CPU Execution Provider, batch size 1).

Measures image loading, preprocessing, model execution, postprocessing and end-to-end
per-image time separately, after warm-up, cycling over real images.

    python scripts/benchmark_cpu.py --onnx models/baseline.onnx
    python scripts/benchmark_cpu.py --onnx models/baseline.onnx --threads 1 2 4 6 --iters 300
Writes artifacts/benchmarks/<model>_<host>_<timestamp>.json
"""
import argparse
import datetime as dt
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarking.latency import measure, stats  # noqa: E402
from src.benchmarking.sysinfo import system_info  # noqa: E402
from src.inference.onnx_detector import OnnxDetector  # noqa: E402


def run(onnx_path: str, images: list[Path], threads: int | None, warmup: int, iters: int, conf: float,
        graph_opt: str) -> dict:
    det = OnnxDetector(onnx_path, intra_threads=threads, graph_opt=graph_opt)
    rec = measure(det, images, warmup, iters, conf)
    out = {k: stats(v) for k, v in rec.items()}
    out["model_only_throughput_img_per_s"] = round(1000 / out["inference"]["mean_ms"], 2)
    out["end_to_end_throughput_img_per_s"] = round(1000 / out["end_to_end"]["mean_ms"], 2)
    out["intra_op_threads"] = threads or "ORT default"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--images", default=str(ROOT / "data" / "yolo" / "baseline" / "images" / "test"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--threads", type=int, nargs="*", default=[0], help="intra-op threads; 0 = ORT default")
    ap.add_argument("--graph-opt", default="all", choices=["none", "basic", "extended", "all"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--label", default="", help="free-text tag, e.g. 'local-laptop' or 'colab-cpu'")
    a = ap.parse_args()

    images = sorted(Path(a.images).glob("*.png")) or sorted(Path(a.images).glob("*.jpg"))
    if not images:
        sys.exit(f"no images in {a.images}")
    sample = cv2.imread(str(images[0]))
    try:
        import psutil
        proc = psutil.Process()
        rss0 = proc.memory_info().rss
    except Exception:
        proc = None
    results = {}
    for t in a.threads:
        results[str(t or "default")] = run(a.onnx, images, t or None, a.warmup, a.iters, a.conf, a.graph_opt)
        r = results[str(t or "default")]
        print(f"threads={t or 'default':>7}: model {r['inference']['mean_ms']:.2f} ms (p95 {r['inference']['p95_ms']:.2f}) | "
              f"end-to-end {r['end_to_end']['mean_ms']:.2f} ms (p95 {r['end_to_end']['p95_ms']:.2f}) | "
              f"load {r['load']['mean_ms']:.2f} pre {r['preprocess']['mean_ms']:.2f} post {r['postprocess']['mean_ms']:.2f}")
    det = OnnxDetector(a.onnx)
    rep = {
        "label": a.label, "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "system": system_info(), "model": a.onnx, "model_bytes": Path(a.onnx).stat().st_size,
        "model_input_shape": det.input_shape, "source_image_resolution": f"{sample.shape[1]}x{sample.shape[0]}",
        "batch_size": 1, "warmup_iters": a.warmup, "measured_iters": a.iters, "n_distinct_images": len(images),
        "execution_provider": "CPUExecutionProvider", "graph_optimization": a.graph_opt,
        "results_by_threads": results,
    }
    if proc:
        rep["process_rss_mb_after"] = round(proc.memory_info().rss / 2**20, 1)
        rep["process_rss_mb_delta"] = round((proc.memory_info().rss - rss0) / 2**20, 1)
    host = socket.gethostname().replace(" ", "_")
    out = ROOT / "artifacts" / "benchmarks" / f"{Path(a.onnx).stem}_{host}_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(f"CPU: {rep['system']['cpu']} | {rep['system'].get('physical_cores')}C/{rep['system']['logical_cores']}T | "
          f"{rep['system']['os']} | ORT {rep['system']['onnxruntime']}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
