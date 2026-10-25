"""Interleaved multi-model CPU benchmark: rounds of (A, B, C, ...) so background load affects all
models alike. Reports the median across rounds of each round's mean, plus the round-to-round range.

    python scripts/benchmark_compare.py --onnx models/variants/*.onnx --rounds 5 --iters 60
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarking.latency import measure, stats  # noqa: E402
from src.benchmarking.sysinfo import system_info  # noqa: E402
from src.inference.onnx_detector import OnnxDetector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", nargs="+", required=True)
    ap.add_argument("--images", default=str(ROOT / "data" / "yolo" / "baseline" / "images" / "test"))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--threads", type=int, default=0, help="intra-op threads, 0 = ORT default")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    images = sorted(Path(a.images).glob("*.png"))
    dets = {Path(p).stem: OnnxDetector(p, intra_threads=a.threads or None) for p in a.onnx}
    rounds = {k: [] for k in dets}
    pooled = {k: {s: [] for s in ("inference", "end_to_end", "load", "preprocess", "postprocess")} for k in dets}
    for r in range(a.rounds):
        for k, d in dets.items():
            rec = measure(d, images, a.warmup, a.iters)
            rounds[k].append({s: float(np.mean(v)) for s, v in rec.items()})
            for s in pooled[k]:
                pooled[k][s] += rec[s]
        print(f"round {r + 1}/{a.rounds}: " + " | ".join(f"{k} {rounds[k][-1]['inference']:.1f}ms" for k in dets))
    res = {}
    for k in dets:
        res[k] = {
            "onnx_bytes": Path(dets[k].path).stat().st_size, "input": f"{dets[k].size}x{dets[k].size}",
            "model_ms_median_of_round_means": round(float(np.median([x["inference"] for x in rounds[k]])), 2),
            "model_ms_round_range": [round(min(x["inference"] for x in rounds[k]), 2), round(max(x["inference"] for x in rounds[k]), 2)],
            "e2e_ms_median_of_round_means": round(float(np.median([x["end_to_end"] for x in rounds[k]])), 2),
            "e2e_ms_round_range": [round(min(x["end_to_end"] for x in rounds[k]), 2), round(max(x["end_to_end"] for x in rounds[k]), 2)],
            "pooled": {s: stats(v) for s, v in pooled[k].items()},
        }
    rep = {"label": a.label, "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "system": system_info(),
           "batch_size": 1, "rounds": a.rounds, "iters_per_round": a.iters, "warmup_per_round": a.warmup,
           "threads": a.threads or "ORT default", "execution_provider": "CPUExecutionProvider", "results": res}
    out = ROOT / "artifacts" / "benchmarks" / f"compare_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"{'model':28s} {'input':8s} {'MB':>6s} {'model ms (range)':>24s} {'p95':>7s} {'e2e ms (range)':>24s} {'e2e p95':>8s}")
    for k, v in res.items():
        print(f"{k:28s} {v['input']:8s} {v['onnx_bytes'] / 2**20:6.2f} {v['model_ms_median_of_round_means']:8.1f} "
              f"({v['model_ms_round_range'][0]:.1f}-{v['model_ms_round_range'][1]:.1f}) {v['pooled']['inference']['p95_ms']:9.1f} "
              f"{v['e2e_ms_median_of_round_means']:8.1f} ({v['e2e_ms_round_range'][0]:.1f}-{v['e2e_ms_round_range'][1]:.1f}) "
              f"{v['pooled']['end_to_end']['p95_ms']:8.1f}")
    print(f"CPU: {rep['system']['cpu']} | saved {out}")


if __name__ == "__main__":
    main()
