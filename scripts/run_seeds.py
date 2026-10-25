"""Repeat training under different seeds to estimate run-to-run variance.

    python scripts/run_seeds.py --configs configs/baseline.yaml configs/improvement.yaml --seeds 1 2
Each run: train -> val (threshold) -> test, named <config-name>_s<seed>. Summary in artifacts/seeds.json.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train import load_config, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    summary_path = ROOT / "artifacts" / "seeds.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for cfgp in a.configs:
        for s in a.seeds:
            cfg = load_config(Path(cfgp))
            base = cfg["name"]
            cfg["seed"], cfg["name"] = s, f"{base}_s{s}"
            if not (ROOT / "models" / f"{cfg['name']}.pt").exists():
                train(cfg, ROOT, a.device)
            ds = cfg["data"].split("/")[2]
            for split in ("val", "test"):
                subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--model",
                                str(ROOT / "models" / f"{cfg['name']}.pt"), "--split", split, "--dataset", ds,
                                "--imgsz", str(cfg["imgsz"])], check=True)
            m = json.loads((ROOT / "artifacts" / "eval" / f"{cfg['name']}_test_metrics.json").read_text())
            summary[cfg["name"]] = {"config": base, "seed": s, "conf": m["conf_threshold"], "mAP50": m["mAP50"],
                                    "mAP50_95": m["mAP50_95"], "P": m["box"]["precision"], "R": m["box"]["recall"],
                                    "img_sens": m["image"]["sensitivity"], "img_fpr": m["image"]["false_positive_rate"]}
            summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
