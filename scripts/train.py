"""Train a detector from a config.   python scripts/train.py --config configs/baseline.yaml [--resume]"""
import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train import load_config, summarize_run, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu' to force CPU")
    ap.add_argument("--resume", action="store_true", help="resume from runs/train/<name>/weights/last.pt")
    ap.add_argument("--epochs", type=int, default=None, help="override (e.g. quick smoke test)")
    ap.add_argument("--workers", type=int, default=None, help="override dataloader workers (0 = lowest RAM)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="only (re)write train_summary.json and models/<name>.pt for a finished run")
    a = ap.parse_args()
    cfg = load_config(Path(a.config))
    if a.epochs:
        cfg["epochs"] = a.epochs
    if a.workers is not None:
        cfg["workers"] = a.workers
    if a.summarize_only:
        run_dir = ROOT / "runs" / "train" / cfg["name"]
        dev = a.device or str(yaml.safe_load((run_dir / "args.yaml").read_text()).get("device"))
        s = summarize_run(cfg, ROOT, run_dir, dev, None)
    else:
        s = train(cfg, ROOT, a.device, a.resume)
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
