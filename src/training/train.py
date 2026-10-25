"""Thin, reproducible wrapper around Ultralytics training driven by a YAML config."""
from __future__ import annotations

import json
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if "base" in cfg:  # simple single-level inheritance: override keys of another config
        base = load_config(Path(path).parent / cfg.pop("base"))
        aug = {**base.get("augment", {}), **cfg.pop("augment", {})}
        cfg = {**base, **cfg, "augment": aug}
    return cfg


def environment_info() -> dict:
    import torch
    import ultralytics
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cuda": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def train(cfg: dict, root: Path, device: str | None = None, resume: bool = False) -> dict:
    import torch
    from ultralytics import YOLO

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = device or ("0" if torch.cuda.is_available() else "cpu")
    project = root / "runs" / "train"
    run_dir = project / cfg["name"]
    last = run_dir / "weights" / "last.pt"

    t0 = time.time()
    if resume and last.exists():
        model = YOLO(str(last))
        # Ultralytics permits memory-related overrides (workers, batch, ...) on resume.
        model.train(resume=str(last), workers=cfg["workers"])
    else:
        model = YOLO(cfg["model"])
        model.train(data=str(root / cfg["data"]), epochs=cfg["epochs"], patience=cfg["patience"],
                    batch=cfg["batch"], imgsz=cfg["imgsz"], optimizer=cfg["optimizer"], lr0=cfg["lr0"],
                    weight_decay=cfg["weight_decay"], cos_lr=cfg.get("cos_lr", False), seed=cfg["seed"],
                    deterministic=cfg["deterministic"], workers=cfg["workers"], device=device,
                    project=str(project), name=cfg["name"], exist_ok=True, plots=True, verbose=False,
                    single_cls=cfg.get("single_cls", False), **cfg["augment"])
    dur = time.time() - t0

    best = run_dir / "weights" / "best.pt"
    results_csv = run_dir / "results.csv"
    summary = {"config": cfg, "device": device, "train_seconds": round(dur, 1),
               "environment": environment_info(), "best_checkpoint": str(best.relative_to(root))}
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
        i = int(fit.idxmax())
        summary.update({"epochs_run": int(df["epoch"].iloc[-1]), "selected_epoch": int(df["epoch"].iloc[i]),
                        "val_at_selected_epoch": {k: float(df[k].iloc[i]) for k in df.columns if k.startswith("metrics/")}})
    models = root / "models"
    models.mkdir(exist_ok=True)
    if best.exists():
        shutil.copy2(best, models / f"{cfg['name']}.pt")
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
