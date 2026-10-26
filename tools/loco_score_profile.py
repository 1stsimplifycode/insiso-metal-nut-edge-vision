"""Descriptive profile of the LOCO (unseen `scratch`) result: top detector score per unseen image and how
many unseen images would be flagged at a few fixed reference thresholds. Nothing here selects a threshold;
the deployed LOCO threshold comes from val only (artifacts/eval/loco_scratch_threshold.json).

    python tools/loco_score_profile.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "eval"


def top_scores(path: Path) -> dict:
    return {e["id"]: max((p[4] for p in e["pred"]), default=0.0) for e in json.loads(path.read_text())}


def main() -> None:
    thr = json.loads((EVAL / "loco_scratch_threshold.json").read_text())["conf"]
    unseen = top_scores(EVAL / "loco_scratch_unseen_predictions.json")
    val = json.loads((EVAL / "loco_scratch_val_predictions.json").read_text())
    val_good_max = max(max((p[4] for p in e["pred"]), default=0.0) for e in val if e["category"] == "good")
    s = sorted(unseen.values())
    refs = sorted({thr, 0.25, 0.11, 0.05})
    out = {"deployed_threshold_from_val": thr, "max_top_score_on_val_good": round(val_good_max, 4),
           "unseen_top_scores_sorted": [round(x, 4) for x in s],
           "unseen_median_top_score": round(s[len(s) // 2], 4),
           "unseen_flagged_at_reference_thresholds": {str(t): f"{sum(x >= t for x in s)}/{len(s)}" for t in refs}}
    (EVAL / "loco_scratch_unseen_score_profile.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
