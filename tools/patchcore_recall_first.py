"""Apply the recall-first threshold rule (val only) to saved PatchCore scores, for a like-for-like
comparison with the detector's E1 operating point."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "artifacts" / "anomaly" / "patchcore_metrics.json"
m = json.loads(p.read_text())
s = json.loads((ROOT / "artifacts" / "anomaly" / "patchcore_scores.json").read_text())
val_good = max(v["score"] for v in s.values() if v["split"] == "val" and v["category"] == "good")
thr = val_good * 1.0001  # lowest threshold with zero val false rejects


def at(split):
    rows = [v for v in s.values() if v["split"] == split]
    cats = {}
    for v in rows:
        c = cats.setdefault(v["category"], [0, 0])
        c[0] += int(v["score"] >= thr)
        c[1] += 1
    d = [v for v in rows if v["category"] != "good"]
    g = [v for v in rows if v["category"] == "good"]
    return {"threshold": thr, "defective_flagged": sum(v["score"] >= thr for v in d), "defective_total": len(d),
            "normal_flagged": sum(v["score"] >= thr for v in g), "normal_total": len(g),
            "flagged_by_category": {k: f"{a}/{b}" for k, (a, b) in cats.items()}}


m["recall_first_rule"] = "threshold just above the highest val good-part score (val only)"
m["val"]["at_recall_first_threshold"] = at("val")
m["test"]["at_recall_first_threshold"] = at("test")
m["test"]["score_range_note"] = {
    "val_max_good": val_good,
    "test_max_good": max(v["score"] for v in s.values() if v["split"] == "test" and v["category"] == "good"),
    "test_min_defective": min(v["score"] for v in s.values() if v["split"] == "test" and v["category"] != "good")}
p.write_text(json.dumps(m, indent=2))
print(json.dumps(m["test"]["at_recall_first_threshold"], indent=1))
print(m["test"]["score_range_note"])
