"""CPU inspection demo: capture -> preprocess -> ONNX -> decode -> threshold -> decision.

    python scripts/run_inspection.py --source path/to/image.png
    python scripts/run_inspection.py --source data/yolo/baseline/images/test --headless --save-dir artifacts/inspection
    python scripts/run_inspection.py --source video.mp4
    python scripts/run_inspection.py --source 0          # webcam index
Results are also emitted as JSON lines through the simulated controller adapter.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.onnx_detector import OnnxDetector  # noqa: E402
from src.inspection.controller_adapter import InspectionMessage, JsonlSink  # noqa: E402
from src.inspection.decision import DecisionPolicy  # noqa: E402

COLOURS = {"PASS": (0, 170, 0), "REJECT": (0, 0, 230), "REVIEW": (0, 170, 255)}


def frames(source: str):
    p = Path(source)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        if not cap.isOpened():
            sys.exit(f"cannot open camera {source}")
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f"cam{source}_{i:06d}", f
            i += 1
    elif p.is_dir():
        for f in sorted(list(p.glob("*.png")) + list(p.glob("*.jpg"))):
            yield f.stem, cv2.imread(str(f))
    elif p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(str(p))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f"{p.stem}_{i:06d}", f
            i += 1
    else:
        yield p.stem, cv2.imread(str(p))


def draw(img, det, decision, ms, model_id, res):
    vis = img.copy()
    for x1, y1, x2, y2, s, _ in det:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(vis, f"defect {s:.2f}", (int(x1), max(15, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2, cv2.LINE_AA)
    h = 64
    bar = np.full((h, vis.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, decision, (10, 44), cv2.FONT_HERSHEY_DUPLEX, 1.3, COLOURS[decision], 2, cv2.LINE_AA)
    cv2.putText(bar, f"{ms:.1f} ms | {model_id} | in {res}", (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--onnx", default=str(ROOT / "models" / "baseline.onnx"))
    ap.add_argument("--reject-conf", type=float, default=None, help="default: val-selected threshold for the model")
    ap.add_argument("--review-conf", type=float, default=None, help="optional REVIEW band lower bound")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--jsonl", default=None, help="append controller messages to this file")
    ap.add_argument("--quiet", action="store_true", help="do not echo JSON messages to stdout")
    a = ap.parse_args()

    model_id = Path(a.onnx).stem
    reject = a.reject_conf
    source = "--reject-conf"
    if reject is None:
        # deployed rule = recall-first (E1); fall back to the max-F1 file, then to 0.25
        ev = ROOT / "artifacts" / "eval"
        for thr in (ev / f"{model_id}_zfr_threshold.json", ev / f"{model_id}_threshold.json"):
            if thr.exists():
                reject, source = json.loads(thr.read_text())["conf"], thr.name
                break
        else:
            reject, source = 0.25, "fallback default"
    print(f"[config] model={model_id} reject_conf={reject} ({source}) review_conf={a.review_conf}", file=sys.stderr)
    policy = DecisionPolicy(reject, a.review_conf)
    det_model = OnnxDetector(a.onnx)
    res = f"{det_model.size}x{det_model.size}"
    sink = JsonlSink(Path(a.jsonl) if a.jsonl else None, None if a.quiet else sys.stdout)
    save = Path(a.save_dir) if a.save_dir else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    floor = min(reject, a.review_conf) if a.review_conf is not None else reject
    tally = {"PASS": 0, "REJECT": 0, "REVIEW": 0}
    times = []
    for sid, img in frames(a.source):
        if img is None:
            continue
        t0 = time.perf_counter()
        det, _ = det_model(img, conf=floor)
        decision = policy.decide(det)
        ms = (time.perf_counter() - t0) * 1e3
        times.append(ms)
        tally[decision] += 1
        shown = det[det[:, 4] >= reject] if decision == "REJECT" else det
        sink.send(InspectionMessage(
            sample_id=sid, decision=decision, max_confidence=float(det[:, 4].max()) if len(det) else 0.0,
            detections=[{"box_xyxy": [round(float(v), 1) for v in d[:4]], "confidence": round(float(d[4]), 4),
                         "class": "defect"} for d in shown],
            processing_ms=round(ms, 2), model_version=model_id, input_resolution=res))
        vis = draw(img, shown, decision, ms, model_id, res)
        if save:
            cv2.imwrite(str(save / f"{sid}_{decision}.jpg"), vis)
        if not a.headless:
            try:
                cv2.imshow("InSiSo metal-nut inspection demo (q to quit)", vis)
                if cv2.waitKey(0 if Path(a.source).is_file() else 1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                print("[warn] no GUI support in this OpenCV build (opencv-python-headless); continuing headless. "
                      "Use --save-dir to keep annotated frames, or install opencv-python for a window.",
                      file=sys.stderr)
                a.headless = True
    sink.close()
    if times:
        print(f"[summary] frames={len(times)} {tally}  preprocess+infer+decode mean {np.mean(times):.1f} ms "
              f"(excludes image loading)", file=sys.stderr)


if __name__ == "__main__":
    main()
