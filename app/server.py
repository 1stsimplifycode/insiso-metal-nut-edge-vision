"""Web demo of the CPU inspection pipeline: upload an image -> PASS / REJECT / REVIEW + annotated frame.

    uvicorn app.server:app --host 0.0.0.0 --port 8000
Same detector, threshold and decision policy as scripts/run_inspection.py; only the I/O differs.
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_inspection import draw  # noqa: E402
from src.inference.onnx_detector import OnnxDetector  # noqa: E402
from src.inspection.decision import DecisionPolicy  # noqa: E402

ONNX = Path(os.environ.get("INSISO_ONNX", ROOT / "models" / "baseline.onnx"))
MODEL_ID = ONNX.stem
MAX_BYTES = 10 * 1024 * 1024


def _threshold() -> tuple[float, str]:
    # deployed rule = recall-first (E1); fall back to the max-F1 file, then to 0.25
    ev = ROOT / "artifacts" / "eval"
    for thr in (ev / f"{MODEL_ID}_zfr_threshold.json", ev / f"{MODEL_ID}_threshold.json"):
        if thr.exists():
            return json.loads(thr.read_text())["conf"], thr.name
    return 0.25, "fallback default"


REJECT_CONF, THRESHOLD_SOURCE = _threshold()
detector = OnnxDetector(str(ONNX), intra_threads=int(os.environ.get("INSISO_THREADS", "0")) or None)
RES = f"{detector.size}x{detector.size}"

app = FastAPI(title="InSiSo metal-nut inspection", version="1.0")


def inspect(img: np.ndarray, review_conf: float | None) -> dict:
    policy = DecisionPolicy(REJECT_CONF, review_conf)
    floor = min(REJECT_CONF, review_conf) if review_conf is not None else REJECT_CONF
    t0 = time.perf_counter()
    det, t = detector(img, conf=floor)
    decision = policy.decide(det)
    ms = (time.perf_counter() - t0) * 1e3
    shown = det[det[:, 4] >= REJECT_CONF] if decision == "REJECT" else det
    vis = draw(img, shown, decision, ms, MODEL_ID, RES)
    ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "decision": decision,
        "max_confidence": round(float(det[:, 4].max()), 4) if len(det) else 0.0,
        "reject_conf": REJECT_CONF,
        "review_conf": review_conf,
        "detections": [{"box_xyxy": [round(float(v), 1) for v in d[:4]], "confidence": round(float(d[4]), 4),
                        "class": "defect"} for d in shown],
        "processing_ms": round(ms, 2),
        "timings_ms": {"preprocess": round(t.preprocess_ms, 2), "inference": round(t.inference_ms, 2),
                       "postprocess": round(t.postprocess_ms, 2)},
        "model_version": MODEL_ID,
        "input_resolution": RES,
        "annotated_jpeg_base64": base64.b64encode(jpg.tobytes()).decode() if ok else None,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "reject_conf": REJECT_CONF, "threshold_source": THRESHOLD_SOURCE}


@app.post("/api/inspect")
async def api_inspect(file: UploadFile = File(...),
                      review_conf: float | None = Query(None, ge=0.0, le=1.0)) -> dict:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image (use PNG or JPEG)")
    return inspect(img, review_conf)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).with_name("index.html").read_text(encoding="utf-8")
            .replace("{{MODEL}}", MODEL_ID).replace("{{RES}}", RES)
            .replace("{{THR}}", f"{REJECT_CONF:g}"))
