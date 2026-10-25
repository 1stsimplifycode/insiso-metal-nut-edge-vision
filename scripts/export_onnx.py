"""Export a trained checkpoint to ONNX and record its I/O contract.

    python scripts/export_onnx.py --weights models/baseline.pt --imgsz 640
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    import ultralytics
    from ultralytics import YOLO

    w = Path(a.weights)
    out = Path(YOLO(str(w)).export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=a.simplify,
                                   dynamic=False, nms=False, half=False, device="cpu"))
    dst = ROOT / "models" / f"{w.stem}.onnx"
    if out.resolve() != dst.resolve():
        shutil.move(str(out), dst)
    m = onnx.load(str(dst))
    onnx.checker.check_model(m)
    dims = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]  # noqa: E731
    sess = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    info = {
        "onnx_path": str(dst.relative_to(ROOT)), "source_checkpoint": str(w),
        "checkpoint_sha256": hashlib.sha256(w.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "onnx_bytes": dst.stat().st_size,
        "export_call": f"YOLO('{w.as_posix()}').export(format='onnx', imgsz={a.imgsz}, opset={a.opset}, "
                       f"simplify={a.simplify}, dynamic=False, nms=False, half=False)",
        "opset": [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0],
        "inputs": [{"name": i.name, "shape": dims(i), "dtype": "float32"} for i in m.graph.input],
        "outputs": [{"name": o.name, "shape": dims(o)} for o in m.graph.output],
        "metadata": {p.key: p.value for p in m.metadata_props},
        "ort_providers": sess.get_providers(),
        "preprocessing": "BGR->RGB, letterbox to imgsz x imgsz (aspect kept, pad value 114, centred), "
                         "HWC->NCHW, float32 / 255",
        "output_layout": "(1, 4+nc, N): rows 0-3 = cx,cy,w,h in letterboxed pixels; rows 4.. = per-class sigmoid scores",
        "postprocessing": "not in graph: confidence filter, class-aware NMS (IoU 0.7), un-letterbox — "
                          "see src/inference/onnx_detector.py",
        "ultralytics": ultralytics.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__,
    }
    dst.with_suffix(".onnx.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
