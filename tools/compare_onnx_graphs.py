"""Compare two ONNX files ignoring model-level metadata (e.g. export date). Usage: compare_onnx_graphs.py a.onnx b.onnx"""
import hashlib
import sys

import onnx

ms = [onnx.load(p) for p in sys.argv[1:3]]
for p, m in zip(sys.argv[1:3], ms):
    meta = {x.key: x.value for x in m.metadata_props}
    print(p, "| graph sha256:", hashlib.sha256(m.graph.SerializeToString()).hexdigest()[:16], "| date:", meta.get("date"))
same = ms[0].graph.SerializeToString() == ms[1].graph.SerializeToString()
print("GRAPHS IDENTICAL" if same else "GRAPHS DIFFER")
sys.exit(0 if same else 1)
