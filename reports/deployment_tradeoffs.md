# Deployment trade-offs (one checkpoint, local CPU)

Threshold per variant chosen on **val** (lowest conf with zero val false rejects), then applied to test (29 defective / 40 good images). Latency: ONNX Runtime CPU EP, batch 1, default threads, 150 iterations after warm-up. End-to-end = PNG load + preprocess + model + postprocess.

| variant | input | ONNX MB | conf | test mAP50 | mAP50-95 | box P | box R | defective flagged | good rejected | model ms mean / p95 | end-to-end ms mean / p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_640 | 640x640 | 10.17 | 0.11 | 0.856 | 0.554 | 0.694 | 0.850 | 29/29 | 0/40 | 188.9 / 326.0 | 235.7 / 371.8 |
| baseline_480 | 480x480 | 10.1 | 0.47 | 0.824 | 0.550 | 0.889 | 0.800 | 28/29 | 1/40 | 66.3 / 94.5 | 99.1 / 138.3 |
| baseline_320 | 320x320 | 10.05 | 0.11 | 0.781 | 0.491 | 0.756 | 0.775 | 28/29 | 1/40 | 18.7 / 28.2 | 36.5 / 49.8 |
| baseline_640_int8 | 640x640 | 3.05 | 0.01 | 0.000 | 0.000 | nan | 0.000 | 0/29 | 0/40 | 110.2 / 185.7 | 159.2 / 261.6 |
