# Two-minute screen recording: script

Record real execution only (terminal plus image viewer). Commands assume the project root with `.venv` active.

| Time | Show | Say (approximately) |
|---|---|---|
| 0–15 s | `examples/` image of a scratched nut next to a good nut | "A customer makes metal nuts. Some are bent, scratched, discoloured or flipped. They want a camera station that rejects bad parts, shows where the defect is, and runs on a small CPU with no cloud." |
| 15–35 s | `reports/annotation_qa/baseline_test_color_003.jpg`, then `bent_015.jpg` | "MVTec gives pixel masks, not boxes. I convert each connected mask region into a box. Here a single colour defect becomes seven boxes, and here one box sits inside another. That fragmentation matters later." |
| 35–60 s | Run `python scripts/run_inspection.py --source data/yolo/baseline/images/test --headless --save-dir artifacts/inspection --quiet`, then open two saved frames (one REJECT, one PASS) | "This is the ONNX model running on my laptop CPU: decision, boxes, confidence and processing time per part." |
| 60–85 s | `reports/failures/baseline_test/failures.md` threshold table, then panels `02_…` and `12_…` | "With the textbook max-F1 threshold, six of 29 defective test parts would pass. Eleven of the 15 missed boxes were found but scored below threshold. All four true misses were small fragments of a larger defect." |
| 85–105 s | Comparison table in `REPORT.md` (section 10) | "So I ran two controlled experiments. A recall-first threshold chosen on validation: 29 of 29 defective rejected, 0 of 40 good rejected, with honest confidence limits. And a relabelling that merges fragments: tidier boxes, but it missed one stained nut completely, so I did not adopt it." |
| 105–120 s | Run `python scripts/benchmark_cpu.py --onnx models/baseline.onnx --iters 100` | "ONNX parity is verified against PyTorch. On this Ryzen laptop the model takes about 40 to 60 milliseconds and full inspection about 56 to 78; INT8 brings that down about 1.8 times. It is a prototype, not production-ready: one dataset, 29 test defects." |
