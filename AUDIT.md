# Adversarial technical audit

PASS requires execution or verification evidence, not just the existence of code. Audited 2026-10-25.

| Dimension | Status | Evidence / reason |
|---|---|---|
| Dataset downloaded | **PARTIAL** | Downloaded and extracted, but from a mirror: the official URL returned HTTP 404 (`data/raw/PROVENANCE.json`). |
| Dataset integrity | **PARTIAL** | All counts match MVTec's published numbers; 0 missing, invalid or orphan files; no duplicates (`reports/dataset_audit.md`). Byte-identity with the official archive is unverified. |
| Leakage-free split | **PASS** | Source-level, seed 42, SHA-256 frozen; disjointness asserted in tests; augmentation only inside training. |
| Annotation correctness | **PASS** | 28/28 tests, including full mask-pixel coverage on real data; visual QA panels inspected. |
| Model training | **PASS** (with disclosure) | Baseline, E2 and LOCO trained 150 epochs each; all three interrupted by low memory and resumed from checkpoints (workers 0). |
| Held-out metrics | **PASS** | Threshold chosen on val and enforced by code; COCO AP unit-tested; Ultralytics cross-check gap disclosed (0.856 vs 0.834). |
| Failure analysis | **PASS** | 15 real errors at the max-F1 point and 21 at E1, auto-classified; 5 cases written up; panels generated from predictions. |
| Unseen defect type (LOCO) | **PASS** | `scratch` held out: 3/23 flagged at the val threshold (0.51); 16/23 score ≤ 0.02, so not a threshold effect. Seen types on test: 20/22, 0/40 (REPORT §13a). One run, no seed repeats. |
| Improvement experiment | **PASS** (with disclosure) | E1 (threshold rule) improves escapes from 6 to 0; E2 (labels) is neutral to negative and reported. The test set informed the decision to revisit the threshold rule. No seed repeats. |
| ONNX export | **PASS** | `onnx.checker` passes; I/O contract in `models/*.onnx.json`. The shipped baseline file is a graph-identical re-export (disclosed). |
| ONNX parity | **PASS** | Raw and detection-level parity on 69 images for both models; ONNX test metrics identical to PyTorch. |
| CPU benchmark | **PASS** (local laptop only) | Ryzen 5 5600H, stage-separated, thread sweep, interleaved variant comparison. Not measured on an industrial controller. |
| Reproducibility | **PARTIAL** | Pinned versions, seeds, frozen manifests, one-command scripts. Training is not bit-reproducible after resume; the Colab notebook has **not** been executed on Colab; no seed repeats. |
| Industrial relevance | **PARTIAL** | Recall-first decision rule, REVIEW band, simulated controller messages, latency trade-offs, PatchCore complement. No real camera, line or controller. |
| Documentation | **PASS** | README, REPORT, written answers A–C, interview Q&A (27), spoken explanation, demo script, licence notes. |
| AI disclosure | **PASS** | `docs/AI_DISCLOSURE.md`: tool, scope and errors caught. |

## Defects found and corrected during the run
| # | Problem | File / experiment | Correction | Re-check |
|---|---|---|---|---|
| 1 | AP used a trapezoid-to-sentinel rule (perfect detector = 0.995) | `src/evaluation/metrics.py` | COCO 101-point rule | unit tests 2/2 that had failed now pass |
| 2 | ONNX parity used one tolerance across pixel and score units | `scripts/verify_onnx.py` | separate tolerances | parity re-run: pass |
| 3 | Fully-quantised INT8 output all-zero scores (mAP 0) | `scripts/deployment_variants.py` | decode head kept in FP32 | test mAP 0.851 |
| 4 | Benchmark contaminated by background load (189 ms vs 40.6 ms) | variant study | interleaved rounds, ranges reported | `compare_*.json` |
| 5 | Variant script moved `models/baseline.onnx` away | `scripts/deployment_variants.py` | export from a private copy | graph identity + parity + metrics re-run |
| 6 | Duplicate boxes mislabelled as "localisation" errors | `src/evaluation/failures.py` | `duplicate` category | galleries regenerated |
| 7 | Demo defaulted to the superseded max-F1 threshold | `scripts/run_inspection.py` | prefers the recall-first file, prints its source | demo: 29 REJECT / 40 PASS |
| 8 | JSONL sink crashed when its folder was missing | `controller_adapter.py` | create parent folder | demo run OK |
| 9 | Leave-one-out split would have reshuffled good images | `src/dataset/split.py` | consume RNG for the excluded class | 0 assignment differences |
| 10 | Post-training summary crashed: Windows Application Control blocked a pandas DLL (after all 150 LOCO epochs) | `src/training/train.py` | summary uses the `csv` module; `scripts/train.py --summarize-only` | `train_summary.json` written; device read from the run's `args.yaml` |
| 11 | `evaluate.py` crashed printing FP/normal-image on a split with no good images | `scripts/evaluate.py` | prints `n/a` | unseen evaluation re-run |

## Not done / open
* Seed repeats (`scripts/run_seeds.py` is ready).
* Synthetic robustness test (Challenge F); webcam and video inputs; real controller integration.
* Verification against the official MVTec archive (needs the user to fill MVTec's download form).
