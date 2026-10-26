# Industrial Metal-Nut Defect Detection: Reproducible Localisation, Failure Analysis and CPU Edge Deployment

**Adishree Gupta** · InSiSo Technologies Computer Vision Intern take-home · October 2026
Code and artifacts: this repository. Every number below is read from a file named in brackets and was produced
by the scripts in `scripts/`. Nothing was estimated or hand-entered.

---

## 1. Problem
A metal-nut manufacturer needs a camera station that **rejects defective parts** (bent, scratched,
discoloured, flipped), **shows where** the defect is, and runs **on a small CPU without cloud inference**.
The costly error is an escaped defect (a defective part that passes), so I evaluate the system first as a
reject/pass decision and only second as a box localiser.

## 2. Dataset
MVTec AD, category `metal_nut` (CC BY-NC-SA 4.0, not redistributed). 335 RGB images of 700×700 px:
242 good (220 in MVTec's `train/`, 22 in `test/`) and 93 defective (bent 25, color 22, flip 23, scratch 23),
each defective image with a binary pixel mask.

**Acquisition (deviation, disclosed).** The official category URL returned HTTP 404 for the whole share on
2026-10-25, and MVTec's page now places downloads behind a form. The automatic downloader
(`scripts/download_dataset.py`: resumable, full-stream archive validation, path-traversal guard, provenance
record) therefore fell back to a pinned Hugging Face mirror. All 10 folder counts match MVTec's published
counts and MVTec's own `license.txt`/`readme.txt` are present, but **byte-identity with the official archive
is unverified** (`--verify-archive` performs a per-file SHA-256 comparison once the archive is obtained).
[`data/raw/PROVENANCE.json`]

## 3. Dataset integrity [`reports/dataset_audit.md`]
* All 335 images are 700×700×3 and all 93 masks are 700×700, binary {0, 255}, and non-empty. Every
  defective image has exactly one mask, and there are no orphan masks.
* SHA-256 exact duplicates: none. 256-bit dHash near-duplicates (Hamming ≤ 4): none; minimum pairwise
  distance 19.
* Class imbalance: 2.6 good images per defective image; 11–12 training images per defect type.
* Mask area as a fraction of the image (median): bent 1.9%, color 2.1%, scratch 5.5%, **flip 48%**. A flipped nut
  is annotated as the whole part, so for `flip` the box is the object, and the task is really classification.

## 4. Mask-to-box conversion [`src/annotations/mask_to_boxes.py`, `reports/annotation_qa/`]
Binarise (>127) → 8-connected components (OpenCV) → one tight box per component → YOLO format, with
coordinate validation. **No minimum-area filter** (0 of 132 components dropped; the smallest is 106 px). Good
images get empty label files. 15 annotation and split tests pass, including "every mask pixel lies inside a
box" (28 tests in total with the metric and inference tests; `pytest -q tests`).

**Finding:** 24 of 93 masks are fragmented: 2–7 components per mask, sometimes nested (a 125 px speck
inside a 16,131 px region, `bent/015`) or scattered (7 colour specks, `color/003`). One-box-per-component is
faithful to the mask but not necessarily to what an inspector calls "one defect". This became experiment E2.

## 5. Model choice
**YOLO11n** (Ultralytics 8.4.188): 2.6 M parameters, 6.4 GFLOPs at 640², 5.3 MB checkpoint, 10.2 MB FP32
ONNX, COCO-pretrained, mature ONNX export.

* **Single class (`defect`), not four.** There are only 11–12 training images per type, and the customer's
  decision is binary. Per-type performance is still reported from each ground-truth box's category.
* **640 input**, close to the native 700 px, so small scratches are not shrunk further.
* **Not larger.** With 45 defective training images, the bottleneck is data and labels, not capacity.
  A larger model would cost latency, which is the scarce resource on a controller (§12).
* **Licensing.** Ultralytics is AGPL-3.0, which matters for commercial deployment (see LICENSE_NOTES.md).
  The deployed inference path itself uses only ONNX Runtime, OpenCV and NumPy.

## 6. Experimental design
**Adapting MVTec's protocol.** MVTec AD is an *unsupervised* benchmark (train on good images only). A
supervised detector needs defect images in training, so defect images are re-split **per defect type**
(≈50/20/30%) and all good images are pooled (172/30/40). The split uses seed 42, is at source-image level, and is
frozen by SHA-256. Tests assert that the splits are disjoint. [`manifests/split_seed42.csv`]
**These results are not comparable with the MVTec AD leaderboard.**

| | bent | color | flip | scratch | defective | good |
|---|---|---|---|---|---|---|
| train | 12 | 11 | 11 | 11 | 45 | 172 |
| val | 5 | 4 | 5 | 5 | 19 | 30 |
| **test** | 8 | 7 | 7 | 7 | **29** | **40** |

**Rules:**

* The checkpoint is selected on val (Ultralytics fitness = 0.1·mAP50 + 0.9·mAP50-95).
* The operating threshold is chosen on val, saved, and only then applied to test. `evaluate.py` refuses to
  score test without a saved val threshold.
* One factor is changed per experiment.

**Metrics** [`src/evaluation/metrics.py`, unit-tested]:

* Box matching is greedy and one-to-one by confidence at IoU ≥ 0.5.
* AP is COCO 101-point interpolation, computed over all predictions with conf ≥ 0.001. It ranks, so it is
  threshold-free.
* Precision and recall are reported at the stated operating threshold.
* Image level: a part is "flagged" if any box is ≥ threshold. **Sensitivity** = defective parts flagged;
  **false-reject rate** = good parts flagged.
* Intervals are exact Clopper-Pearson 95% intervals [`tools/binomial_ci.py`].

**Acquisition correlation.** All images come from one setup. A random split cannot reveal camera, lighting or
batch changes, so these numbers are an *upper bound* on field performance.

## 7. Baseline training [`runs/train/baseline/`, `configs/baseline.yaml`]
* Setup: AdamW, lr 0.001, weight decay 5e-4, batch 16, 150 epochs, patience 40, seed 0, Ultralytics-default
  augmentation (HSV, translate, scale, mosaic, horizontal flip).
* Hardware: GTX 1650 (4 GB), about 45–50 min in total.
* **Selected epoch: 117.**
* **Interruption (disclosed):** the OS ran short of memory at epoch 65 and the process was stopped. Training
  resumed from `last.pt` (optimiser state preserved) with dataloader workers 2 → 0. This can change the
  augmentation random stream, so the run is not bit-identical to an uninterrupted one.
* Validation (Ultralytics) at epoch 117: P 0.80, R 0.70, mAP50 0.76, mAP50-95 0.55.

## 8. Held-out evaluation [`artifacts/eval/baseline_test_metrics.json`]
Threshold from val by the textbook rule, **maximum box-F1 → conf 0.73**. Test set: 69 images (29 defective,
40 good), 40 ground-truth boxes.

| Metric | Value |
|---|---|
| **Precision (box, @0.73)** | **1.000** (25 TP / 25 predictions) |
| **Recall (box, @0.73)** | **0.625** (25 / 40 GT boxes) |
| **mAP@0.5** | **0.856** |
| mAP@0.5:0.95 | 0.554 |
| F1 (box) | 0.769 |
| Defective parts flagged | 23/29 = 0.793 [0.603, 0.920] |
| Good parts falsely rejected | 0/40 = 0.000 [0.000, 0.088] |
| False-positive boxes per good image | 0.000 |
| Box recall by type | bent 6/13, color 6/9, flip 7/7, scratch 6/11 |
| Box recall by size (√area) | <50 px 0/4, 50–150 px 13/18, ≥150 px 12/18 |

*Cross-check:* Ultralytics' validator on the same split reports mAP50 0.834 and mAP50-95 0.532. The
difference comes from AP interpolation (its trapezoid-to-sentinel rule gives a perfect detector 0.995) and
its rectangular batching. I report my COCO-rule value and disclose the gap.

## 9. Failure analysis [`reports/failures/baseline_test/`, `reports/failures/baseline_zfr_test/`]
At the max-F1 threshold, the 15 box errors were **all missed boxes; there were 0 false positives**. An
automatic classifier labelled them: 11 *below threshold* (a correctly placed box existed, IoU ≥ 0.5, but its
confidence was < 0.73) and 4 *missed* (no box of any confidence matched). At image level, **6 defective
parts would have passed**. Five representative cases follow, each written as observed fact → hypothesis →
test → outcome.

**Case 1, `scratch_000` (defective part passed).**
- *Fact:* scratch 171×140 px; best box confidence 0.33, far below 0.73.
- *Hypothesis:* the model detects the scratch but is under-confident on this texture.
- *Test:* E1 (§10). *Outcome:* flagged at 0.11.

**Case 2, `bent_010` (passed, plus 2 hard misses).**
- *Fact:* the main bent region (155×184) was detected at 0.55, below threshold. Two further mask fragments
  (38×94 with local contrast 1.5 grey levels, and 11×19) lie *inside* the main region's box and have no
  matching prediction at any confidence (panel `02_test_bent_010_FN.jpg`).
- *Hypothesis:* the nested fragments are an artefact of the per-component annotation policy, not a missed
  defect.
- *Test:* E2 (§10). *Outcome:* the part is flagged at the E1 threshold; under E2's merged labels the
  fragments become part of one box.

**Case 3, `color_004` (passed).**
- *Fact:* colour defect 77×112 detected at 0.68, below threshold.
- *Hypothesis:* under-confidence (operating point).
- *Test:* E1. *Outcome:* flagged. But the **E2 model scores this part 0.001**, a genuine regression,
  so the colour type is sensitive to how fragments are labelled.

**Case 4, `color_007` (hard miss, part still rejected).**
- *Fact:* a 10×12 px speck next to a large colour blob that *was* detected at 0.91 (panel
  `08_test_color_007_FN.jpg`).
- *Interpretation:* a box-metric error with no decision consequence, since the part is rejected anyway.

**Case 5, `scratch_012` (passed, plus 1 hard miss).**
- *Fact:* the large scratch (316×504) was detected at 0.40; an adjacent 51×105 fragment was predicted at 0.47
  with IoU 0.08 against its own box, because the model drew one box spanning several fragments (panel
  `12_test_scratch_012_FN.jpg`).
- *Hypothesis:* fragmentation plus under-confidence. *Test:* E1 and E2.

**Supporting observations.**
* On **validation**, the highest score on any good part was **0.10** and the lowest top score on any defective
  part was **0.63**. The ranking separates the classes; the max-F1 rule placed the threshold *inside* the
  defective range.
* All 4 hard misses are secondary fragments of multi-component masks.
* Small boxes (< 50 px side) are 0/4. Resolution limitation and fragment ambiguity are confounded here; the
  sample is too small to separate them.
* `flip` is 7/7 (whole-object box). I had suspected that the default mirror augmentation might confuse flipped
  parts (a metal nut is chiral). **The data do not support that suspicion.**
* Not observed: false alarms on good parts, background confusion, reflections causing false positives.
* Not separable with 29 test defects: per-type difficulty ranking, low contrast vs size.

## 10. Controlled improvement experiments
**E1 — operating-point rule (no retraining).** For inspection, use a **recall-first rule: the lowest confidence
at which no validation good part is flagged** (val → 0.11).
*Disclosure:* I decided to revisit the rule after seeing the test failures above. The rule itself uses only
validation data, but the test set has now informed one decision, so the E1 test numbers are mildly optimistic.

**E2 — annotation policy (retrain).** Merge mask components within 20 px into one box (132 → 106 boxes); the
model, hyper-parameters, split and seed are identical [`configs/improvement.yaml`]. This also stopped once for
low memory (epoch 41) and was resumed; workers were 0 throughout. Selected epoch 120.

All rows: same 69 test images; thresholds from val.

| Model / rule | conf | Defective flagged | Good rejected | Box P | Box R | mAP50 (component GT) | mAP50 / 50-95 (merged GT) |
|---|---|---|---|---|---|---|---|
| Baseline, max-F1 | 0.73 | 23/29 | 0/40 | 1.000 | 0.625 | 0.856 | — |
| **Baseline, E1 recall-first** | **0.11** | **29/29** [0.881, 1] | **0/40** [0, 0.088] | 0.694 | 0.850 | 0.856 | 0.877 / 0.578 |
| E2 merged labels, max-F1 | 0.19 | 27/29 | 0/40 | 0.861 | 0.775 | 0.798 | — |
| E2 merged labels, recall-first | 0.05 | 28/29 [0.822, 0.999] | 0/40 | 0.744 | 0.800 | 0.798 | 0.879 / 0.602 |

**Interpretation.**
* **E1 is worthwhile.** Escaped defects fall from 6 to 0 with no false rejects. mAP is unchanged, as it must
  be, because a threshold only moves the operating point on the same ranking. The cost is lower box precision
  (0.694): 15 false-positive boxes, all on defective parts (6 duplicates that survived NMS at IoU 0.7,
  6 fragment mismatches, 2 extra boxes on defective parts, 1 localisation). None of these changes a decision.
  On test, good parts scored ≤ 0.024 and defective parts ≥ 0.332, so the threshold sits in a wide gap here;
  on validation the gap was 0.10–0.63. **The margin to the highest good-part score is the number to monitor
  in production.**
* **E2 is not adopted.** It did what it targeted at box level: box precision 0.694 → 0.744 and slightly higher
  mAP50-95 against merged labels. But it **lost one defective part entirely** (`color_004`, score 0.001), and
  against the original per-component labels its mAP50 falls to 0.798, partly by construction since it predicts
  merged boxes. The one-part difference (29 vs 28 of 29) is within noise. The honest conclusion: relabelling
  fragments did not improve the inspection decision. Fragment ambiguity is mostly a **metric** problem, not a
  **detection** problem.
* Not done: seed repeats. Each training run takes about 45 min and was twice stopped by low memory on this
  laptop. `scripts/run_seeds.py` is ready; without it, single-image differences between models must not be read
  as real effects.

## 11. ONNX export [`models/baseline.onnx`, `models/baseline.onnx.json`, `artifacts/onnx_parity/baseline.json`]
* *Disclosed:* the original `baseline.onnx` was accidentally moved away by the variant script (bug since
  fixed). The file shipped is a re-export. ONNX export embeds a timestamp, so file hashes differ, but the graph
  is byte-identical to the 640 variant used in §12 (`tools/compare_onnx_graphs.py`). Parity and test metrics
  were re-run on the shipped file with identical results.
* **E2 model.** `models/improvement.onnx` passes the same parity test (43 matched detections, max score Δ 1.7e-6).
* **Source and export.** Exported from `models/baseline.pt` (epoch 117) with
  `YOLO(...).export(format='onnx', imgsz=640, opset=17, simplify=True, dynamic=False, nms=False)`.
* **I/O.** Input `images` float32 [1,3,640,640], RGB, /255, letterboxed (aspect kept, pad 114). Output
  `output0` [1,5,8400]: cx, cy, w, h in letterbox pixels plus one sigmoid score.
* **Postprocessing outside the graph.** Confidence filter, NMS (IoU 0.7) and un-letterboxing are implemented
  in NumPy [`src/inference/onnx_detector.py`].
* **Parity on all 69 test images (CPU EP).**
  - Raw outputs: box rows max |Δ| 0.0033 px; score rows max |Δ| 1.0e-5.
  - Final detections vs Ultralytics' PyTorch pipeline: identical counts; max score difference 1.0e-6; max box
    difference 0.0002 px.
  - Re-evaluating the ONNX model end-to-end reproduces the PyTorch test metrics exactly.
* *Disclosed:* the first parity run applied one 1e-3 tolerance to pixel- and score-valued rows together and
  "failed" on a 0.0033 px difference. I fixed the unit error with separate tolerances (0.01 px, 1e-4).

## 12. CPU benchmarking (local laptop) [`artifacts/benchmarks/`]
**Hardware and settings:** AMD Ryzen 5 5600H (6 cores / 12 threads), Windows 11, Python 3.12.10, ONNX
Runtime 1.31.0, CPU EP, batch 1, source image 700×700 PNG, model input 640×640. Warm-up 20 iterations, then
200 timed iterations cycling over the 69 test images. The laptop was on AC power with the Balanced plan and
normal background load.

| Stage (isolated run, ORT default threads) | mean ms | p95 ms |
|---|---|---|
| Image load (PNG decode) | 10.5 | — |
| Preprocess (letterbox, normalise) | 4.7 | — |
| **Model execution** | **40.6** | **48.9** |
| Postprocess (decode + NMS) | 0.3 | — |
| **End-to-end inspection** | **56.1** | **66.5** |

Thread sweep (model mean ms): 1 thread → 82.9, 2 → 62.8, 4 → 49.7, 6 → 44.6, ORT default → 40.6.
Model-only throughput is about 24.6 images/s; end-to-end about 17.8 images/s.

**Deployment variants** [`reports/deployment_tradeoffs.md`, `artifacts/benchmarks/compare_*.json`]: same
checkpoint, threshold re-chosen on val for each variant, latency from an *interleaved* benchmark
(5 rounds × 60 iterations, models alternated so background load hits all equally).

| Variant | ONNX MB | Model ms, median (round range) | End-to-end ms | Test mAP50 | Defective flagged | Good rejected |
|---|---|---|---|---|---|---|
| FP32 640 | 10.2 | 58.6 (45.1–68.6) | 77.6 | 0.856 | 29/29 | 0/40 |
| **INT8 640** | **3.1** | **33.3** (30.5–36.1) | **51.2** | 0.851 | 29/29 | 0/40 |
| FP32 480 | 10.1 | 25.2 (24.8–27.2) | 40.1 | 0.824 | 28/29 | 1/40 |
| FP32 320 | 10.1 | 10.7 (10.3–11.6) | 24.0 | 0.781 | 28/29 | 1/40 |

**Notes and disclosures:**

* The FP32 640 model varies most between rounds (45–69 ms), against 40.6 ms in the isolated run. The CPU has
  turbo boost and the laptop had background load, so I quote both figures. A first, non-interleaved variant
  run produced 189 ms for the same model; I discarded it as contaminated and re-measured.
* **INT8:** static, QDQ, per-channel, calibrated on 64 training images. The first attempt quantised the whole
  graph and produced **mAP 0**: the output tensor concatenates 0–640 px boxes with 0–1 scores, so a shared
  8-bit scale zeroed every score. Keeping the detection-head decode in FP32 fixed it. The INT8 speed-up
  (≈1.8×) is specific to this CPU and must be re-measured on the target controller.
* Batching was not used. It raises throughput, not single-part latency.

## 13. Industrial implications
* **The decision rule matters more than the architecture here.** The same model escapes 6 defective parts or
  none, depending on a validation-chosen threshold policy. A deployment should fix the policy by the cost of
  an escaped defect, then monitor the margin between good-part scores and the threshold.
* **Normal-only anomaly detection is a strong complement** [`artifacts/anomaly/patchcore_metrics.json`].
  A PatchCore baseline (WideResNet-50 features, memory bank from the 172 training good parts only; no defect
  images at all) reaches **image AUROC 0.996 on test**. With the same recall-first rule it flags **29/29**
  defective parts but **rejects 4/40 good parts** (10% [2.8%, 23.7%]), because some good test parts score
  above every good validation part (2.28 vs 2.07). It produces heatmaps (`reports/anomaly_heatmaps/`), not
  boxes, so only image-level metrics are comparable. Its strength is coverage of defect types never labelled
  (§13a); its weakness here is false rejects.
* **Inspection demo:** `scripts/run_inspection.py` was run headless on the 69 test images with the ONNX model
  and the deployed threshold: 29 REJECT, 40 PASS, 0 REVIEW, which agrees with the evaluation pipeline.
  Every frame was emitted as a JSON message through the **simulated** controller adapter
  [`artifacts/inspection/messages.jsonl`]. Webcam and video inputs are implemented but **not tested** (no
  camera available).
* **Latency:** 10 ms is not met end-to-end by any variant on this CPU. The model alone approaches it only at
  320 px, at an accuracy cost (written answer C).

### 13a. Unseen defect type (leave `scratch` out) [`artifacts/eval/loco_scratch_*`]
**Status: completed** (150 epochs; interrupted once by low memory at about epoch 25 and resumed from
`last.pt` with workers 0, as for the other runs).

* **Protocol, fixed before training.** All 23 `scratch` images are removed from train/val and held out as an
  `unseen` split. Every other image keeps exactly its main-split assignment (verified: 0 differences)
  [`manifests/split_seed42_loco_scratch.csv`]. Same config as the baseline. Checkpoint selected on val
  (epoch 132); threshold chosen on val (which contains no scratch) by the recall-first rule.
* **Threshold.** The rule gives **0.51**, not 0.11 as for the baseline, because one good validation part
  scored 0.509 with this model. The safety margin is again one score step.

| LOCO model (no scratch in training), threshold 0.51 from val | Defective rejected | Good rejected | Box P / R | mAP@0.5 |
|---|---|---|---|---|
| val (bent, color, flip + good) | 11/14 | 0/30 | 0.833 / 0.588 | 0.703 |
| test, seen types (bent, color, flip + good) | 20/22 (91% [71%, 99%]) | 0/40 ([0%, 8.8%]) | 0.905 / 0.655 | 0.740 |
| **unseen: all 23 scratch images** | **3/23 (13% [2.8%, 34%])** | — | 1.000 / 0.068 | 0.216 |

* **Finding: the supervised detector does not transfer to a defect type it never saw.** Only 3 of 23 scratched
  nuts would be rejected; 41 of 44 scratch boxes are missed.
* **It is not a threshold problem.** Descriptively (not used for any choice), 16 of 23 scratch images have a
  top score ≤ 0.02, and 10 have no prediction at all above 0.001; even at 0.05 only 7/23 would be flagged
  [`artifacts/eval/loco_scratch_unseen_score_profile.json`, from `tools/loco_score_profile.py`]. The model
  learned "what bent, discoloured and flipped nuts look like", not "what is unlike a good nut".
* **Contrast with the normal-only baseline.** PatchCore never sees any defect label, so every defect type is
  unseen for it. It flagged 7/7 scratch images in the main test split (§13). It was not run on the 16
  scratch images that sit in the main train/val splits, so the two numbers are not on identical images.
* **Seen types are also weaker** (20/22 vs 22/22 for the baseline on the same non-scratch test images at its
  own threshold). The LOCO model trained on 34 instead of 45 defect images and has a higher val threshold;
  with one run per configuration this difference is not significant.

What it can and cannot show: on the same parts, camera and lighting, a supervised detector misses most of a
defect type absent from training. It does **not** show anything about new parts or stations. This is the
measured basis for written answer A's recommendation to use a normal-only gate as the primary reject trigger.

## 14. Limitations
* **Data.** One public dataset and one imaging setup, so nothing is known about other cameras, lighting,
  part batches or truly unseen parts. The supervised split adapts an unsupervised benchmark.
* **Statistics.** 29 defective test images: a 29/29 result is compatible with true sensitivity as low as 88%.
  There are no seed repeats, so differences of 1–2 images between models are not significant.
* **Protocol.** The test set informed one decision (revisiting the threshold rule).
* **Provenance and reproducibility.** The data came from a mirror and byte-identity is unverified. Both
  training runs were interrupted and resumed.
* **Latency.** Measured on one laptop CPU under background load, not on an industrial controller.
* **Integration and licensing.** The controller adapter is simulated (`src/inspection/controller_adapter.py`)
  and untested on any PLC or Bosch Rexroth ctrlX hardware. AGPL-3.0 implications apply to the
  Ultralytics-trained weights.
* **Not production-ready.**

## 15. Written answers
See **[docs/written_answers.md](docs/written_answers.md)**: (A) 30 defective vs 3,000 good images; (B) misses two
weeks after go-live; (C) 10 ms budget vs 40 ms model, including the measured trade-off table above.

## 16. AI-assistance disclosure
This project was built with **Claude Code** (an AI coding agent) under my direction. The AI wrote essentially
all code, ran the experiments on my laptop, and drafted the documentation including this report. I defined the
requirements and constraints, handled the environment, and am responsible for reviewing every claim. Several
errors made during the run were caught by tests or checks and are disclosed above: the AP definition, the
parity tolerance units, the fully-quantised INT8 model, a contaminated benchmark, and a mislabelled
failure category. Details are in [docs/AI_DISCLOSURE.md](docs/AI_DISCLOSURE.md).

---
### Appendix: artifact index
`reports/dataset_audit.md` · `manifests/split_seed42*.csv(+.sha256)` · `reports/annotation_qa/` ·
`artifacts/eval/*_metrics.json` (all metrics, per image) · `artifacts/eval/*_threshold.json` (val-chosen thresholds) ·
`reports/failures/*/failures.md` (galleries + threshold sweeps) · `artifacts/onnx_parity/baseline.json` ·
`artifacts/benchmarks/*.json` · `reports/deployment_tradeoffs.md` · `artifacts/anomaly/` · `runs/train/*/train_summary.json` (local only)
