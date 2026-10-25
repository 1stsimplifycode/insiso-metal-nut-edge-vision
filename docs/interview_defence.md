# Interview defence: questions and answers

All numbers refer to artifacts in this repository (`artifacts/`, `reports/`). Test set = 69 images
(29 defective, 40 good), frozen in `manifests/split_seed42.csv`.

## Data and protocol

**1. MVTec AD is an unsupervised benchmark. How did you turn it into a supervised detection task, and what does that cost you?**
Defect images only exist in MVTec's `test/` folder, so I re-partitioned them per defect type into
train/val/test (≈50/20/30%), and pooled all 242 good images, then split them too. The cost: results are not
comparable with the MVTec leaderboard (which trains on good images only and tests on all of `test/`), and the
test set is small, so every rate has a wide confidence interval.

**2. How do you know there is no leakage between splits?**
The split is done at source-image level, with one row per original file and a SHA-256-frozen manifest. A test
asserts the three sets of file paths are disjoint. Labels are generated *after* splitting, and augmentation
happens on the fly inside training only. I also checked exact duplicates (SHA-256: none) and near-duplicates
(256-bit dHash: minimum pairwise Hamming distance 19, so no near-copies).

**3. What leakage risk remains?**
Acquisition correlation: all images come from one camera setup, one lighting rig, and probably one batch
of nuts. A model can look excellent on this split and still fail on a new line. The split cannot detect that;
only data from a different session or station can.

**4. Did you ever use the test set for a decision?**
Yes, once, and I disclose it. After the first test evaluation I saw that the max-F1 threshold rule let 6
defective parts pass. That prompted me to change the *rule* to "lowest confidence with zero false rejects on
validation". The rule uses validation data only, but the decision to revisit it was informed by test failures,
so the test numbers after that change are mildly optimistic.

**5. Why is the dataset from a Hugging Face mirror?**
The official per-category link returned 404 for the whole share on 2026-10-25, and MVTec now uses a request
form. The mirror's folder counts all match the published MVTec counts and it contains MVTec's own licence
file, but byte-identity is unverified. `download_dataset.py --verify-archive` checks it once the official
archive is obtained. The mirror labels itself MIT, which is wrong: the data are CC BY-NC-SA 4.0.

## Labels

**6. How are boxes generated from masks?**
Binarise at >127, take 8-connected components with OpenCV, and use one tight box per component, with no
minimum-area filter (none of the 132 components was dropped). Tests check that every mask pixel lies inside
some box, that normal images get empty label files, and that all YOLO coordinates are in range.

**7. What is wrong with per-component boxes?**
24 of 93 masks are fragmented. One annotated defect region can become 2–7 boxes, sometimes nested (a speck
box inside a large box). A detector usually predicts one box for the region, which IoU-matching then scores
as "missed fragments" plus "poorly localised". All 4 hard misses of the baseline were such secondary fragments.

**8. Why are `flip` boxes so large?**
A flipped nut is a whole-object defect: its mask covers ≈48% of the image, so the box is essentially the
whole part. That makes `flip` easy for the detector (7/7) but means its "localisation" carries no information.
It is really a classification problem.

## Model and training

**9. Why YOLO11n?**
2.6 M parameters, 6.4 GFLOPs, 5.3 MB checkpoint, mature ONNX export, and COCO pretraining (important with
45 defect images). Bigger models would mainly add latency. With this little data, the limits are data and
labels, not capacity.

**10. Why one class, not four?**
With 11–12 training images per defect type, a 4-class head would learn class boundaries from a handful of
examples, and the customer's decision is binary (reject or pass) anyway. I still report per-type recall
using the ground-truth category of each box.

**11. How was the checkpoint selected?**
By Ultralytics' validation fitness (0.1·mAP50 + 0.9·mAP50-95) on the validation split only. That was epoch 117
of 150 for the baseline. Early stopping (patience 40) did not trigger.

**12. The training run was interrupted. Is that a problem?**
The OS ran low on memory at epoch 65. I resumed from `last.pt` (optimiser state included) with dataloader
workers 2→0 to save RAM. This can change the augmentation random stream, so the run is not bit-identical to an
uninterrupted one. I disclose it rather than present it as a single clean run.

## Metrics

**13. Define precision, recall and mAP@0.5 as you computed them.**
Predictions are sorted by confidence and greedily matched one-to-one to the ground-truth box with the highest
IoU ≥ 0.5. Precision = TP / all predictions above threshold; recall = TP / all ground-truth boxes.
AP@0.5 is the COCO 101-point interpolated area under the precision-recall curve, using *all* predictions
(conf ≥ 0.001). It is threshold-independent.

**14. Your mAP@0.5 is 0.856 but Ultralytics says 0.834. Which is right?**
Both, under different definitions. Ultralytics integrates a trapezoid toward a (recall 1, precision 0)
sentinel (a perfect detector scores 0.995), and its validator batches images with rectangular padding.
Mine is COCO's 101-point rule, unit-tested (a perfect detector scores 1.0). I report mine and disclose the gap.

**15. Box-level vs image-level metrics: which matters to the customer?**
Image level: does a defective part get rejected, and does a good part pass? Box metrics measure localisation
quality, which matters for explaining a reject but not for the decision. At the recall-first threshold the
baseline has box precision 0.694 but rejected 29/29 defective and 0/40 good test parts.

**16. Why did the threshold change not change mAP?**
mAP integrates over all thresholds; it measures ranking. A threshold only picks an operating point on that
same precision-recall curve.

**17. What does "29/29, 0/40" actually prove?**
Not much in absolute terms. Exact 95% Clopper-Pearson intervals: sensitivity 88–100%, false-reject rate 0–8.8%. Also, the threshold (0.11) sits just above the highest
validation score on a good part (0.10). The margin is thin, and a new lighting condition could close it.

## Failure analysis

**18. Walk me through your failures.**
At the max-F1 threshold (0.73): 15 missed boxes, 0 false positives. 11 of the 15 had a correctly placed box
whose confidence was below threshold; 4 had no matching box at all, and all 4 were secondary mask fragments
(nested in or next to the main region). At image level, 6 defective parts would have passed, with top scores
of 0.33–0.70. On validation, good parts never scored above 0.10. So the dominant failure was operating-point
choice, not detection.

**19. Which defects are hardest?**
By box recall at the recall-first point: bent 9/13, scratch 10/11, color 8/9, flip 7/7. By size, small boxes
(<50 px side) are 1/4. Small and fragmented regions are the weakness, consistent with the fragment hypothesis.
Each of those counts is too small to rank defect types with confidence.

**19b. Your second experiment (merging fragments) didn't help. Why report it?**
Because it answers the question it was designed for. The four hard misses were fragments, so I merged
fragments within 20 px and retrained with everything else identical. Box precision rose (0.694→0.744), but the
model lost one colour defect completely (`color_004`, score 0.001), and decision-level performance did not
improve (28/29 vs 29/29). That is within noise on 29 images, so the conclusion is "no evidence of benefit",
not "it is worse". Hiding it would make E1 look like the only thing I tried.

## Deployment

**20. How did you verify the ONNX model is correct?**
Two levels on all 69 test images. Raw output tensors: max |Δ| 0.0033 px on box rows and 1e-5 on score rows.
Final detections through my framework-free NumPy pipeline vs Ultralytics: identical counts, max score
difference 1e-6. Plus a full re-evaluation of the ONNX model, which reproduces the PyTorch test metrics exactly.
My first parity check used a single 1e-3 tolerance across pixel and score units and "failed". I fixed the
unit error and disclose it.

**21. What happened with INT8?**
The first static INT8 model gave mAP 0. The output tensor concatenates 0–640 px box coordinates with 0–1
scores, so one shared 8-bit scale rounded every score to zero. Keeping the decode head in FP32 fixed it:
test mAP@0.5 0.851 (vs 0.856), 29/29 defective, 0/40 good, 3.1 MB vs 10.2 MB, ≈1.8× faster model time on
this CPU. INT8 gains depend on the CPU's integer instructions, so it must be re-measured on the target.

**22. What is your latency, exactly?**
On a Ryzen 5 5600H laptop, ONNX Runtime CPU EP, batch 1, 640×640 input: model ≈40.6 ms in an isolated run
and 58.6 ms median (45–69 ms range) in an interleaved run under background load. End-to-end including PNG
decode (≈10 ms) and letterbox (≈5 ms) is ≈56–78 ms. Model-only time is not inspection time.

**23. Why not batch to go faster?**
Batching raises throughput but not single-part latency. A part waiting for a batch to fill is *later*, not
earlier. For a per-part deadline, batch size is 1.

**24. Is this production-ready?**
No. One dataset, one camera setup, 29 defective test images, no real-line validation, no hardware-in-the-loop
test, and the controller adapter is simulated. It is a reproducible prototype with an honest error analysis.

**25. What would you monitor after deployment?**
The distribution of top scores on good parts (drift alarm before false rejects appear), reject and review
rates per shift, image statistics (brightness, sharpness), latency p95/p99, and periodic golden defective
samples to prove the system still catches known defects. Plus a labelled queue of every disputed decision.

**26. Camera or lighting changes: what breaks first?**
Probably the thin threshold margin (0.10 vs 0.11): a brightness or white-balance shift raises scores on good
parts, or lowers them on low-contrast defects such as colour spots and fine scratches. The synthetic
robustness test (if run; see report) only approximates this; real re-imaging is needed.

**27. Safety: what should happen if inference fails or times out?**
Fail safe: REJECT or REVIEW, never PASS. The decision policy is separate from the detector so this can be
enforced and tested independently.
