# Written answers

## A. "30 defective images, 3,000 good images — how would you build a reliable system?"

Thirty defect images cannot support a supervised detector whose miss rate is known with any confidence:
split 20/10, a single extra miss on the 10 held-out defects moves recall by 10 points, and rare defect
modes may not appear at all. In this project, 45 training defects across four types produced a
detector that *ranked* parts well, but whose operating threshold, chosen by the textbook max-F1 rule,
let 6 of 29 defective test parts pass; and with only 29 defective test images, even a perfect 29/29
only bounds true sensitivity above 88% (exact 95% interval). So I would not make a supervised detector the
primary safety mechanism.

**Proposed system.** (1) *Primary gate: normal-only anomaly detection.* Fit a PatchCore-style memory bank
of pretrained CNN patch features on ~2,500 of the good images (keep ~500 good images untouched for
calibration and testing). It needs no defect labels and responds to defect types nobody has seen yet. (Measured here: a detector
retrained without the `scratch` type rejected only 3/23 scratched nuts, most scoring ≤ 0.02; REPORT §13a.)
(2) *Secondary: defect localisation/classification.* Fine-tune a small pretrained detector (transfer
learning from COCO/ImageNet) on the 30 defects plus good images, used to explain *where* and *what*,
not as the only reject trigger. (3) *Human review band.* Scores between a "certain pass" and
"certain reject" threshold go to an operator; their decisions become new labels (active learning:
prioritise the most uncertain and the most anomalous-but-passed parts for labelling).

**Augmentation, carefully.** Photometric jitter matching measured lighting variation, small
rotations/shifts consistent with the fixture. Check that no transformation turns a good part into
something that looks like a defect class: for example, a mirror image of a chiral part such as this
metal nut is geometrically similar to the "flipped" defect. (In this project the default mirror
augmentation did *not* measurably hurt the flip class, 7/7 detected, but that should be checked per
product, not assumed.) Synthetic defects (cut-paste, CutPaste/DRAEM-style) can help train the localiser, but
results must never be validated on synthetic defects — only on real ones.

**Threshold calibration and validation.** Set the anomaly threshold on held-out good images for a
target false-reject rate (e.g. ≤1%), then report recall on *all* 10 held-out real defects with a
binomial confidence interval and state plainly how weak that evidence is. Before go-live, run a
shadow-mode factory trial: the system scores every part while existing inspection continues, and every
disagreement is reviewed. Collect a deliberately seeded set of known-defective parts to measure the
miss rate on the line. **Monitoring:** daily score distributions on good parts (drift alarm),
reject/review rates, periodic re-insertion of golden defective samples, and a labelled-failure queue
feeding retraining under version control.

## B. "Scored well in testing; two weeks after go-live it starts missing defects. What do you check, in order?"

I check cheapest-and-most-likely first, and **I do not retrain until the imaging and software path is
proven unchanged**, because retraining on a broken pipeline hides the real fault.

1. **Verify the misses.** Pull the stored images, scores and decisions for the escaped parts. Were they
   imaged at all (trigger timing)? Is the defect visible in the stored image? Was the score just below
   threshold or nowhere near? This separates *imaging* problems from *model* problems.
2. **Camera alignment and focus.** Compare today's frames against go-live reference frames (part
   position, scale, sharpness via Laplacian variance). Bumped mounts and drifted focus are common.
3. **Lens and window cleanliness.** Oil mist and dust reduce contrast — exactly what hides scratches.
4. **Illumination and reflections.** Ageing or failed LEDs, a changed ambient light source, a new
   reflective surface near the station. Compare image brightness/contrast histograms with reference.
5. **Exposure, gain and white balance.** Auto-exposure or auto-white-balance left on, or settings reset
   after a power cycle. Colour defects are especially sensitive to white balance.
6. **Motion blur and conveyor speed.** A line-speed increase with the same exposure time blurs fine defects.
7. **Preprocessing and software configuration.** Colour order (BGR/RGB), resize/letterbox, crop/ROI,
   image format/compression changes, a changed camera driver.
8. **Model and threshold versions.** Confirm the deployed model hash and threshold match the validated
   release; check for an accidental rollback or a "temporary" threshold change.
9. **Hardware runtime behaviour.** Thermal throttling, dropped or skipped frames when latency exceeds
   the cycle time, timeouts silently defaulting to PASS (that would be a design bug: fail-safe should
   be REJECT/REVIEW).
10. **New defect types.** Are the escapes a defect mode absent from training (e.g. a new tool wear pattern)?
11. **Manufacturing-process drift.** New material batch, coating, supplier or surface finish shifting the
    appearance of *good* parts and therefore the score distribution.
12. **Collect and label the failures**, plus recent good parts, under version control.
13. **Evaluate corrective actions** (fix imaging, recalibrate threshold, retrain/fine-tune) on a frozen
    held-out set *that includes the new failures*, never on the data used to tune the fix.
14. **Rollback and monitoring.** Keep the previous model one command away; add drift alarms on input
    statistics and score distributions, and periodic golden-sample checks so this is detected by the
    system, not by the customer.

## C. "The controller can spend at most 10 ms per image. Your model takes 40 ms. What are your options?"

40 ms does not meet a 10 ms budget, and averages don't satisfy a hard budget — the p95/p99 must fit,
with margin. Options, in the order I would try them:

1. **Profile end-to-end first.** Separate image transfer, decode, resize/letterbox, model, NMS and
   result I/O. In this project the YOLO11n model happens to take ≈40–59 ms on a Ryzen 5 5600H laptop
   CPU, and PNG decoding (≈10 ms) plus letterboxing (≈5 ms) add another ≈15 ms. So even a model that
   ran in 0 ms would miss a 10 ms budget unless image acquisition hands over raw, already-sized pixels.
   The 10 ms must be defined (model only? camera-to-decision?) before optimising.
2. **Runtime tuning (no accuracy cost).** ONNX Runtime full graph optimisation, pin intra-op threads to
   the physical core count, disable spinning contention with other control tasks, keep a warm session,
   avoid per-call allocations. Usually tens of percent, not 4×.
3. **Reduce input resolution / crop to a region of interest.** Latency scales roughly with pixel count:
   640→320 is ~4× fewer pixels. With a fixtured part, cropping a tight ROI around the nut keeps defect
   pixel density while shrinking the input. Must be re-validated: small scratches suffer first.
4. **Smaller or simpler model.** Fewer channels/layers, a detector head with fewer scales, or
   distillation from the larger model into a smaller student to recover some accuracy.
5. **Quantisation (INT8).** Static quantisation with calibration images can give 1.5–3× on CPUs with
   VNNI/AMX/dot-product instructions, and little or nothing on CPUs without them. Measure on the target
   CPU; check accuracy on the held-out defects, not just mAP.
6. **Hardware acceleration.** An NPU/GPU/VPU on or next to the controller, or a dedicated vision PC that
   returns only the decision to the controller. Often the most reliable route to a 4× gap.
7. **Change the system architecture.** Pipeline capture and inference so throughput meets the line rate,
   if (and only if) the 10 ms is a *throughput* budget. If it is a per-image *latency* deadline, batching
   does not help. Ask which it is.
8. **Re-validate the trade-off.** Every speed-up changes missed-defect and false-reject rates. Present
   the customer with a Pareto table (latency p95 vs recall on real defects vs false-reject rate) measured
   on the actual target hardware, and let the missed-defect cost drive the choice.

**Measured example from this project** (same checkpoint, local CPU, interleaved benchmark, threshold
re-chosen on validation for each variant; test = 29 defective / 40 good parts):

| variant | model ms (median) | end-to-end ms | defective flagged | good rejected |
|---|---|---|---|---|
| FP32 640×640 | 58.6 (isolated run: 40.6) | 77.6 | 29/29 | 0/40 |
| INT8 640×640 | 33.3 | 51.2 | 29/29 | 0/40 |
| FP32 480×480 | 25.2 | 40.1 | 28/29 | 1/40 |
| FP32 320×320 | 10.7 | 24.0 | 28/29 | 1/40 |

Only 320×320 approaches 10 ms for the *model*, at a measurable cost (one more escaped defect, one false
reject on a small test set), and no variant meets 10 ms *end-to-end* on this CPU. My recommendation
would therefore be INT8 + a tight ROI crop + raw-frame handover, re-measured on the customer's actual
controller, and, if that still misses, a hardware accelerator, rather than trading away defect recall.
