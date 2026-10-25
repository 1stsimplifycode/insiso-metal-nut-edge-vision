# AI-assistance disclosure

InSiSo's brief permits AI assistance. This is an accurate account of how it was used.

**Tool:** Claude Code (Anthropic, model Claude Opus 5.5), run as an agent in a terminal on my laptop, with
permission to run Python/PowerShell and read and write files in this project folder.

| Area | What the AI did | What I did |
|---|---|---|
| Code generation | Wrote essentially all source code in `src/`, `scripts/`, `tests/` and `tools/`, and the Colab notebook generator. | Set the requirements (the assignment brief plus my own detailed execution brief), reviewed the code, and approved the execution environment. |
| Experiment design | Proposed the split protocol, the one-box-per-component baseline, the recall-first threshold rule (E1) and the merged-fragment annotation experiment (E2), each derived from observed failures. | Specified the required protocol constraints (leakage-free split, val-only selection, one factor per experiment, no fabricated results) and the priorities. |
| Execution | Downloaded data, trained, evaluated, exported and benchmarked on my laptop (GTX 1650 for training, Ryzen 5 5600H CPU for benchmarks). Every reported number comes from these runs. | Kept the machine available (closed applications when memory ran out) and decided how to continue after the training run was stopped. |
| Debugging | Found and fixed its own mistakes during the run, each recorded in the report: an AP implementation that did not match the COCO definition (caught by unit tests), a unit-mixed ONNX parity tolerance, a fully-quantised INT8 model that output zero scores, and benchmark contamination from background load. | — |
| Documentation | Drafted the README, the technical report, the written answers A–C, the interview Q&A and this disclosure. | To do before submission: read every document and rewrite in my own words where needed, so that I can defend each claim personally. |

**What was *not* done by AI or anyone:** no result in this repository was hand-entered or estimated. Tables
are copied from JSON artifacts produced by the scripts. Anything that was not executed is marked as such.

**Not verified:** byte-identity of the dataset with MVTec's official archive (official link was down),
any physical camera, PLC or Bosch Rexroth ctrlX integration, and any deployment CPU other than my laptop.
