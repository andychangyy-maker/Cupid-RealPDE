# Results and evidence

[Home](../README.md)

## Official final result

The [organizers' final results](https://realpdecompetition.github.io/results.html),
published October 17, updated November 4, and checked November 5, 2026, list **Cupid (andychang)** at
**joint 9th place** in Track 2. The total score is **82.6104**, with standard
deviation **0.0653**.

Organizers trained each team's package from scratch three times on A800 GPUs,
then evaluated each resulting model five times. The table reports the mean and
standard deviation across those 15 evaluations. Shared places follow the
organizers' stated rule; we retain their `9=` designation.

The [official NeurIPS poster entry](https://nips.cc/virtual/2026/loc/sydney/181002)
lists the solution title and all three authors. This is a competition-session
poster listing, not a claim of an accepted NeurIPS main-track paper.

This is an official private-test result. The earlier development submission
948801 scored 82.186978; these are different evaluation sets and should not be
treated as a before/after method improvement.

## Local Docker Engine reproduction

The submitted source and frozen image were used for the October 10 run. This
local execution evidence is separate from the organizer-run final evaluation.

| Measurement | Recorded value |
|---|---|
| GPU | One NVIDIA A800-SXM4-80GB |
| CPU / RAM allocation | 16 CPUs / 128,000,000,000 bytes |
| Training isolation | Docker Engine, `--network none`, public input read-only |
| Initial state | Empty output; no external model checkpoint mounted |
| Completed stages | 31 / 31 |
| Training time | 14,843.55 s (4 h 07 min 24 s) |
| PyTorch allocated / reserved GPU peaks | 23,974,356,992 / 25,352,470,528 bytes |
| API and paired replay time | 976.61 s |
| API check | Shape, finite ordered bounds, zero Real pressure, and reset checked |
| Compilation / graph checks | All six recorded checks passed for both models |
| Complete replay | 81 Real trajectories; 3,341 causal steps per model |

The GPU peaks are maxima of the per-stage PyTorch allocator records, not
whole-device NVML peak measurements.

### Paired quality comparison

The rows below come directly from the two preserved JSON replay receipts.
The tail table uses the 716-window diagnostic view. Higher scores are better.

| Tail-window metric | Fresh Docker-trained model | Sealed 948801 | Fresh − sealed |
|---|---:|---:|---:|
| Relative-L2 score | 96.65649782 | 96.65439235 | +0.00210547 |
| TKE score | 82.95642799 | 82.96895923 | −0.01253124 |
| MVPE score | 97.04538779 | 97.04640600 | −0.00101821 |
| SPS score | 56.42880746 | 56.44005763 | −0.01125017 |
| Four-metric quality mean | 83.27178026 | 83.27745380 | −0.00567354 |

Across all 3,341 steps, the quality means were 83.26091942 and 83.27197313
respectively. This mean excludes the official Time score. The data overlaps
training; these are reproduction diagnostics, not hidden-set performance.

### Timing scope

Synchronized `ttt_step` mean times were **2.656574 ms** (fresh) and
**2.661548 ms** (sealed). This is one paired timing observation, not a stable
speedup claim. Loading/compilation, data normalization, scoring, and result copies
are excluded. Do not compare these numbers to full evaluator wall time or the
official efficiency score.

## Inspectable evidence

The [`evidence/`](../evidence/) directory contains the original training report,
Final API report, both replay reports, and the validation supervisor report,
alongside a derived execution summary and SHA256 provenance. The latter identifies
which fields were selected from the original training supervisor record.

Some October 10 receipts say `release_ready=false` or describe official acceptance
as still pending. Those fields describe the historical local check and are
preserved unchanged. The later official final-results page is separate evidence;
we do not rewrite old receipts to make them claim knowledge they did not have.

## Limits

- This release does not establish an independent benefit for every component; no
  new ablation is inferred from the competition score.
- Five-angle calibration is stage-only OOF, not full-pipeline leave-one-angle-out.
- Our local rejection thresholds were internal checks, not organizer-approved
  reproducibility tolerances.
- The local single-run records do not constitute a repeated-seed variance study.
- This is a competition solution release, not an accepted-method paper or a
  claim of general-purpose state-of-the-art performance.
