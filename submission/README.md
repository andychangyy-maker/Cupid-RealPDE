# Cupid: Track 2 training package

## Codabench
- Team: Cupid; team leader: Yaoyu Chang; members: Wenhao Liu and Hao Shi.
- Username: andychang.
- Submission id this package reproduces: 948801 (L33).
- Historical official final_score: 82.186978. This is the development submission's
  recorded score, not a measured official score for the freshly trained weights.

## How to reproduce
Build for linux/amd64 from this directory:

```sh
docker build --platform linux/amd64 -t realpde-track2-final .
```

Start with an empty writable output directory and the original public training
release. The default organizer command applies without an alternative entry point:

```sh
docker run --rm --network none --gpus '"device=0"' --shm-size 32g \
  -v /absolute/public-training-data:/data:ro \
  -v /absolute/empty-output:/output realpde-track2-final \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

`/data` contains `train_sim/` (100 public trajectories), `train_real/` (81 public
trajectories, with `7575_0.h5` excluded), and the official `mean_std_real.pt`.
Training writes all intermediate learned artifacts, logs and resource receipts to
`/output/training`, and the inference package to `/output/model`.
For evaluation, import `solution` in the same image and call
`solution.get_ttt_model('/output', 'cuda')`. Its `reset_ttt_state()` and
`ttt_step(input_norm, prev_target_norm=None)` follow the Track 2 Codabench contract.
The current target is never provided before its prediction.

No data download, pretrained teacher weights, old prediction cache or development
submission archive is required by `train()`. Training only mounts the public data
and empty output directories. A saved reference submission is used solely in our
separate paired validation, after training has ended.

## Hardware your submitted model was trained on
The historical development model was assembled from multiple component jobs.
The recovered P6645 point/UQ/tail records came from AutoDL 047; the September 27
machine inventory identifies that instance as one A800 PCIe 80 GB. Student,
calibration-fold and shape records came from D76, whose same-day inventory shows
RTX PRO 6000 Blackwell 96 GB GPUs. The D76 node had six concurrent GPUs; the logs
do not establish six-GPU distributed training of an individual component.
These are dated machine-inventory associations, not per-stage resource receipts.
The earlier A/B/C teacher checkpoints preserve recipes and progress logs but do
not identify their original training hardware.
The attached original logs retain those jobs and their recorded timings; they
are distinct from the new complete reproduction. Some original per-job GPU model,
CPU/RAM and peak-memory measurements were not preserved. We do not substitute
later reproduction measurements for missing historical measurements.

From-scratch training on October 8-9, 2026:
- GPU: one NVIDIA A800-SXM4-80GB; driver 580.65.06.
- Allocated CPU and memory: 10 CPU cores and 100 GiB RAM.
- Training time: 14,060.03 seconds inside the container (3 h 54 min 20 s); 14,124 seconds from Job creation to completion.
- Peak PyTorch allocated/reserved GPU memory: 23,974,356,992 / 25,352,470,528 bytes (per-stage allocator peaks, not whole-device NVML peaks).
- Complete from-scratch stages: 31/31; no training retry.
- Runtime: the supplied OCI image under Kubernetes/containerd, with a separate
  `unshare --user --net` network namespace for the training and validation process.
  IPv4, IPv6 and DNS isolation, read-only public data and initially empty output
  were checked. This was not a literal Docker-engine GPU invocation. The image
  also passed a separate local CPU test under Docker `--network none`.

The organizer's default is one A800, 16 CPU cores and 128 GB RAM. We request no
additional resources. The isolated reproduction used the smaller allocation above;
its measured results and limitations are reported separately from official scores.

## Environment
- Base: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`, pinned in the Dockerfile to
  `sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726`.
- Added packages: numpy 1.26.4, h5py 3.11.0, scipy 1.14.1, einops 0.8.0,
  ninja 1.11.1.4. Their bundled wheels are installed with `--no-index`.
- CUDA toolkit version: 12.4; PyTorch: 2.4.1 from the pinned base image.
- TinyCC and required headers are bundled for offline runtime compilation.
  Their source archive and build record are included.
- Thread environment: OMP, MKL and OpenBLAS each set to 4; CUBLAS workspace
  configuration is `:4096:8`.

## Data, weights and code
- Only released public Sim and clean81 Real training data are used.
- Three Sim teachers, student pretraining, five calibration fits, the full Real
  fit, shape models and UQ are rebuilt in the fixed 31-stage pipeline.
- Official baseline checkpoints: none. `assets/mean_std_real.pt` is the public
  normalization artifact, not a learned initialization.
- Historical teacher/student/calibration/shape/UQ weights are not packaged.
- Augmentation: noisy history in student pretraining; physical vertical reflection
  with probability 0.25 in the Real stages; the fixed time-jitter and sampling
  rules in the native UQ recipe. METHOD.md and the included recipes specify them.
- `THIRD_PARTY_NOTICES.md` lists bundled code, versions and licenses. License
  texts are included in `third_party_licenses/` and the corresponding vendor tree.

## Randomness
- Sim teachers A/B/C: seeds 0/0/1; student pretraining: 1; Real point fits: 4.
- Native UQ: 18; shape initialization: 18618; shape batch generator: 18619.
- Deployed UQ: 20260923; reflection distillation: 20260924.
- Inference feature projection: 20260925.
- Training uses fixed update endpoints and declared EMA/blend rules. It does not
  choose a new endpoint using private evaluation results. Fold point models use
  5,334 updates; the final point model uses 6,645. See METHOD.md for the full order.
- A controlled repeated-seed variance study has not been performed.

## Track 2 only
- LLM gateway: not used.
- The neural networks remain frozen during evaluation. Causal statistics and a
  regularized regional linear readout adapt using already revealed targets.

## Contact
Yaoyu Chang, andychangyy@gmail.com, team Cupid. Contact for Decision Phase replies
within 48 hours.

## Notes
The five-angle calibration is stage-only out-of-fold calibration, not a full
pipeline holdout. Earlier Sim/student stages use shared public Real prefixes and
other fold fits overlap. The final Real fit includes public diagnostic tails.
Consequently the complete public replay is a reproduction diagnostic and is not
an estimate of the hidden development or private final-test score.

Separate attachments retain the original component training logs and the new
complete training/validation receipts. 

The October 9 paired validation completed on the same A800 and unchanged
GPU-tested inference code, using real Torch compilation and CUDA graphs.
Final API, ordered finite bounds, zero pressure, per-trajectory reset and all six
compilation/graph checks passed. Both models consumed identical input and target
hashes, the same clean81 manifest and the same official scoring implementation.

| Public replay view | New quality mean | Sealed 948801 quality mean | Difference |
|---|---:|---:|---:|
| All 3,341 steps | 83.26290092 | 83.27194460 | -0.00904368 |
| 716 tail windows | 83.26968931 | 83.27747344 | -0.00778414 |
| 648 early windows | 83.15250990 | 83.15962521 | -0.00711531 |

The quality mean is the mean of four quality metrics and excludes the official
Time score. The tail SPS difference was -0.00447253 points. This is public
training-overlap evidence, not a score on the hidden development or final set.
Our internal rejection limits were quality-mean difference >= -0.25 and SPS
difference >= -0.5; these are not tolerances supplied or approved by the organizers.

Mean synchronized ttt_step time was 2.516556 ms for the new model and 2.504131 ms
for the sealed baseline (+0.4962%). Across 3,341 steps, those calls took 8.407815
and 8.366303 seconds, respectively. This excludes model loading/compilation, input
normalization, scoring and output copies; it is not the official Time measurement.

The first post-training validation stopped before scoring because a historical
input reader had been copied to a shallower directory. The one validation retry
loaded the byte-identical reader from its original packaged directory and passed.
No training, model, scoring, data or quality-threshold change was made for this fix.
The successful validation took 905.43 seconds including API and both replays.

This release updates only README.md and METHOD.md relative to the GPU-tested
candidate. All other 1,833 package files are byte-identical. A separate local
Docker CPU check covers the documentation-updated image; it was not retrained on
GPU after this documentation-only rebuild.

