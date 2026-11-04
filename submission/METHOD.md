# Cupid: Track 2 method

## 1. Summary
A frozen multi-member U-Net predicts point estimates and uncertainty bounds. An
independent Real-data model supplies normalized fluctuation shape; input-conditioned
mean calibration and causal online statistics adjust the deployed forecast.
All learned offline components, including three Sim teachers, are rebuilt by train().

## 2. Model
The deployed student uses base width 128, depth 3, four members and a UQ hidden
width of 160: 18,778,953 parameters before storage compression, including 2,814,022
UQ parameters. The independent shape model has 4,403,660 parameters. The model
consumes 20 frames at 32 by 64 with u, v and a zero Real-pressure channel, and
predicts the next 20 frames. Public official normalization and the model's own
training normalization are explicitly converted by the runtime. Teachers, student
pretraining and the shape model begin from random initialization; later stages
initialize only from earlier outputs of the same train() call.

## 3. Data
100 released Sim trajectories and 81 released Real trajectories are used.
7575_0.h5 is excluded, matching the Final kit. The fixed clean81 manifest contains
2,544 prefix and 716 tail windows. Five angle-specific point/shape fits exclude
their own calibration angle. This is stage-only OOF, not full-pipeline holdout:
Sim pretraining has already used shared Real prefixes, other-fold fits overlap,
and the final Real fit uses full public trajectories. No private targets are read.
Physical vertical reflection has probability 0.25 in the Real stages; native UQ
also uses the recovered time-jitter rule. Sim teacher strides are 20, 20 and 16.

## 4. Training
The table lists every stage in execution order. Loss definitions and sampling
constants are included in the fixed source recipes; stages.py adapts file paths
and replaces historical artifact hashes with hashes generated during this run.

| Stage | Data | Loss | Optimizer and learning rate schedule | Steps or epochs | Batch size |
|---|---|---|---|---|---|
| 1. teacher / A | Public Sim | Point/member accuracy and quantile pinball losses | AdamW 3e-4, cosine, wd 1e-5 | 10,000 | 8 |
| 2. teacher / B | Public Sim | Point/member accuracy, quantile pinball and differentiable SPS losses | AdamW 3e-4, cosine, wd 1e-5 | 10,000 | 8 |
| 3. teacher / C | Public Sim | Point/member accuracy and quantile pinball losses | AdamW 3e-4, cosine, wd 1e-5 | 10,000 | 8 |
| 4. cache / full | Sim teacher predictions | No optimization; A/B point blend plus C temporal-mean correction | None | 4,800 windows | 16 |
| 5. closure / full | Real prefixes | Public-data closure statistics | None | Fixed manifest | 64 |
| 6. sim / pretext | Sim + Real prefixes | Noisy-history relative-L2 future prediction | AdamW 3e-4, cosine, wd 1e-5 | 4,000 | 4 + 4 domains |
| 7. sim / standard1 | Sim + regenerated teacher/closure caches | Recovered standard Sim point, UQ and distillation objectives | AdamW 3e-4, cosine, wd 1e-5 | 10,000 | 8 |
| 8. sim / standard2 | Sim + regenerated teacher/closure caches | Recovered standard Sim point, UQ and distillation objectives | AdamW 3e-4, cosine, wd 1e-5 | 10,000 | 8 |
| 9. point / 0 | Four Real angles, excluding 0 | Point/member accuracy + differentiable SPS weight 0.80 + final-score term | AdamW 2e-4, cosine T_max=12,000, wd 1e-5; EMA 0.999 | 5,334 | 8 |
| 10. shape / 0 | Four Real angles, excluding 0 | 1k normalized MSE warmup, then accuracy + 0.05 MSE | AdamW 3e-4 to 3e-5 cosine; EMA 0.999 | 20,000 | 32 |
| 11. mean_cache / 0 | Held-angle 0 public Real tails | No optimization; deployed point + same-fold shape predictions | None | Fixed tail windows | Original evaluation batches |
| 12. point / 5 | Four Real angles, excluding 5 | Point/member accuracy + differentiable SPS weight 0.80 + final-score term | AdamW 2e-4, cosine T_max=12,000, wd 1e-5; EMA 0.999 | 5,334 | 8 |
| 13. shape / 5 | Four Real angles, excluding 5 | 1k normalized MSE warmup, then accuracy + 0.05 MSE | AdamW 3e-4 to 3e-5 cosine; EMA 0.999 | 20,000 | 32 |
| 14. mean_cache / 5 | Held-angle 5 public Real tails | No optimization; deployed point + same-fold shape predictions | None | Fixed tail windows | Original evaluation batches |
| 15. point / 10 | Four Real angles, excluding 10 | Point/member accuracy + differentiable SPS weight 0.80 + final-score term | AdamW 2e-4, cosine T_max=12,000, wd 1e-5; EMA 0.999 | 5,334 | 8 |
| 16. shape / 10 | Four Real angles, excluding 10 | 1k normalized MSE warmup, then accuracy + 0.05 MSE | AdamW 3e-4 to 3e-5 cosine; EMA 0.999 | 20,000 | 32 |
| 17. mean_cache / 10 | Held-angle 10 public Real tails | No optimization; deployed point + same-fold shape predictions | None | Fixed tail windows | Original evaluation batches |
| 18. point / 15 | Four Real angles, excluding 15 | Point/member accuracy + differentiable SPS weight 0.80 + final-score term | AdamW 2e-4, cosine T_max=12,000, wd 1e-5; EMA 0.999 | 5,334 | 8 |
| 19. shape / 15 | Four Real angles, excluding 15 | 1k normalized MSE warmup, then accuracy + 0.05 MSE | AdamW 3e-4 to 3e-5 cosine; EMA 0.999 | 20,000 | 32 |
| 20. mean_cache / 15 | Held-angle 15 public Real tails | No optimization; deployed point + same-fold shape predictions | None | Fixed tail windows | Original evaluation batches |
| 21. point / 20 | Four Real angles, excluding 20 | Point/member accuracy + differentiable SPS weight 0.80 + final-score term | AdamW 2e-4, cosine T_max=12,000, wd 1e-5; EMA 0.999 | 5,334 | 8 |
| 22. shape / 20 | Four Real angles, excluding 20 | 1k normalized MSE warmup, then accuracy + 0.05 MSE | AdamW 3e-4 to 3e-5 cosine; EMA 0.999 | 20,000 | 32 |
| 23. mean_cache / 20 | Held-angle 20 public Real tails | No optimization; deployed point + same-fold shape predictions | None | Fixed tail windows | Original evaluation batches |
| 24. point / full | All clean81 Real | Point/member accuracy + differentiable SPS weight 0.80 + final-score term | AdamW 2e-4, cosine T_max=12,000, wd 1e-5; EMA 0.999 | 6,645 | 8 |
| 25. native_uq / full | All clean81 Real | Fixed sharp2 pinball/SPS/CVaR interval objective; point frozen | AdamW 1e-4, cosine, wd 1e-5 | 3,200 | 8 |
| 26. shape / full | All clean81 Real | 1k normalized MSE warmup, then accuracy + 0.05 MSE | AdamW 3e-4 to 3e-5 cosine; EMA 0.999 | 20,000 | 32 |
| 27. mean / full | 716 tail windows from five stage-OOF caches | Regional temporal-mean least squares, conditioned on input log-RMS | Closed-form 16 two-coefficient systems | One fit | All cache rows |
| 28. tail / uq8000 | All clean81 Real | Deployed-output interval objective with adaptive temperature | AdamW 1e-5 to 2e-6 cosine; EMA 0.999 | 8,000 | 8 |
| 29. tail / reflection4000 | All clean81 Real | Physical-reflection ensemble distillation plus deployed-output objective | AdamW point 3e-6 to 3e-7, UQ 1e-5 to 2e-6; cosine | 4,000 | 8 |
| 30. tail / blend | New UQ endpoints | No optimization; 0.5/0.5 UQ-parameter blend | None | One blend | N/A |
| 31. export / full | All newly trained components | No optimization; fixed per-tensor INT8/FP16 export | None | One export | N/A |


The teacher accuracy objectives use relative-L2, TKE and MVPE weights (1,4,1)
for A/B and (3,2,5) for C, with normalized MSE weight 0.25, temporal-mean
loss weight 0.25, member weight 1, member bootstrap probability 0.75, and
quantile pinball weight 0.05. B additionally uses soft-SPS weight 0.05 with
1,500-step warmup and temperature fraction 0.05. Exact configurations are in
`assets/teacher_recipes.json`. The cache takes the midpoint of A and B, then
adds half the difference between C's temporal mean and that midpoint's mean.

Real point/member accuracy weights are (5,3,2). An additional accuracy term
uses (5,1,1) with coefficient 1, differentiating the batch-mean errors. Native
soft-SPS is coupled to the point center and factor, with weight 0.80,
1,500-step warmup and temperature fraction 0.05. The implementation and
per-stage receipts are in `recipes/point/train.py` and `point_endpoint.json`.

Fixed update endpoints are used. The final point model uses 6,645 updates while
calibration-fold point models retain 5,334 updates. The final blend takes only
its declared UQ parameter groups; the exported point backbone remains the P6645
endpoint. Seeds are teachers 0/0/1, student pretraining 1, point 4, native UQ 18,
shape 18618 with generator 18619, deployed UQ 20260923, and reflection 20260924.
Run-to-run spread has not yet been measured.

## 5. Uncertainty bounds
Native quantile and refinement heads predict asymmetric bounds. Offline UQ
training sees the complete deployed transformation. During inference, causal
regional and pixel residual CDF statistics use only already revealed targets,
with regional empirical shrinkage and sparse refreshes. Lower and upper bounds
are returned in the official normalized coordinate system.

## 6. Inference
The backbone stays frozen. Per-trajectory statistics update regional means,
fluctuation amplitudes, empirical bounds and a 32-feature regional linear readout.
The readout uses a seeded orthogonal feature projection, accumulated normal
equations and ridge regularization, rather than gradient steps (zero neural-network gradient updates per window;
no inference learning rate). Current targets
are not visible until after their predictions. reset_ttt_state clears every
trajectory-specific accumulator and cached observation. The method does not call
an LLM gateway. Exact L33 runtime source is preserved under runtime/track2.
The readout projection uses seed 20260925, 32 projected features plus a bias,
and 8 spatial regions. With n revealed targets, ridge strength is
`5/max(n,1)` times the mean diagonal of the region's accumulated feature
normal matrix, lower-bounded by 1e-8. Corrections begin at n=3 with weight
`n/(n+5)`; residual-CDF refresh occurs at n=3 and powers of two. The feature
readout preserves the fluctuation energy while adjusting its temporal shape.

## 7. Speed
Weights use the frozen per-name INT8/FP16 storage policy and FP16 inference.
The runtime uses Torch compilation and CUDA graphs, reusable buffers, sparse
CDF solves, and conv_1x1_as_mm=True. The October 9 A800/CUDA 12.4 paired replay verified all six compilation/graph
checks. Mean synchronized ttt_step time was 2.516556 ms for the fresh model
and 2.504131 ms for sealed 948801. These calls exclude loading, preprocessing
and scoring and are not the official Time score.

## 8. What mattered
The development submission 948801 scored 82.18698. That number belongs to the
historical submission, not this fresh run. The complete 3,341-step public paired replay yielded quality means
83.26290092 (new) and 83.27194460 (sealed). On 716 tail windows the difference
was -0.00778414; tail SPS differed by -0.00447253. This comparison checks
reproduction, not the causal benefit of any one component. Controlled
component-isolated effect sizes are not available; no unrun ablation is claimed.

## 9. Compute
The historical submission was assembled from component jobs whose original
logs accompany this package. A single historical end-to-end time and complete
per-job hardware/peak-memory record were not preserved. The October 8
reproduction runs on one A800-SXM4-80GB, 10 CPU cores and 100 GiB RAM in the
same OCI image under Kubernetes/containerd with process network isolation.
Training completed 31/31 stages without retry in 14,060.03 container seconds. Peak PyTorch allocated/reserved memory was 23,974,356,992 / 25,352,470,528 bytes. Final API and both full compiled replays subsequently passed in 905.43 seconds;
see README.md and the paired validation receipts for metrics and timing scope. This is distinct from
literal Docker-engine GPU execution and from the official hidden evaluation.

## 10. References
The public RealPDE Starting Kits provide the data contract and scoring functions.
Implementation dependencies are PyTorch (BSD-3-Clause), NumPy/SciPy (BSD), h5py
(BSD), and einops (MIT). Recovered TinyCC files include LGPL-2.1 COPYING. The release documentation uses measured October 8-9 training and validation receipts.
Contact: Yaoyu Chang, andychangyy@gmail.com, team Cupid.
