# Reproduction guide

[Home](../README.md) · [Data preparation](data.md) · [Recorded results](results.md)

## Tested environment

The October 10, 2026 local reproduction used Docker Engine, a single
NVIDIA A800-SXM4-80GB, 16 CPUs, a 128,000,000,000-byte RAM limit, and
`--network none`. Training completed all 31 stages in 14,843.55 seconds.
The subsequent API and full paired replay took 976.61 seconds.

The Dockerfile pins PyTorch 2.4.1 / CUDA 12.4 / cuDNN 9 and the base image digest.
NumPy 1.26.4, h5py 3.11.0, SciPy 1.14.1, einops 0.8.0, and Ninja 1.11.1.4
are installed from bundled wheels without a package-index connection. TinyCC
and its sources are included for runtime compilation.

Host requirements: Linux/amd64, a compatible NVIDIA driver, Docker Engine, and
GPU container support. The driver and container runtime are host prerequisites,
not installed by this repository. Other GPUs may work but the quoted time and
memory measurements are from the A800 run.

## 1. Verify and build

From the repository root:

```bash
python scripts/verify_source.py
docker build --platform linux/amd64 -t cupid-realpde:948801 ./submission
```

The integrity check compares all 1,835 original package files with SHA256 values
computed from the submitted source ZIP. The source ZIP's SHA256 is:

```text
57d75bbb16c1ef7195b3f8be30dc08e724dcc51d296c30b857b2ad0326fe308b
```

Building can require downloading the pinned base image. **Network-isolated
training starts after the image has been built.** A fresh Docker build can have
different image metadata/digests; the historical submitted registry digest is
an identity record, not a guarantee of bit-identical rebuild metadata:

```text
andychang123/realpde-track2:20261009-release1
sha256:9457e8f5594b459d69e3cbf8510f8a6ab1b2acf4b03961bbe742afbdf3669048
```

The historical registry repository is private following organizer acquisition.
Use the source-build path above for public reproduction.

## 2. Prepare and check the public data

Follow [data.md](data.md). Set `DATA` to the absolute path of the prepared
directory. The check below reads file counts, normalization hash, and Real
trajectory metadata, without training or requiring a GPU:

```bash
DATA=/absolute/path/to/public-training-data
docker run --rm --network none \
  --mount type=bind,src="$DATA",dst=/data,readonly \
  cupid-realpde:948801 \
  python -c "import json, pipeline; print(json.dumps(pipeline.inspect_data('/data'), indent=2))"
```

Expected values: 100 Sim files, 81 Real files, 2,544 prefix windows, 716 tail
windows, and normalization SHA256
`bae5235f73431ac7d76e6bdcea480b8ca9fda0ac01e57ab26af66bad20dc3b40`.

This check is a data-contract preflight, not a claim to validate every array
value or the entire training run.

## 3. Train from scratch

`OUTPUT` must be a new, empty directory. The training API rejects nonempty
output to prevent accidental reuse of an earlier run. The command consumes GPU
compute; allow approximately the measured four hours plus environment-dependent
headroom. The pipeline enforces its own 24-hour training limit.

```bash
OUTPUT=/absolute/path/to/new-empty-output
mkdir -p "$OUTPUT"
docker run --rm --network none --gpus '"device=0"' \
  --cpus 16 --memory 128000000000 --shm-size 32g \
  --mount type=bind,src="$DATA",dst=/data,readonly \
  --mount type=bind,src="$OUTPUT",dst=/output \
  cupid-realpde:948801 \
  python -c "import solution; solution.train('/data', '/output', 'cuda')"
```

Expose **exactly one GPU**. The training API accepts `cuda` or `cuda:0` and checks
the pinned PyTorch/CUDA versions. Data is read-only, and the training container
does not mount a reference model or previous submission.

Successful output contains:

```text
output/
├── training_report.json       # complete=true; 31 completed stages
├── training/
│   ├── logs/                  # per-stage logs
│   └── ...                    # newly trained components and generated caches
└── model/                     # exported runtime and newly trained model files
```

Keep the report and stage logs. A process exit alone is insufficient: check
`complete`, stage count, and the exported model. If a run fails, inspect its
recorded error; do not relabel a partial output as a successful reproduction.

## 4. Use the Final inference API

Inside the same image, mount completed output read-only at `/output` and call:

```python
import solution

model = solution.get_ttt_model('/output', 'cuda')
model.reset_ttt_state()

# input_norm: official-normalized float32 [1, 20, 32, 64, 3].
# No feedback is supplied for the first forecast of a trajectory.
prediction, info = model.ttt_step(input_norm, prev_target_norm=None)
lower, upper = info['lower'], info['upper']

# Later, after that forecast's target has actually been revealed:
prediction, info = model.ttt_step(next_input_norm, prev_target_norm=revealed_target_norm)
```

This snippet illustrates the contract; the named input tensors come from the
official evaluation stream. The current target must never be passed as feedback
for its own prediction. Do not carry state across trajectories. Bounds and
prediction use official normalized coordinates; Real pressure is zero.

Initial loading includes compilation and CUDA-graph warmup and can take minutes.
The approximately 2.66 ms step timing in our records **excludes** that startup,
normalization, scoring, and output-copy work. It is not an end-to-end latency or
the competition's official Time score.

Use the Track 2 Final starting kit from
[Codabench](https://www.codabench.org/competitions/17385/) for the official
evaluation protocol and scorer. This repository's public receipts can be
inspected without access to private evaluation data.

## Reproduction scope

The original `submission/README.md` records the earlier containerd validation.
Its text is preserved for source identity; the later Docker Engine receipts in
[`evidence/`](../evidence/) supersede that dated runtime limitation.

The historical paired replay also loaded our sealed 948801 package as a reference
**after training**. Reference weights and public trajectories are not bundled
here. The supplied source is sufficient to train a fresh model; replaying the
exact historical two-model comparison additionally requires the historical
reference artifact. Full private-set evaluation is available only to organizers.
